//! Conversational pipeline - coordinates VAD, STT, LLM, TTS, and audio
//!
//! State machine:
//! - IDLE: Listening for speech, VAD running
//! - PROCESSING: STT → LLM → TTS pipeline active
//!
//! Interrupt: Speech detected during PROCESSING → abort and reset

use anyhow::Result;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex, watch};
use tracing::{debug, error, info, warn};

use crate::audio::{AudioCapture, AudioPlayer};
use crate::clients::{ChatMessage, LlmClient, SttClient, TtsClient, TtsResult, TtsStreamEvent};
use crate::config::Config;
use crate::gui::{GuiCommand, GuiState, PipelineState};
use crate::logging::{start_conversation_logger, ConversationLogger};
use crate::notifications::{NotificationQueue, NotificationServer};
use crate::vad::{Segmentor, SileroVad, SpeechSegment};

/// Pipeline state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Idle,
    Processing,
}

/// Conversation message
#[derive(Debug, Clone)]
struct Message {
    role: String,
    content: String,
}

/// Run the conversational pipeline
pub async fn run(cfg: Config) -> Result<()> {
    info!("Initializing pipeline...");

    // Initialize VAD
    info!("Loading VAD model from {}", cfg.vad.model_path);
    let mut vad = SileroVad::new(&cfg.vad.model_path, cfg.audio.sample_rate)?;
    info!("VAD model loaded");

    // Initialize segmentor
    let mut segmentor = Segmentor::new(cfg.audio.sample_rate, cfg.audio.frame_ms, &cfg.vad);

    // Initialize audio capture
    let mut audio_capture = AudioCapture::new(&cfg.audio)?;
    audio_capture.start()?;

    // Initialize audio player
    let audio_player = Arc::new(AudioPlayer::new(24000)?); // TTS outputs 24kHz
    audio_player.start()?;

    // Initialize clients
    let stt_client = Arc::new(Mutex::new(SttClient::new(&cfg.stt)));
    let tts_client = Arc::new(Mutex::new(TtsClient::new(&cfg.tts)));
    let llm_client = Arc::new(LlmClient::new(&cfg.llm));

    // Connect to servers
    {
        let mut stt = stt_client.lock().await;
        stt.connect().await?;
    }
    {
        let mut tts = tts_client.lock().await;
        tts.connect().await?;
    }

    // State
    let state = Arc::new(Mutex::new(State::Idle));
    let conversation_history = Arc::new(Mutex::new(Vec::<Message>::new()));
    let current_assistant_sentences = Arc::new(Mutex::new(Vec::<String>::new()));
    let generation = Arc::new(AtomicU64::new(0));

    // Pipeline tracking - counts sentences in each stage
    let sentences_sent_to_tts = Arc::new(AtomicUsize::new(0));
    let sentences_audio_queued = Arc::new(AtomicUsize::new(0));

    // Cancellation
    let (cancel_tx, _cancel_rx) = watch::channel(false);
    let cancelled = Arc::new(AtomicBool::new(false));

    // Notification system
    let last_speech_time = Arc::new(Mutex::new(Instant::now()));
    let notification_queue = NotificationQueue::new();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Conversation logger (async, non-blocking)
    let logger = start_conversation_logger();

    // Channels for segment processing
    let (segment_tx, mut segment_rx) = mpsc::channel::<SpeechSegment>(16);

    // Speech detection tracking
    let mut was_in_speech = false;

    info!("Pipeline initialized, listening...");
    info!("Speak to chat with the AI assistant. Press Ctrl+C to stop.\n");

    // Spawn transcription handler
    let stt_client_clone = stt_client.clone();
    let state_clone = state.clone();
    let llm_client_clone = llm_client.clone();
    let conversation_history_clone = conversation_history.clone();
    let current_assistant_sentences_clone = current_assistant_sentences.clone();
    let generation_clone = generation.clone();
    let cancelled_clone = cancelled.clone();
    let logger_clone = logger.clone();

    let transcription_task = tokio::spawn(async move {
        while let Some(segment) = segment_rx.recv().await {
            if let Err(e) = handle_segment(
                segment,
                &stt_client_clone,
                &state_clone,
                &llm_client_clone,
                &conversation_history_clone,
                &current_assistant_sentences_clone,
                &generation_clone,
                &cancelled_clone,
                &logger_clone,
            )
            .await
            {
                error!("Error handling segment: {}", e);
            }
        }
    });

    // Spawn LLM→TTS→Audio forwarder
    let llm_client_clone = llm_client.clone();
    let tts_client_clone = tts_client.clone();
    let audio_player_clone = audio_player.clone();
    let current_assistant_sentences_clone = current_assistant_sentences.clone();
    let cancelled_clone = cancelled.clone();
    let sentences_sent_to_tts_clone = sentences_sent_to_tts.clone();
    let sentences_audio_queued_clone = sentences_audio_queued.clone();

    let llm_tts_task = tokio::spawn(async move {
        loop {
            if cancelled_clone.load(Ordering::SeqCst) {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                continue;
            }

            let sentences = llm_client_clone.get_ready_sentences().await;
            for sentence in sentences {
                if cancelled_clone.load(Ordering::SeqCst) {
                    break;
                }

                // Safe truncation - find valid char boundary
                let display_end = {
                    let max = 60.min(sentence.len());
                    let mut end = max;
                    while end > 0 && !sentence.is_char_boundary(end) {
                        end -= 1;
                    }
                    end
                };
                info!("[LLM→TTS] {}", &sentence[..display_end]);

                // Track sentence for history
                current_assistant_sentences_clone.lock().await.push(sentence.clone());

                // Track that we're sending to TTS
                sentences_sent_to_tts_clone.fetch_add(1, Ordering::SeqCst);

                // Send to TTS with streaming
                let stream_result = {
                    let mut tts = tts_client_clone.lock().await;
                    if let Err(e) = tts.ensure_connected().await {
                        error!("TTS connection error: {}", e);
                        sentences_audio_queued_clone.fetch_add(1, Ordering::SeqCst);
                        continue;
                    }

                    // Streaming TTS with callback
                    let audio_player = audio_player_clone.clone();
                    let mut sample_rate = 32000u32;
                    let mut first_chunk = true;
                    let tts_start = Instant::now();

                    let result = tts
                        .synthesize_stream(&sentence, |event| {
                            match event {
                                TtsStreamEvent::Started { sample_rate: sr } => {
                                    sample_rate = sr;
                                    debug!("[TTS] Stream started @ {}Hz", sr);
                                }
                                TtsStreamEvent::Chunk { index, samples } => {
                                    if first_chunk {
                                        let first_chunk_received = tts_start.elapsed();
                                        info!("[TTS] first_chunk_received: {:.1}ms", first_chunk_received.as_secs_f64() * 1000.0);

                                        audio_player.enqueue_pcm_f32(samples, sample_rate);

                                        let first_chunk_enqueued = tts_start.elapsed();
                                        info!("[TTS] first_chunk_enqueued: {:.1}ms", first_chunk_enqueued.as_secs_f64() * 1000.0);

                                        // Check if playback started (is_playing becomes true after enqueue)
                                        if audio_player.is_playing() {
                                            let playback_started = tts_start.elapsed();
                                            info!("[TTS] playback_started: {:.1}ms", playback_started.as_secs_f64() * 1000.0);
                                        }

                                        first_chunk = false;
                                    } else {
                                        audio_player.enqueue_pcm_f32(samples, sample_rate);
                                    }
                                }
                                TtsStreamEvent::Completed { total_chunks } => {
                                    let total_time = tts_start.elapsed();
                                    info!(
                                        "[TTS→Audio] Stream complete ({} chunks in {:.1}ms)",
                                        total_chunks,
                                        total_time.as_secs_f64() * 1000.0
                                    );
                                }
                                TtsStreamEvent::Cancelled => {
                                    debug!("[TTS] Stream cancelled");
                                }
                                TtsStreamEvent::Error(e) => {
                                    error!("[TTS] Stream error: {}", e);
                                }
                            }
                        })
                        .await;

                    // If streaming failed, try reconnecting and fall back to batch mode
                    if let Err(e) = result {
                        warn!("TTS streaming error, trying batch mode: {}", e);
                        if let Err(e) = tts.connect().await {
                            error!("TTS reconnection failed: {}", e);
                            Err(e)
                        } else {
                            // Fall back to batch mode
                            match tts.synthesize(&sentence).await {
                                Ok(TtsResult::Audio(wav_bytes)) => {
                                    info!("[TTS→Audio] Playing {} bytes (batch fallback)", wav_bytes.len());
                                    if let Err(e) = audio_player_clone.enqueue_wav(&wav_bytes) {
                                        error!("Failed to enqueue audio: {}", e);
                                    }
                                    Ok(())
                                }
                                Ok(TtsResult::Cancelled) => {
                                    debug!("TTS cancelled (batch fallback)");
                                    Ok(())
                                }
                                Ok(TtsResult::Error(e)) => {
                                    error!("TTS error (batch fallback): {}", e);
                                    Ok(())
                                }
                                Err(e) => Err(e),
                            }
                        }
                    } else {
                        Ok(())
                    }
                };

                // Track that audio is queued regardless of result
                sentences_audio_queued_clone.fetch_add(1, Ordering::SeqCst);

                if let Err(e) = stream_result {
                    error!("TTS failed completely: {}", e);
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    });

    // Spawn state monitor
    let state_clone = state.clone();
    let llm_client_clone = llm_client.clone();
    let audio_player_clone = audio_player.clone();
    let conversation_history_clone = conversation_history.clone();
    let current_assistant_sentences_clone = current_assistant_sentences.clone();
    let sentences_sent_to_tts_clone = sentences_sent_to_tts.clone();
    let sentences_audio_queued_clone = sentences_audio_queued.clone();
    let logger_clone = logger.clone();

    let state_monitor_task = tokio::spawn(async move {
        loop {
            let current_state = *state_clone.lock().await;

            if current_state == State::Processing {
                let llm_done = !llm_client_clone.is_processing().await;
                let sent = sentences_sent_to_tts_clone.load(Ordering::SeqCst);
                let queued = sentences_audio_queued_clone.load(Ordering::SeqCst);
                let tts_done = sent == queued; // All TTS requests completed
                let audio_done = !audio_player_clone.is_playing();

                if llm_done && tts_done && audio_done {
                    // Commit sentences to history
                    let sentences = current_assistant_sentences_clone.lock().await.clone();
                    if !sentences.is_empty() {
                        let content = sentences.join(" ");
                        // Log assistant message
                        logger_clone.log("assistant", &content);
                        conversation_history_clone.lock().await.push(Message {
                            role: "assistant".to_string(),
                            content,
                        });
                    }
                    current_assistant_sentences_clone.lock().await.clear();

                    // Reset pipeline counters
                    sentences_sent_to_tts_clone.store(0, Ordering::SeqCst);
                    sentences_audio_queued_clone.store(0, Ordering::SeqCst);

                    info!("[State] → IDLE");
                    *state_clone.lock().await = State::Idle;
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
    });

    // Spawn notification server (if enabled)
    if cfg.notifications.enabled {
        let notification_queue_clone = notification_queue.clone();
        let port = cfg.notifications.port;
        let shutdown_rx_clone = shutdown_rx.clone();

        let _notification_server_task = tokio::spawn(async move {
            let server = NotificationServer::new(port, notification_queue_clone);
            if let Err(e) = server.run(shutdown_rx_clone).await {
                error!("[Notifications] Server error: {}", e);
            }
        });
        info!("[Notifications] Server starting on port {}", port);
    }

    // Spawn notification dispatcher
    let notification_queue_clone = notification_queue.clone();
    let state_clone = state.clone();
    let last_speech_time_clone = last_speech_time.clone();
    let llm_client_clone = llm_client.clone();
    let conversation_history_clone = conversation_history.clone();
    let current_assistant_sentences_clone = current_assistant_sentences.clone();
    let cancelled_clone = cancelled.clone();
    let logger_clone2 = logger.clone();
    let idle_poll_threshold = cfg.notifications.idle_poll_threshold_s;
    let idle_immediate_threshold = cfg.notifications.idle_immediate_threshold_s;
    let batch_grace_period = cfg.notifications.batch_grace_period_ms;
    let notification_prompt = cfg.notifications.system_prompt.clone();

    let _notification_dispatcher_task = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;

            // Only act if IDLE
            let current_state = *state_clone.lock().await;
            if current_state != State::Idle {
                continue;
            }

            // Check if queue is empty
            if notification_queue_clone.is_empty().await {
                continue;
            }

            // Calculate idle time
            let idle_secs = last_speech_time_clone.lock().await.elapsed().as_secs();

            // Immediate trigger: new notification arrived recently + idle > 5s
            let has_recent = notification_queue_clone.has_recent(Duration::from_secs(2)).await;
            let immediate = has_recent && idle_secs >= idle_immediate_threshold;

            // Poll trigger: idle > 10s
            let poll = idle_secs >= idle_poll_threshold;

            if !immediate && !poll {
                continue;
            }

            // GRACE PERIOD: Wait to batch incoming notifications
            tokio::time::sleep(Duration::from_millis(batch_grace_period)).await;

            // Re-check we're still IDLE after grace period (user might have started speaking)
            let current_state = *state_clone.lock().await;
            if current_state != State::Idle {
                continue;
            }

            // BATCH: Drain ALL notifications from queue
            let notifications = notification_queue_clone.drain_all().await;
            if notifications.is_empty() {
                continue;
            }

            let notification_count = notifications.len();
            info!(
                "[Notifications] Delivering {} notification(s)",
                notification_count
            );

            // Transition to Processing
            *state_clone.lock().await = State::Processing;
            cancelled_clone.store(false, Ordering::SeqCst);
            current_assistant_sentences_clone.lock().await.clear();

            // Build user message with notification(s)
            let user_msg = if notifications.len() == 1 {
                let n = &notifications[0];
                let content = if n.body.is_empty() {
                    n.title.clone()
                } else {
                    format!("{} - {}", n.title, n.body)
                };
                format!(
                    "Info from system - a new notification to the user has come:\n\n\
                     Time: {}\n\
                     From: {}\n\
                     Content: {}\n\n\
                     Please inform the user about the message.",
                    n.timestamp_str(),
                    n.source,
                    content
                )
            } else {
                // Multiple notifications
                let notification_list: Vec<String> = notifications
                    .iter()
                    .map(|n| {
                        let content = if n.body.is_empty() {
                            n.title.clone()
                        } else {
                            format!("{} - {}", n.title, n.body)
                        };
                        format!(
                            "Time: {}\nFrom: {}\nContent: {}",
                            n.timestamp_str(),
                            n.source,
                            content
                        )
                    })
                    .collect();
                format!(
                    "Info from system - {} new notifications to the user have come:\n\n{}\n\n\
                     Please inform the user about these messages.",
                    notification_count,
                    notification_list.join("\n\n")
                )
            };

            // System prompt is just instructions (no embedded notification)
            let mut messages = vec![ChatMessage::system(&notification_prompt)];

            // Add recent conversation history for context (last 4 exchanges = 8 messages)
            {
                let history = conversation_history_clone.lock().await;
                let recent: Vec<_> = history.iter().rev().take(8).collect();
                for msg in recent.into_iter().rev() {
                    messages.push(ChatMessage {
                        role: msg.role.clone(),
                        content: msg.content.clone(),
                    });
                }
            }

            // Add notification as user message
            messages.push(ChatMessage::user(&user_msg));

            // Log and add to conversation history
            logger_clone2.log("notification", &user_msg);
            conversation_history_clone.lock().await.push(Message {
                role: "user".to_string(),
                content: user_msg,
            });

            // Start LLM generation
            info!("[LLM] Starting notification response...");
            let llm = llm_client_clone.clone();
            tokio::spawn(async move {
                if let Err(e) = llm.generate(messages).await {
                    error!("[LLM] Notification generation error: {}", e);
                }
            });
        }
    });

    // Main loop - process audio frames
    let mut frame_count = 0u64;

    loop {
        // Get next audio frame
        let frame = match audio_capture.try_next_frame() {
            Some(f) => f,
            None => {
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                continue;
            }
        };

        frame_count += 1;

        // Process VAD
        let prob = match vad.process(&frame) {
            Ok(p) => p,
            Err(e) => {
                warn!("VAD error: {}", e);
                continue;
            }
        };

        // Convert to PCM16 for segmentor
        let pcm16: Vec<i16> = frame.iter().map(|&s| (s * 32767.0) as i16).collect();

        // Update segmentor
        if let Some(segment) = segmentor.update(pcm16, prob) {
            info!(
                "[VAD] Segment: {} ms, {} bytes",
                segment.duration_ms,
                segment.audio_bytes.len()
            );

            // Send segment for processing
            if segment_tx.send(segment).await.is_err() {
                error!("Failed to send segment to handler");
            }
        }

        // Interrupt detection
        let in_speech = segmentor.is_in_speech();
        if in_speech && !was_in_speech {
            let current_state = *state.lock().await;
            if current_state == State::Processing {
                info!("[Interrupt] Speech detected during processing!");

                // Bump generation
                generation.fetch_add(1, Ordering::SeqCst);

                // Signal cancellation
                cancelled.store(true, Ordering::SeqCst);

                // Abort LLM
                llm_client.abort();

                // Cancel TTS
                if let Ok(mut tts) = tts_client.try_lock() {
                    let _ = tts.cancel().await;
                }

                // Stop audio playback
                audio_player.fade_out_and_stop();

                // Brief pause for cleanup
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

                // Clear LLM pending
                llm_client.clear_pending().await;

                // Reset pipeline counters
                sentences_sent_to_tts.store(0, Ordering::SeqCst);
                sentences_audio_queued.store(0, Ordering::SeqCst);

                // Reset cancellation for next request
                cancelled.store(false, Ordering::SeqCst);
            }
        }

        // Update last speech time when speech ends (for notification idle tracking)
        if was_in_speech && !in_speech {
            *last_speech_time.lock().await = Instant::now();
        }
        was_in_speech = in_speech;

        // Heartbeat logging
        if frame_count % 3000 == 0 {
            let current_state = *state.lock().await;
            info!(
                "[Heartbeat] frames={}, state={:?}, in_speech={}, prob_ema={:.3}",
                frame_count,
                current_state,
                in_speech,
                segmentor.prob_ema()
            );
        }

        // Small yield to allow other tasks to run
        if frame_count % 100 == 0 {
            tokio::task::yield_now().await;
        }
    }
}

/// Handle a speech segment
async fn handle_segment(
    segment: SpeechSegment,
    stt_client: &Arc<Mutex<SttClient>>,
    state: &Arc<Mutex<State>>,
    llm_client: &Arc<LlmClient>,
    conversation_history: &Arc<Mutex<Vec<Message>>>,
    current_assistant_sentences: &Arc<Mutex<Vec<String>>>,
    generation: &Arc<AtomicU64>,
    cancelled: &Arc<AtomicBool>,
    logger: &ConversationLogger,
) -> Result<()> {
    // Transcribe with retry on connection error
    let result = {
        let mut stt = stt_client.lock().await;

        // First attempt
        stt.ensure_connected().await?;
        match stt.transcribe(
            &segment.audio_bytes,
            &segment.start_iso,
            &segment.end_iso,
            Some("en"),
        ).await {
            Ok(r) => r,
            Err(e) => {
                // Connection might have died, try reconnecting once
                warn!("STT error, reconnecting: {}", e);
                stt.connect().await?;
                stt.transcribe(
                    &segment.audio_bytes,
                    &segment.start_iso,
                    &segment.end_iso,
                    Some("en"),
                ).await?
            }
        }
    };

    let text = result.text.trim();
    if text.is_empty() {
        return Ok(());
    }

    info!("[User] {}", text);

    let current_state = *state.lock().await;

    match current_state {
        State::Idle => {
            // Transition to processing
            *state.lock().await = State::Processing;
            info!("[State] → PROCESSING");

            // Reset for new turn
            cancelled.store(false, Ordering::SeqCst);
            current_assistant_sentences.lock().await.clear();

            // Add to history and log
            logger.log("user", text);
            conversation_history.lock().await.push(Message {
                role: "user".to_string(),
                content: text.to_string(),
            });

            // Prepare messages for LLM
            let mut messages = vec![ChatMessage::system(llm_client.system_prompt())];
            for msg in conversation_history.lock().await.iter() {
                messages.push(ChatMessage {
                    role: msg.role.clone(),
                    content: msg.content.clone(),
                });
            }

            // Start LLM generation
            info!("[LLM] Starting generation...");
            let llm = llm_client.clone();
            tokio::spawn(async move {
                if let Err(e) = llm.generate(messages).await {
                    error!("LLM generation error: {}", e);
                }
            });
        }
        State::Processing => {
            // Interrupt already handled in main loop
            // This transcription will be processed after interrupt completes
            info!("[Interrupt] Queued transcription: {}", text);
        }
    }

    Ok(())
}

/// Calculate RMS level from audio samples (returns 0.0 to 1.0)
fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    // Scale up for visibility (mic input is typically quiet)
    ((sum_sq / samples.len() as f32).sqrt() * 10.0).min(1.0)
}

/// Run the conversational pipeline with GUI integration
pub async fn run_with_gui(cfg: Config, gui_state: Arc<GuiState>) -> Result<()> {
    info!("Initializing pipeline with GUI...");

    // Initialize VAD
    info!("Loading VAD model from {}", cfg.vad.model_path);
    let mut vad = SileroVad::new(&cfg.vad.model_path, cfg.audio.sample_rate)?;
    info!("VAD model loaded");

    // Initialize segmentor
    let mut segmentor = Segmentor::new(cfg.audio.sample_rate, cfg.audio.frame_ms, &cfg.vad);

    // Initialize audio capture
    let mut audio_capture = AudioCapture::new(&cfg.audio)?;
    audio_capture.start()?;

    // Initialize audio player
    let audio_player = Arc::new(AudioPlayer::new(24000)?); // TTS outputs 24kHz
    audio_player.start()?;

    // Initialize clients
    let stt_client = Arc::new(Mutex::new(SttClient::new(&cfg.stt)));
    let tts_client = Arc::new(Mutex::new(TtsClient::new(&cfg.tts)));
    let llm_client = Arc::new(LlmClient::new(&cfg.llm));

    // Connect to servers
    {
        let mut stt = stt_client.lock().await;
        stt.connect().await?;
    }
    {
        let mut tts = tts_client.lock().await;
        tts.connect().await?;
    }

    // State
    let state = Arc::new(Mutex::new(State::Idle));
    let conversation_history = Arc::new(Mutex::new(Vec::<Message>::new()));
    let current_assistant_sentences = Arc::new(Mutex::new(Vec::<String>::new()));
    let generation = Arc::new(AtomicU64::new(0));

    // Pipeline tracking - counts sentences in each stage
    let sentences_sent_to_tts = Arc::new(AtomicUsize::new(0));
    let sentences_audio_queued = Arc::new(AtomicUsize::new(0));

    // Cancellation
    let (_cancel_tx, _cancel_rx) = watch::channel(false);
    let cancelled = Arc::new(AtomicBool::new(false));

    // Notification system
    let last_speech_time = Arc::new(Mutex::new(Instant::now()));
    let notification_queue = NotificationQueue::new();
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Conversation logger (async, non-blocking)
    let logger = start_conversation_logger();

    // Channels for segment processing
    let (segment_tx, mut segment_rx) = mpsc::channel::<SpeechSegment>(16);

    // Speech detection tracking
    let mut was_in_speech = false;
    let mut is_listening = true;

    info!("Pipeline initialized, listening...");
    info!("Speak to chat with the AI assistant. Close the window to stop.\n");

    // Spawn transcription handler
    let stt_client_clone = stt_client.clone();
    let state_clone = state.clone();
    let llm_client_clone = llm_client.clone();
    let conversation_history_clone = conversation_history.clone();
    let current_assistant_sentences_clone = current_assistant_sentences.clone();
    let generation_clone = generation.clone();
    let cancelled_clone = cancelled.clone();
    let logger_clone = logger.clone();

    let _transcription_task = tokio::spawn(async move {
        while let Some(segment) = segment_rx.recv().await {
            if let Err(e) = handle_segment(
                segment,
                &stt_client_clone,
                &state_clone,
                &llm_client_clone,
                &conversation_history_clone,
                &current_assistant_sentences_clone,
                &generation_clone,
                &cancelled_clone,
                &logger_clone,
            )
            .await
            {
                error!("Error handling segment: {}", e);
            }
        }
    });

    // Spawn LLM→TTS→Audio forwarder
    let llm_client_clone = llm_client.clone();
    let tts_client_clone = tts_client.clone();
    let audio_player_clone = audio_player.clone();
    let current_assistant_sentences_clone = current_assistant_sentences.clone();
    let cancelled_clone = cancelled.clone();
    let sentences_sent_to_tts_clone = sentences_sent_to_tts.clone();
    let sentences_audio_queued_clone = sentences_audio_queued.clone();

    let _llm_tts_task = tokio::spawn(async move {
        loop {
            if cancelled_clone.load(Ordering::SeqCst) {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                continue;
            }

            let sentences = llm_client_clone.get_ready_sentences().await;
            for sentence in sentences {
                if cancelled_clone.load(Ordering::SeqCst) {
                    break;
                }

                // Safe truncation - find valid char boundary
                let display_end = {
                    let max = 60.min(sentence.len());
                    let mut end = max;
                    while end > 0 && !sentence.is_char_boundary(end) {
                        end -= 1;
                    }
                    end
                };
                info!("[LLM→TTS] {}", &sentence[..display_end]);

                // Track sentence for history
                current_assistant_sentences_clone.lock().await.push(sentence.clone());

                // Track that we're sending to TTS
                sentences_sent_to_tts_clone.fetch_add(1, Ordering::SeqCst);

                // Send to TTS with streaming
                let stream_result = {
                    let mut tts = tts_client_clone.lock().await;
                    if let Err(e) = tts.ensure_connected().await {
                        error!("TTS connection error: {}", e);
                        sentences_audio_queued_clone.fetch_add(1, Ordering::SeqCst);
                        continue;
                    }

                    // Streaming TTS with callback
                    let audio_player = audio_player_clone.clone();
                    let mut sample_rate = 32000u32;
                    let mut first_chunk = true;
                    let tts_start = Instant::now();

                    let result = tts
                        .synthesize_stream(&sentence, |event| {
                            match event {
                                TtsStreamEvent::Started { sample_rate: sr } => {
                                    sample_rate = sr;
                                    debug!("[TTS] Stream started @ {}Hz", sr);
                                }
                                TtsStreamEvent::Chunk { index, samples } => {
                                    if first_chunk {
                                        let first_chunk_received = tts_start.elapsed();
                                        info!("[TTS] first_chunk_received: {:.1}ms", first_chunk_received.as_secs_f64() * 1000.0);

                                        audio_player.enqueue_pcm_f32(samples, sample_rate);

                                        let first_chunk_enqueued = tts_start.elapsed();
                                        info!("[TTS] first_chunk_enqueued: {:.1}ms", first_chunk_enqueued.as_secs_f64() * 1000.0);

                                        // Check if playback started (is_playing becomes true after enqueue)
                                        if audio_player.is_playing() {
                                            let playback_started = tts_start.elapsed();
                                            info!("[TTS] playback_started: {:.1}ms", playback_started.as_secs_f64() * 1000.0);
                                        }

                                        first_chunk = false;
                                    } else {
                                        audio_player.enqueue_pcm_f32(samples, sample_rate);
                                    }
                                }
                                TtsStreamEvent::Completed { total_chunks } => {
                                    let total_time = tts_start.elapsed();
                                    info!(
                                        "[TTS→Audio] Stream complete ({} chunks in {:.1}ms)",
                                        total_chunks,
                                        total_time.as_secs_f64() * 1000.0
                                    );
                                }
                                TtsStreamEvent::Cancelled => {
                                    debug!("[TTS] Stream cancelled");
                                }
                                TtsStreamEvent::Error(e) => {
                                    error!("[TTS] Stream error: {}", e);
                                }
                            }
                        })
                        .await;

                    // If streaming failed, try reconnecting and fall back to batch mode
                    if let Err(e) = result {
                        warn!("TTS streaming error, trying batch mode: {}", e);
                        if let Err(e) = tts.connect().await {
                            error!("TTS reconnection failed: {}", e);
                            Err(e)
                        } else {
                            // Fall back to batch mode
                            match tts.synthesize(&sentence).await {
                                Ok(TtsResult::Audio(wav_bytes)) => {
                                    info!("[TTS→Audio] Playing {} bytes (batch fallback)", wav_bytes.len());
                                    if let Err(e) = audio_player_clone.enqueue_wav(&wav_bytes) {
                                        error!("Failed to enqueue audio: {}", e);
                                    }
                                    Ok(())
                                }
                                Ok(TtsResult::Cancelled) => {
                                    debug!("TTS cancelled (batch fallback)");
                                    Ok(())
                                }
                                Ok(TtsResult::Error(e)) => {
                                    error!("TTS error (batch fallback): {}", e);
                                    Ok(())
                                }
                                Err(e) => Err(e),
                            }
                        }
                    } else {
                        Ok(())
                    }
                };

                // Track that audio is queued regardless of result
                sentences_audio_queued_clone.fetch_add(1, Ordering::SeqCst);

                if let Err(e) = stream_result {
                    error!("TTS failed completely: {}", e);
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    });

    // Spawn state monitor
    let state_clone = state.clone();
    let llm_client_clone = llm_client.clone();
    let audio_player_clone = audio_player.clone();
    let conversation_history_clone = conversation_history.clone();
    let current_assistant_sentences_clone = current_assistant_sentences.clone();
    let sentences_sent_to_tts_clone = sentences_sent_to_tts.clone();
    let sentences_audio_queued_clone = sentences_audio_queued.clone();
    let logger_clone = logger.clone();
    let gui_state_clone = gui_state.clone();

    let _state_monitor_task = tokio::spawn(async move {
        loop {
            let current_state = *state_clone.lock().await;

            if current_state == State::Processing {
                let llm_done = !llm_client_clone.is_processing().await;
                let sent = sentences_sent_to_tts_clone.load(Ordering::SeqCst);
                let queued = sentences_audio_queued_clone.load(Ordering::SeqCst);
                let tts_done = sent == queued; // All TTS requests completed
                let is_playing = audio_player_clone.is_playing();

                // Update output level based on actual playback amplitude
                let output_level = if is_playing {
                    audio_player_clone.current_amplitude()
                } else {
                    0.0
                };
                gui_state_clone.output_level.store(output_level);

                // Update GUI state
                if is_playing {
                    gui_state_clone.set_state(PipelineState::Speaking);
                } else if !tts_done {
                    gui_state_clone.set_state(PipelineState::Processing);
                }

                if llm_done && tts_done && !is_playing {
                    // Commit sentences to history
                    let sentences = current_assistant_sentences_clone.lock().await.clone();
                    if !sentences.is_empty() {
                        let content = sentences.join(" ");
                        // Log assistant message
                        logger_clone.log("assistant", &content);
                        conversation_history_clone.lock().await.push(Message {
                            role: "assistant".to_string(),
                            content,
                        });
                    }
                    current_assistant_sentences_clone.lock().await.clear();

                    // Reset pipeline counters
                    sentences_sent_to_tts_clone.store(0, Ordering::SeqCst);
                    sentences_audio_queued_clone.store(0, Ordering::SeqCst);

                    info!("[State] → IDLE");
                    *state_clone.lock().await = State::Idle;
                    gui_state_clone.set_state(PipelineState::Idle);
                    gui_state_clone.output_level.store(0.0);
                }
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
    });

    // Spawn notification server (if enabled)
    if cfg.notifications.enabled {
        let notification_queue_clone = notification_queue.clone();
        let port = cfg.notifications.port;
        let shutdown_rx_clone = shutdown_rx.clone();

        let _notification_server_task = tokio::spawn(async move {
            let server = NotificationServer::new(port, notification_queue_clone);
            if let Err(e) = server.run(shutdown_rx_clone).await {
                error!("[Notifications] Server error: {}", e);
            }
        });
        info!("[Notifications] Server starting on port {}", port);
    }

    // Spawn notification dispatcher
    let notification_queue_clone = notification_queue.clone();
    let state_clone = state.clone();
    let last_speech_time_clone = last_speech_time.clone();
    let llm_client_clone = llm_client.clone();
    let conversation_history_clone = conversation_history.clone();
    let current_assistant_sentences_clone = current_assistant_sentences.clone();
    let cancelled_clone = cancelled.clone();
    let logger_clone2 = logger.clone();
    let idle_poll_threshold = cfg.notifications.idle_poll_threshold_s;
    let idle_immediate_threshold = cfg.notifications.idle_immediate_threshold_s;
    let batch_grace_period = cfg.notifications.batch_grace_period_ms;
    let notification_prompt = cfg.notifications.system_prompt.clone();
    let gui_state_clone = gui_state.clone();

    let _notification_dispatcher_task = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;

            // Only act if IDLE
            let current_state = *state_clone.lock().await;
            if current_state != State::Idle {
                continue;
            }

            // Check if queue is empty
            if notification_queue_clone.is_empty().await {
                continue;
            }

            // Calculate idle time
            let idle_secs = last_speech_time_clone.lock().await.elapsed().as_secs();

            // Immediate trigger: new notification arrived recently + idle > 5s
            let has_recent = notification_queue_clone.has_recent(Duration::from_secs(2)).await;
            let immediate = has_recent && idle_secs >= idle_immediate_threshold;

            // Poll trigger: idle > 10s
            let poll = idle_secs >= idle_poll_threshold;

            if !immediate && !poll {
                continue;
            }

            // GRACE PERIOD: Wait to batch incoming notifications
            tokio::time::sleep(Duration::from_millis(batch_grace_period)).await;

            // Re-check we're still IDLE after grace period (user might have started speaking)
            let current_state = *state_clone.lock().await;
            if current_state != State::Idle {
                continue;
            }

            // BATCH: Drain ALL notifications from queue
            let notifications = notification_queue_clone.drain_all().await;
            if notifications.is_empty() {
                continue;
            }

            let notification_count = notifications.len();
            info!(
                "[Notifications] Delivering {} notification(s)",
                notification_count
            );

            // Transition to Processing
            *state_clone.lock().await = State::Processing;
            gui_state_clone.set_state(PipelineState::Processing);
            cancelled_clone.store(false, Ordering::SeqCst);
            current_assistant_sentences_clone.lock().await.clear();

            // Build user message with notification(s)
            let user_msg = if notifications.len() == 1 {
                let n = &notifications[0];
                let content = if n.body.is_empty() {
                    n.title.clone()
                } else {
                    format!("{} - {}", n.title, n.body)
                };
                format!(
                    "Info from system - a new notification to the user has come:\n\n\
                     Time: {}\n\
                     From: {}\n\
                     Content: {}\n\n\
                     Please inform the user about the message.",
                    n.timestamp_str(),
                    n.source,
                    content
                )
            } else {
                // Multiple notifications
                let notification_list: Vec<String> = notifications
                    .iter()
                    .map(|n| {
                        let content = if n.body.is_empty() {
                            n.title.clone()
                        } else {
                            format!("{} - {}", n.title, n.body)
                        };
                        format!(
                            "Time: {}\nFrom: {}\nContent: {}",
                            n.timestamp_str(),
                            n.source,
                            content
                        )
                    })
                    .collect();
                format!(
                    "Info from system - {} new notifications to the user have come:\n\n{}\n\n\
                     Please inform the user about these messages.",
                    notification_count,
                    notification_list.join("\n\n")
                )
            };

            // System prompt is just instructions (no embedded notification)
            let mut messages = vec![ChatMessage::system(&notification_prompt)];

            // Add recent conversation history for context (last 4 exchanges = 8 messages)
            {
                let history = conversation_history_clone.lock().await;
                let recent: Vec<_> = history.iter().rev().take(8).collect();
                for msg in recent.into_iter().rev() {
                    messages.push(ChatMessage {
                        role: msg.role.clone(),
                        content: msg.content.clone(),
                    });
                }
            }

            // Add notification as user message
            messages.push(ChatMessage::user(&user_msg));

            // Log and add to conversation history
            logger_clone2.log("notification", &user_msg);
            conversation_history_clone.lock().await.push(Message {
                role: "user".to_string(),
                content: user_msg,
            });

            // Start LLM generation
            info!("[LLM] Starting notification response...");
            let llm = llm_client_clone.clone();
            tokio::spawn(async move {
                if let Err(e) = llm.generate(messages).await {
                    error!("[LLM] Notification generation error: {}", e);
                }
            });
        }
    });

    // Main loop - process audio frames
    let mut frame_count = 0u64;

    loop {
        // Check for GUI commands
        if let Some(cmd) = gui_state.try_recv_command() {
            match cmd {
                GuiCommand::StopListening => {
                    info!("[GUI] Stop listening");
                    is_listening = false;
                    audio_capture.stop()?;
                    gui_state.set_state(PipelineState::Stopped);
                }
                GuiCommand::StartListening => {
                    info!("[GUI] Start listening");
                    is_listening = true;
                    audio_capture.start()?;
                    gui_state.set_state(PipelineState::Idle);
                }
                GuiCommand::Shutdown => {
                    info!("[GUI] Shutdown requested");
                    let _ = shutdown_tx.send(true);
                    return Ok(());
                }
            }
        }

        // Skip processing if not listening
        if !is_listening {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            continue;
        }

        // Get next audio frame
        let frame = match audio_capture.try_next_frame() {
            Some(f) => f,
            None => {
                tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
                continue;
            }
        };

        frame_count += 1;

        // Update GUI input level
        let input_rms = calculate_rms(&frame);
        gui_state.input_level.store(input_rms);

        // Process VAD
        let prob = match vad.process(&frame) {
            Ok(p) => p,
            Err(e) => {
                warn!("VAD error: {}", e);
                continue;
            }
        };

        // Update GUI VAD probability
        gui_state.vad_probability.store(prob);

        // Convert to PCM16 for segmentor
        let pcm16: Vec<i16> = frame.iter().map(|&s| (s * 32767.0) as i16).collect();

        // Update segmentor
        if let Some(segment) = segmentor.update(pcm16, prob) {
            info!(
                "[VAD] Segment: {} ms, {} bytes",
                segment.duration_ms,
                segment.audio_bytes.len()
            );

            // Send segment for processing
            if segment_tx.send(segment).await.is_err() {
                error!("Failed to send segment to handler");
            }
        }

        // Interrupt detection
        let in_speech = segmentor.is_in_speech();

        // Update GUI state based on speech detection
        let current_state = *state.lock().await;
        if in_speech && current_state == State::Idle {
            gui_state.set_state(PipelineState::Listening);
        } else if !in_speech && current_state == State::Idle && gui_state.state() == PipelineState::Listening {
            gui_state.set_state(PipelineState::Idle);
        }

        if in_speech && !was_in_speech {
            if current_state == State::Processing {
                info!("[Interrupt] Speech detected during processing!");

                // Bump generation
                generation.fetch_add(1, Ordering::SeqCst);

                // Signal cancellation
                cancelled.store(true, Ordering::SeqCst);

                // Abort LLM
                llm_client.abort();

                // Cancel TTS
                if let Ok(mut tts) = tts_client.try_lock() {
                    let _ = tts.cancel().await;
                }

                // Stop audio playback
                audio_player.fade_out_and_stop();

                // Brief pause for cleanup
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

                // Clear LLM pending
                llm_client.clear_pending().await;

                // Reset pipeline counters
                sentences_sent_to_tts.store(0, Ordering::SeqCst);
                sentences_audio_queued.store(0, Ordering::SeqCst);

                // Reset cancellation for next request
                cancelled.store(false, Ordering::SeqCst);
            }
        }

        // Update last speech time when speech ends (for notification idle tracking)
        if was_in_speech && !in_speech {
            *last_speech_time.lock().await = Instant::now();
        }
        was_in_speech = in_speech;

        // Heartbeat logging
        if frame_count % 3000 == 0 {
            let current_state = *state.lock().await;
            info!(
                "[Heartbeat] frames={}, state={:?}, in_speech={}, prob_ema={:.3}",
                frame_count,
                current_state,
                in_speech,
                segmentor.prob_ema()
            );
        }

        // Small yield to allow other tasks to run
        if frame_count % 100 == 0 {
            tokio::task::yield_now().await;
        }
    }
}
