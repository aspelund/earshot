//! HTTP/SSE client for LLM API (OpenAI-compatible)
//!
//! Protocol:
//! - POST /v1/chat/completions with JSON body
//! - Response is SSE stream: data: {...}
//! - Parse choices[0].delta.content for text chunks
//! - data: [DONE] marks end of stream

use anyhow::{Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::config::LlmConfig;

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: content.to_string(),
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: content.to_string(),
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.to_string(),
        }
    }
}

/// Request body for chat completions
#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: i32,
    stream: bool,
}

/// Delta content from streaming response
#[derive(Debug, Deserialize)]
struct Delta {
    content: Option<String>,
}

/// Choice from streaming response
#[derive(Debug, Deserialize)]
struct Choice {
    delta: Delta,
}

/// Streaming response chunk
#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<Choice>,
}

/// LLM client with streaming and cancellation support
pub struct LlmClient {
    url: String,
    model: String,
    temperature: f32,
    system_prompt: String,
    client: Client,

    // Pending sentences (complete sentences ready for TTS)
    pending_sentences: Arc<Mutex<Vec<String>>>,

    // Cancellation
    generation: Arc<AtomicU64>,
    cancelled: Arc<AtomicBool>,

    // State
    is_generating: Arc<AtomicBool>,
}

impl LlmClient {
    /// Create a new LLM client
    pub fn new(cfg: &LlmConfig) -> Self {
        Self {
            url: cfg.url.clone(),
            model: cfg.model.clone(),
            temperature: cfg.temperature,
            system_prompt: cfg.system_prompt.clone(),
            client: Client::new(),
            pending_sentences: Arc::new(Mutex::new(Vec::new())),
            generation: Arc::new(AtomicU64::new(0)),
            cancelled: Arc::new(AtomicBool::new(false)),
            is_generating: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Generate LLM response for given messages
    ///
    /// Streams the response and splits on sentence boundaries.
    /// Complete sentences are added to pending_sentences.
    pub async fn generate(&self, messages: Vec<ChatMessage>) -> Result<()> {
        self.cancelled.store(false, Ordering::SeqCst);
        self.is_generating.store(true, Ordering::SeqCst);

        let gen_at_start = self.generation.load(Ordering::SeqCst);

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: self.temperature,
            max_tokens: -1,
            stream: true,
        };

        debug!("Starting LLM generation");

        let response = self
            .client
            .post(&self.url)
            .json(&request)
            .send()
            .await
            .context("Failed to send LLM request")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            error!("LLM API error: {} - {}", status, text);
            self.is_generating.store(false, Ordering::SeqCst);
            return Err(anyhow::anyhow!("LLM API error: {}", status));
        }

        let mut stream = response.bytes_stream();
        let mut accumulated_text = String::new();
        let mut buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            // Check cancellation
            if self.cancelled.load(Ordering::SeqCst) {
                debug!("LLM generation cancelled");
                break;
            }

            // Check generation (in case abort happened mid-stream)
            if gen_at_start != self.generation.load(Ordering::SeqCst) {
                debug!("LLM generation aborted (generation mismatch)");
                break;
            }

            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    warn!("Stream error: {}", e);
                    continue;
                }
            };

            // Add to buffer and process complete lines
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete lines
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if line.is_empty() || !line.starts_with("data: ") {
                    continue;
                }

                let data_str = &line[6..]; // Remove "data: " prefix

                if data_str == "[DONE]" {
                    debug!("LLM stream complete");
                    break;
                }

                // Parse JSON
                if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data_str) {
                    if let Some(choice) = chunk.choices.first() {
                        if let Some(content) = &choice.delta.content {
                            accumulated_text.push_str(content);

                            // Check for sentence boundaries and emit complete sentences
                            self.extract_and_emit_sentences(&mut accumulated_text, gen_at_start)
                                .await;
                        }
                    }
                }
            }
        }

        // Emit any remaining text as final sentence
        if !accumulated_text.trim().is_empty() {
            if !self.cancelled.load(Ordering::SeqCst)
                && gen_at_start == self.generation.load(Ordering::SeqCst)
            {
                let mut pending = self.pending_sentences.lock().await;
                pending.push(accumulated_text.trim().to_string());
                debug!("Added final sentence: {}", accumulated_text.trim());
            }
        }

        self.is_generating.store(false, Ordering::SeqCst);
        debug!("LLM generation finished");

        Ok(())
    }

    /// Extract complete sentences and add to pending
    async fn extract_and_emit_sentences(&self, text: &mut String, gen_at_start: u64) {
        // Find sentence boundaries (. ! ?)
        let mut last_boundary = 0;

        for (i, c) in text.char_indices() {
            if c == '.' || c == '!' || c == '?' {
                // Check if this is followed by a space or end of text
                let next_idx = i + c.len_utf8();
                if next_idx >= text.len()
                    || text[next_idx..].starts_with(' ')
                    || text[next_idx..].starts_with('\n')
                {
                    // Found sentence boundary
                    let sentence = text[last_boundary..=i].trim();
                    if !sentence.is_empty() {
                        // Check generation before adding
                        if gen_at_start == self.generation.load(Ordering::SeqCst) {
                            let mut pending = self.pending_sentences.lock().await;
                            pending.push(sentence.to_string());
                            debug!("Sentence complete: {}", sentence);
                        }
                    }
                    last_boundary = next_idx;
                }
            }
        }

        // Keep only the incomplete part
        *text = text[last_boundary..].to_string();
    }

    /// Get all ready sentences (clears the list)
    pub async fn get_ready_sentences(&self) -> Vec<String> {
        let mut pending = self.pending_sentences.lock().await;
        std::mem::take(&mut *pending)
    }

    /// Check if there are pending sentences
    pub async fn has_pending_sentences(&self) -> bool {
        !self.pending_sentences.lock().await.is_empty()
    }

    /// Abort current generation
    pub fn abort(&self) {
        // Increment generation (invalidates in-flight responses)
        self.generation.fetch_add(1, Ordering::SeqCst);
        // Signal cancellation
        self.cancelled.store(true, Ordering::SeqCst);
        // Clear pending sentences synchronously is tricky, we'll do it async
        debug!("LLM abort signaled");
    }

    /// Clear pending sentences
    pub async fn clear_pending(&self) {
        self.pending_sentences.lock().await.clear();
    }

    /// Check if currently generating
    pub fn is_generating(&self) -> bool {
        self.is_generating.load(Ordering::SeqCst)
    }

    /// Check if processing (generating or has pending)
    pub async fn is_processing(&self) -> bool {
        self.is_generating() || !self.pending_sentences.lock().await.is_empty()
    }

    /// Get system prompt
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }
}
