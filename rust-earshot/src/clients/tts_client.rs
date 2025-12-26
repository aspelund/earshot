//! WebSocket client for TTS server
//!
//! Protocol (batch mode):
//! - Client sends: JSON {"type": "synthesize", "text": "...", "request_id": "..."}
//! - Client sends: JSON {"type": "cancel", "request_id": "..."} to abort
//! - Server sends: Binary WAV audio on success
//! - Server sends: JSON {"type": "error"/"cancelled", ...} on failure
//!
//! Protocol (streaming mode):
//! - Client sends: JSON {"type": "synthesize", "text": "...", "request_id": "...", "stream": true}
//! - Server sends: JSON {"type": "stream_start", "sample_rate": 32000, ...}
//! - Server sends: Binary [4-byte u32 chunk_index][f32 PCM samples...] (multiple)
//! - Server sends: JSON {"type": "stream_end", "total_chunks": N, ...}

use anyhow::{anyhow, Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    connect_async, tungstenite::protocol::Message, MaybeTlsStream, WebSocketStream,
};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::config::TtsConfig;

/// TTS synthesize request (batch mode)
#[derive(Debug, Serialize)]
struct SynthesizeRequest {
    #[serde(rename = "type")]
    request_type: &'static str,
    text: String,
    request_id: String,
}

/// TTS synthesize request (streaming mode)
#[derive(Debug, Serialize)]
struct StreamingSynthesizeRequest {
    #[serde(rename = "type")]
    request_type: &'static str,
    text: String,
    request_id: String,
    stream: bool,
}

/// TTS cancel request
#[derive(Debug, Serialize)]
struct CancelRequest {
    #[serde(rename = "type")]
    request_type: &'static str,
    request_id: String,
}

/// TTS response status
#[derive(Debug, Deserialize)]
struct TtsResponse {
    #[serde(rename = "type")]
    response_type: String,
    error: Option<String>,
    request_id: Option<String>,
}

/// Result of TTS synthesis (batch mode)
#[derive(Debug)]
pub enum TtsResult {
    /// Successful synthesis - WAV audio bytes
    Audio(Vec<u8>),
    /// Request was cancelled
    Cancelled,
    /// Error occurred
    Error(String),
}

/// Events from streaming TTS synthesis
#[derive(Debug)]
pub enum TtsStreamEvent {
    /// Stream started with metadata
    Started {
        sample_rate: u32,
    },
    /// Audio chunk received
    Chunk {
        index: u32,
        samples: Vec<f32>,
    },
    /// Stream completed successfully
    Completed {
        total_chunks: u32,
    },
    /// Stream was cancelled
    Cancelled,
    /// Error occurred
    Error(String),
}

/// WebSocket client for TTS server
pub struct TtsClient {
    url: String,
    auth_token: Option<String>,
    ws: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
    current_request_id: Option<String>,
}

impl TtsClient {
    /// Create a new TTS client
    pub fn new(cfg: &TtsConfig) -> Self {
        Self {
            url: cfg.url.clone(),
            auth_token: cfg.auth_token.clone(),
            ws: None,
            current_request_id: None,
        }
    }

    /// Connect to the TTS server
    pub async fn connect(&mut self) -> Result<()> {
        info!("Connecting to TTS server at {}", self.url);

        let (ws_stream, _) = connect_async(&self.url)
            .await
            .context("Failed to connect to TTS server")?;

        self.ws = Some(ws_stream);

        // Authenticate if token is provided
        if let Some(token) = &self.auth_token {
            if let Some(ws) = &mut self.ws {
                ws.send(Message::Text(token.clone())).await?;
                debug!("Sent authentication token");
            }
        }

        info!("Connected to TTS server");
        Ok(())
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.ws.is_some()
    }

    /// Synthesize text to audio
    ///
    /// Returns WAV audio bytes on success
    pub async fn synthesize(&mut self, text: &str) -> Result<TtsResult> {
        let ws = self
            .ws
            .as_mut()
            .ok_or_else(|| anyhow!("Not connected to TTS server"))?;

        let request_id = Uuid::new_v4().to_string();
        self.current_request_id = Some(request_id.clone());

        let request = SynthesizeRequest {
            request_type: "synthesize",
            text: text.to_string(),
            request_id: request_id.clone(),
        };

        debug!("Requesting TTS synthesis: {}...", &text[..text.len().min(50)]);

        // Send request
        ws.send(Message::Text(serde_json::to_string(&request)?))
            .await?;

        // Wait for response
        loop {
            match ws.next().await {
                Some(Ok(Message::Binary(data))) => {
                    debug!("Received {} bytes of audio", data.len());
                    self.current_request_id = None;
                    return Ok(TtsResult::Audio(data));
                }
                Some(Ok(Message::Text(text))) => {
                    let response: TtsResponse = serde_json::from_str(&text)
                        .context("Failed to parse TTS response")?;

                    self.current_request_id = None;

                    match response.response_type.as_str() {
                        "cancelled" => {
                            debug!("TTS request was cancelled");
                            return Ok(TtsResult::Cancelled);
                        }
                        "error" => {
                            let error_msg = response.error.unwrap_or_else(|| "Unknown error".to_string());
                            warn!("TTS error: {}", error_msg);
                            return Ok(TtsResult::Error(error_msg));
                        }
                        _ => {
                            warn!("Unknown TTS response type: {}", response.response_type);
                        }
                    }
                }
                Some(Ok(Message::Ping(data))) => {
                    ws.send(Message::Pong(data)).await?;
                }
                Some(Ok(Message::Pong(_))) => {}
                Some(Ok(Message::Close(_))) => {
                    self.ws = None;
                    self.current_request_id = None;
                    return Err(anyhow!("TTS server closed connection"));
                }
                Some(Ok(Message::Frame(_))) => {}
                Some(Err(e)) => {
                    self.ws = None;
                    self.current_request_id = None;
                    return Err(anyhow!("WebSocket error: {}", e));
                }
                None => {
                    self.ws = None;
                    self.current_request_id = None;
                    return Err(anyhow!("TTS server connection closed"));
                }
            }
        }
    }

    /// Cancel the current synthesis request
    pub async fn cancel(&mut self) -> Result<()> {
        if let (Some(ws), Some(request_id)) = (&mut self.ws, &self.current_request_id) {
            let request = CancelRequest {
                request_type: "cancel",
                request_id: request_id.clone(),
            };

            debug!("Cancelling TTS request: {}", request_id);
            ws.send(Message::Text(serde_json::to_string(&request)?))
                .await?;
        }
        Ok(())
    }

    /// Reconnect if disconnected
    pub async fn ensure_connected(&mut self) -> Result<()> {
        if !self.is_connected() {
            self.connect().await?;
        }
        Ok(())
    }

    /// Synthesize text to audio with streaming
    ///
    /// Calls the callback for each event as chunks arrive.
    /// Returns after stream completes, is cancelled, or errors.
    pub async fn synthesize_stream<F>(&mut self, text: &str, mut on_event: F) -> Result<()>
    where
        F: FnMut(TtsStreamEvent),
    {
        let ws = self
            .ws
            .as_mut()
            .ok_or_else(|| anyhow!("Not connected to TTS server"))?;

        let request_id = Uuid::new_v4().to_string();
        self.current_request_id = Some(request_id.clone());

        let request = StreamingSynthesizeRequest {
            request_type: "synthesize",
            text: text.to_string(),
            request_id: request_id.clone(),
            stream: true,
        };

        debug!(
            "Requesting streaming TTS synthesis: {}...",
            &text[..text.len().min(50)]
        );

        // Send request
        ws.send(Message::Text(serde_json::to_string(&request)?))
            .await?;

        // Process streaming response
        loop {
            match ws.next().await {
                Some(Ok(Message::Binary(data))) => {
                    // Parse: [4-byte u32 index][f32 samples...]
                    if data.len() < 4 {
                        warn!("Received binary message too short: {} bytes", data.len());
                        continue;
                    }

                    let index = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    let samples: Vec<f32> = data[4..]
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();

                    debug!("Received chunk {}: {} samples", index, samples.len());
                    on_event(TtsStreamEvent::Chunk { index, samples });
                }
                Some(Ok(Message::Text(text))) => {
                    let response: serde_json::Value =
                        serde_json::from_str(&text).context("Failed to parse TTS response")?;

                    match response["type"].as_str() {
                        Some("stream_start") => {
                            let sample_rate =
                                response["sample_rate"].as_u64().unwrap_or(32000) as u32;
                            debug!("Stream started @ {}Hz", sample_rate);
                            on_event(TtsStreamEvent::Started { sample_rate });
                        }
                        Some("stream_end") => {
                            let total_chunks =
                                response["total_chunks"].as_u64().unwrap_or(0) as u32;
                            debug!("Stream completed: {} chunks", total_chunks);
                            on_event(TtsStreamEvent::Completed { total_chunks });
                            self.current_request_id = None;
                            return Ok(());
                        }
                        Some("cancelled") => {
                            debug!("Stream was cancelled");
                            on_event(TtsStreamEvent::Cancelled);
                            self.current_request_id = None;
                            return Ok(());
                        }
                        Some("error") => {
                            let error_msg = response["error"]
                                .as_str()
                                .unwrap_or("Unknown error")
                                .to_string();
                            warn!("Stream error: {}", error_msg);
                            on_event(TtsStreamEvent::Error(error_msg));
                            self.current_request_id = None;
                            return Ok(());
                        }
                        _ => {
                            warn!("Unknown response type: {:?}", response["type"]);
                        }
                    }
                }
                Some(Ok(Message::Ping(data))) => {
                    ws.send(Message::Pong(data)).await?;
                }
                Some(Ok(Message::Pong(_))) => {}
                Some(Ok(Message::Close(_))) => {
                    self.ws = None;
                    self.current_request_id = None;
                    on_event(TtsStreamEvent::Error("Connection closed".to_string()));
                    return Err(anyhow!("TTS server closed connection"));
                }
                Some(Ok(Message::Frame(_))) => {}
                Some(Err(e)) => {
                    self.ws = None;
                    self.current_request_id = None;
                    on_event(TtsStreamEvent::Error(e.to_string()));
                    return Err(anyhow!("WebSocket error: {}", e));
                }
                None => {
                    self.ws = None;
                    self.current_request_id = None;
                    on_event(TtsStreamEvent::Error("Connection closed".to_string()));
                    return Err(anyhow!("TTS server connection closed"));
                }
            }
        }
    }

    /// Close the connection
    pub async fn close(&mut self) -> Result<()> {
        if let Some(mut ws) = self.ws.take() {
            ws.close(None).await?;
        }
        Ok(())
    }
}
