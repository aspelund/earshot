//! WebSocket client for Chatterbox TTS server
//!
//! Protocol:
//! - Client sends: JSON {"type": "synthesize", "text": "...", "request_id": "..."}
//! - Client sends: JSON {"type": "cancel", "request_id": "..."} to abort
//! - Server sends: Binary WAV audio on success
//! - Server sends: JSON {"type": "error"/"cancelled", ...} on failure

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

/// TTS synthesize request
#[derive(Debug, Serialize)]
struct SynthesizeRequest {
    #[serde(rename = "type")]
    request_type: &'static str,
    text: String,
    request_id: String,
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

/// Result of TTS synthesis
#[derive(Debug)]
pub enum TtsResult {
    /// Successful synthesis - WAV audio bytes
    Audio(Vec<u8>),
    /// Request was cancelled
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

    /// Close the connection
    pub async fn close(&mut self) -> Result<()> {
        if let Some(mut ws) = self.ws.take() {
            ws.close(None).await?;
        }
        Ok(())
    }
}
