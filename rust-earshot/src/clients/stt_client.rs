//! WebSocket client for Parakeet STT server
//!
//! Protocol:
//! - Client sends: [4-byte header len (BE)][JSON header][PCM16 audio]
//! - Server sends: JSON transcription result

use anyhow::{anyhow, Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    connect_async, tungstenite::protocol::Message, MaybeTlsStream, WebSocketStream,
};
use tracing::{debug, error, info, warn};

use crate::config::SttConfig;

/// STT request header
#[derive(Debug, Serialize)]
struct SttRequestHeader {
    start_utc: String,
    end_utc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
}

/// Word-level timestamp from transcription
#[derive(Debug, Clone, Deserialize)]
pub struct WordTimestamp {
    pub w: String,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
}

/// STT transcription result
#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionResult {
    #[serde(rename = "type")]
    pub result_type: Option<String>,
    pub start_utc: Option<String>,
    pub end_utc: Option<String>,
    pub latency_s: Option<f64>,
    pub text: String,
    #[serde(default)]
    pub tokens: Vec<WordTimestamp>,
}

/// WebSocket client for STT server
pub struct SttClient {
    url: String,
    auth_token: Option<String>,
    ws: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
}

impl SttClient {
    /// Create a new STT client
    pub fn new(cfg: &SttConfig) -> Self {
        Self {
            url: cfg.url.clone(),
            auth_token: cfg.auth_token.clone(),
            ws: None,
        }
    }

    /// Connect to the STT server
    pub async fn connect(&mut self) -> Result<()> {
        info!("Connecting to STT server at {}", self.url);

        let (ws_stream, _) = connect_async(&self.url)
            .await
            .context("Failed to connect to STT server")?;

        self.ws = Some(ws_stream);

        // Authenticate if token is provided
        if let Some(token) = &self.auth_token {
            if let Some(ws) = &mut self.ws {
                ws.send(Message::Text(token.clone())).await?;
                debug!("Sent authentication token");
            }
        }

        info!("Connected to STT server");
        Ok(())
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.ws.is_some()
    }

    /// Transcribe audio segment
    ///
    /// # Arguments
    /// * `audio_bytes` - PCM16 audio data (little-endian)
    /// * `start_iso` - Segment start time (ISO-8601)
    /// * `end_iso` - Segment end time (ISO-8601)
    /// * `language` - Optional language hint (e.g., "en")
    pub async fn transcribe(
        &mut self,
        audio_bytes: &[u8],
        start_iso: &str,
        end_iso: &str,
        language: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let ws = self
            .ws
            .as_mut()
            .ok_or_else(|| anyhow!("Not connected to STT server"))?;

        // Build header
        let header = SttRequestHeader {
            start_utc: start_iso.to_string(),
            end_utc: end_iso.to_string(),
            language: language.map(String::from),
        };

        let header_json = serde_json::to_vec(&header)?;
        let header_len = header_json.len() as u32;

        // Build message: [4-byte header len (BE)][JSON header][PCM16 audio]
        let mut message = Vec::with_capacity(4 + header_json.len() + audio_bytes.len());
        message.extend_from_slice(&header_len.to_be_bytes());
        message.extend_from_slice(&header_json);
        message.extend_from_slice(audio_bytes);

        let duration_s = audio_bytes.len() as f64 / 2.0 / 16000.0;
        debug!("Sending audio segment: {:.2}s", duration_s);

        // Send request
        ws.send(Message::Binary(message)).await?;

        // Wait for response
        loop {
            match ws.next().await {
                Some(Ok(Message::Text(text))) => {
                    let result: TranscriptionResult = serde_json::from_str(&text)
                        .context("Failed to parse STT response")?;
                    debug!("Transcription: {}", result.text);
                    return Ok(result);
                }
                Some(Ok(Message::Binary(_))) => {
                    warn!("Received unexpected binary message from STT server");
                }
                Some(Ok(Message::Ping(data))) => {
                    ws.send(Message::Pong(data)).await?;
                }
                Some(Ok(Message::Pong(_))) => {}
                Some(Ok(Message::Close(_))) => {
                    self.ws = None;
                    return Err(anyhow!("STT server closed connection"));
                }
                Some(Ok(Message::Frame(_))) => {}
                Some(Err(e)) => {
                    self.ws = None;
                    return Err(anyhow!("WebSocket error: {}", e));
                }
                None => {
                    self.ws = None;
                    return Err(anyhow!("STT server connection closed"));
                }
            }
        }
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
