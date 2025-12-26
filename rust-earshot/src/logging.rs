//! Async conversation logging
//!
//! Logs conversation history to JSONL files without blocking the main s2s flow.

use chrono::{DateTime, Local};
use serde::Serialize;
use std::path::PathBuf;
use tokio::fs::{self, OpenOptions};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tracing::{error, info};

/// A single log entry
#[derive(Debug, Serialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub role: String,
    pub content: String,
}

/// Handle for sending log entries (clone this for each component that needs to log)
#[derive(Clone)]
pub struct ConversationLogger {
    tx: mpsc::UnboundedSender<LogEntry>,
}

impl ConversationLogger {
    /// Log a message (non-blocking, fire-and-forget)
    pub fn log(&self, role: &str, content: &str) {
        let entry = LogEntry {
            timestamp: Local::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
            role: role.to_string(),
            content: content.to_string(),
        };
        // Ignore send errors (logger task may have stopped)
        let _ = self.tx.send(entry);
    }
}

/// Start the conversation logger
/// Returns a handle for logging and spawns a background task for file I/O
pub fn start_conversation_logger() -> ConversationLogger {
    let (tx, rx) = mpsc::unbounded_channel();

    // Spawn the background logging task
    tokio::spawn(logger_task(rx));

    ConversationLogger { tx }
}

async fn logger_task(mut rx: mpsc::UnboundedReceiver<LogEntry>) {
    // Create logs directory
    let logs_dir = PathBuf::from("logs");
    if let Err(e) = fs::create_dir_all(&logs_dir).await {
        error!("[Logger] Failed to create logs directory: {}", e);
        return;
    }

    // Generate session filename
    let session_id = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let log_path = logs_dir.join(format!("session_{}.jsonl", session_id));

    info!("[Logger] Logging conversation to {:?}", log_path);

    // Process log entries
    while let Some(entry) = rx.recv().await {
        if let Err(e) = append_entry(&log_path, &entry).await {
            error!("[Logger] Failed to write log entry: {}", e);
        }
    }
}

async fn append_entry(path: &PathBuf, entry: &LogEntry) -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;

    let json = serde_json::to_string(entry).unwrap_or_default();
    file.write_all(json.as_bytes()).await?;
    file.write_all(b"\n").await?;
    file.flush().await?;

    Ok(())
}
