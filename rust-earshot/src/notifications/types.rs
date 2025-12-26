//! Notification types for external event delivery

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Incoming notification from external programs
#[derive(Debug, Clone)]
pub struct Notification {
    /// Unique identifier (auto-incrementing)
    pub id: u64,

    /// Source application name
    pub source: String,

    /// Short title/summary
    pub title: String,

    /// Detailed body text (optional)
    pub body: String,

    /// Timestamp when received (for immediate-delivery check)
    pub received_at: Instant,

    /// Clock time when received (for display in history)
    pub received_time: DateTime<Local>,
}

/// JSON format for incoming notifications
#[derive(Debug, Deserialize)]
pub struct NotificationRequest {
    /// Source application name (required)
    pub source: String,

    /// Short title/summary (required)
    pub title: String,

    /// Detailed body text (optional)
    #[serde(default)]
    pub body: String,
}

/// Response sent back to notification client
#[derive(Debug, Serialize)]
pub struct NotificationResponse {
    pub success: bool,
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub position: usize,
}

impl Notification {
    /// Create a new notification from a request with given id
    pub fn from_request(req: NotificationRequest, id: u64) -> Self {
        Self {
            id,
            source: req.source,
            title: req.title,
            body: req.body,
            received_at: Instant::now(),
            received_time: Local::now(),
        }
    }

    /// Format timestamp as HH:MM:SS
    pub fn timestamp_str(&self) -> String {
        self.received_time.format("%H:%M:%S").to_string()
    }
}
