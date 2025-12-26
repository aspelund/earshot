//! Simple FIFO notification queue

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::debug;

use super::types::{Notification, NotificationRequest};

/// Thread-safe FIFO notification queue
#[derive(Clone)]
pub struct NotificationQueue {
    queue: Arc<Mutex<VecDeque<Notification>>>,
    next_id: Arc<AtomicU64>,
}

impl NotificationQueue {
    pub fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Add notification request to the queue
    /// Returns (id, position)
    pub async fn add(&self, request: NotificationRequest) -> (u64, usize) {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let notification = Notification::from_request(request, id);
        let mut queue = self.queue.lock().await;
        queue.push_back(notification);
        let position = queue.len();
        debug!("Notification {} queued at position {}", id, position);
        (id, position)
    }

    /// Add notification to the back of the queue
    /// Returns queue position (1-indexed)
    pub async fn push(&self, notification: Notification) -> usize {
        let mut queue = self.queue.lock().await;
        queue.push_back(notification);
        let position = queue.len();
        debug!("Notification queued at position {}", position);
        position
    }

    /// Pop the next notification from the front of the queue
    pub async fn pop(&self) -> Option<Notification> {
        self.queue.lock().await.pop_front()
    }

    /// Peek at the next notification without removing
    pub async fn peek(&self) -> Option<Notification> {
        self.queue.lock().await.front().cloned()
    }

    /// Check if queue is empty
    pub async fn is_empty(&self) -> bool {
        self.queue.lock().await.is_empty()
    }

    /// Get queue length
    pub async fn len(&self) -> usize {
        self.queue.lock().await.len()
    }

    /// Check if any notification arrived within the given duration
    /// Used for immediate-delivery trigger
    pub async fn has_recent(&self, within: Duration) -> bool {
        let queue = self.queue.lock().await;
        let now = Instant::now();
        queue
            .iter()
            .any(|n| now.duration_since(n.received_at) < within)
    }

    /// Clear all notifications
    pub async fn clear(&self) {
        self.queue.lock().await.clear();
    }

    /// Drain all notifications from queue (for batch delivery)
    pub async fn drain_all(&self) -> Vec<Notification> {
        let mut queue = self.queue.lock().await;
        queue.drain(..).collect()
    }
}

impl Default for NotificationQueue {
    fn default() -> Self {
        Self::new()
    }
}
