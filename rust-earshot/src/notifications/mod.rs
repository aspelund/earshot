//! Notification system for external event delivery
//!
//! This module provides:
//! - TCP server accepting JSON notifications from external programs
//! - Simple FIFO queue for pending notifications
//! - Types for notification requests and responses
//!
//! # Usage
//!
//! External programs can send notifications via TCP:
//! ```bash
//! echo '{"source":"calendar","title":"Meeting in 5 minutes"}' | nc localhost 9999
//! ```

mod queue;
mod server;
mod types;

pub use queue::NotificationQueue;
pub use server::NotificationServer;
