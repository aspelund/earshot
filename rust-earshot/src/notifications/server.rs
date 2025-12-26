//! TCP notification server

use anyhow::Result;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tracing::{debug, error, info, warn};

use super::queue::NotificationQueue;
use super::types::{NotificationRequest, NotificationResponse};

/// TCP server for receiving notifications from external programs
pub struct NotificationServer {
    port: u16,
    queue: NotificationQueue,
}

impl NotificationServer {
    pub fn new(port: u16, queue: NotificationQueue) -> Self {
        Self { port, queue }
    }

    /// Run the server (call in tokio::spawn)
    pub async fn run(&self, mut shutdown: watch::Receiver<bool>) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.port);
        let listener = TcpListener::bind(&addr).await?;
        info!("[Notifications] Server listening on {}", addr);

        loop {
            tokio::select! {
                accept_result = listener.accept() => {
                    match accept_result {
                        Ok((socket, addr)) => {
                            debug!("[Notifications] Client connected from {}", addr);
                            let queue = self.queue.clone();
                            tokio::spawn(async move {
                                if let Err(e) = handle_client(socket, queue).await {
                                    warn!("[Notifications] Client handler error: {}", e);
                                }
                            });
                        }
                        Err(e) => {
                            error!("[Notifications] Accept error: {}", e);
                        }
                    }
                }
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        info!("[Notifications] Server shutting down");
                        break;
                    }
                }
            }
        }

        Ok(())
    }
}

async fn handle_client(socket: TcpStream, queue: NotificationQueue) -> Result<()> {
    let (reader, mut writer) = socket.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line).await?;

        if bytes_read == 0 {
            // Connection closed
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Parse JSON notification
        let response = match serde_json::from_str::<NotificationRequest>(trimmed) {
            Ok(req) => {
                let (id, position) = queue.add(req).await;

                info!(
                    "[Notifications] Received: #{} (position {})",
                    id, position
                );

                NotificationResponse {
                    success: true,
                    id,
                    error: None,
                    position,
                }
            }
            Err(e) => {
                warn!("[Notifications] Invalid JSON: {}", e);
                NotificationResponse {
                    success: false,
                    id: 0,
                    error: Some(format!("Invalid JSON: {}", e)),
                    position: 0,
                }
            }
        };

        // Send response
        let response_json = serde_json::to_string(&response)?;
        writer.write_all(response_json.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;
    }

    debug!("[Notifications] Client disconnected");
    Ok(())
}
