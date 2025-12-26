//! Earshot - Real-time conversational AI client
//!
//! Handles: Audio I/O, VAD, Segmentation, LLM/STT/TTS client coordination
//! Connects to: Parakeet (STT) and Chatterbox (TTS) Python servers

mod audio;
mod clients;
mod config;
mod logging;
mod notifications;
mod pipeline;
mod vad;

use anyhow::Result;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    info!("Earshot - Real-time Conversational AI");
    info!("Loading configuration...");

    let cfg = config::load_config("config.yaml")?;

    info!("Starting pipeline...");
    pipeline::run(cfg).await?;

    Ok(())
}
