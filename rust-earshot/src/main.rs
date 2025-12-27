//! Earshot - Real-time conversational AI client
//!
//! Handles: Audio I/O, VAD, Segmentation, LLM/STT/TTS client coordination
//! Connects to: Parakeet (STT) and Chatterbox (TTS) Python servers

// Hide console window on Windows
#![windows_subsystem = "windows"]

mod audio;
mod clients;
mod config;
mod gui;
mod logging;
mod notifications;
mod pipeline;
mod vad;

use anyhow::Result;
use std::sync::Arc;
use std::thread;
use tracing::{info, Level};
use tracing_subscriber::fmt::writer::MakeWriterExt;

use crate::gui::{EarshotApp, GuiState};

fn main() -> Result<()> {
    // Initialize file logging (logs to earshot.log in current directory)
    let file_appender = tracing_appender::rolling::never(".", "earshot.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_writer(non_blocking.with_max_level(Level::INFO))
        .with_ansi(false)
        .init();

    info!("Earshot - Real-time Conversational AI");
    info!("Loading configuration...");

    let cfg = config::load_config("config.yaml")?;

    // Create shared GUI state
    let gui_state = GuiState::new();
    let gui_state_for_pipeline = Arc::clone(&gui_state);

    // Spawn tokio runtime on a separate thread
    let pipeline_thread = thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");

        rt.block_on(async {
            info!("Starting pipeline...");
            if let Err(e) = pipeline::run_with_gui(cfg, gui_state_for_pipeline).await {
                tracing::error!("Pipeline error: {}", e);
            }
        });
    });

    // Run GUI on main thread (required by some platforms)
    // Try wgpu first, fall back to glow if GPU not available (e.g., WSL2)
    let renderer = if std::env::var("EARSHOT_USE_GLOW").is_ok() {
        info!("Using Glow (OpenGL) renderer");
        eframe::Renderer::Glow
    } else {
        info!("Using wgpu renderer");
        eframe::Renderer::Wgpu
    };

    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([800.0, 700.0])
            .with_min_inner_size([600.0, 500.0])
            .with_title("Earshot - Voice Assistant"),
        renderer,
        ..Default::default()
    };

    eframe::run_native(
        "Earshot",
        native_options,
        Box::new(move |cc| Ok(Box::new(EarshotApp::new(cc, gui_state)))),
    )
    .map_err(|e| anyhow::anyhow!("GUI error: {}", e))?;

    // Wait for pipeline to finish (when GUI closes, pipeline should exit)
    let _ = pipeline_thread.join();

    Ok(())
}
