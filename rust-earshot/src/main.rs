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
    // Default to wgpu (DX12 on Windows) for GPU-accelerated visualization
    // Set EARSHOT_USE_GLOW=1 to fall back to OpenGL if wgpu has issues
    let use_glow = std::env::var("EARSHOT_USE_GLOW").is_ok();

    let native_options = if use_glow {
        info!("Using Glow (OpenGL) renderer");
        eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default()
                .with_inner_size([800.0, 700.0])
                .with_min_inner_size([600.0, 500.0])
                .with_title("Earshot - Voice Assistant"),
            renderer: eframe::Renderer::Glow,
            ..Default::default()
        }
    } else {
        info!("Using wgpu renderer (DX12/Vulkan)");

        // Configure wgpu for optimal Windows performance
        let wgpu_setup = eframe::egui_wgpu::WgpuSetup::CreateNew(
            eframe::egui_wgpu::WgpuSetupCreateNew {
                instance_descriptor: eframe::wgpu::InstanceDescriptor {
                    // Prefer DX12 on Windows, Vulkan as fallback
                    backends: eframe::wgpu::Backends::DX12 | eframe::wgpu::Backends::VULKAN,
                    // Disable validation in release for performance
                    flags: eframe::wgpu::InstanceFlags::empty(),
                    ..Default::default()
                },
                power_preference: eframe::wgpu::PowerPreference::HighPerformance,
                device_descriptor: std::sync::Arc::new(|_adapter| {
                    eframe::wgpu::DeviceDescriptor {
                        label: Some("earshot"),
                        required_features: eframe::wgpu::Features::empty(),
                        required_limits: eframe::wgpu::Limits::downlevel_defaults(),
                        memory_hints: eframe::wgpu::MemoryHints::Performance,
                    }
                }),
                ..Default::default()
            }
        );

        eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default()
                .with_inner_size([800.0, 700.0])
                .with_min_inner_size([600.0, 500.0])
                .with_title("Earshot - Voice Assistant"),
            renderer: eframe::Renderer::Wgpu,
            wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
                wgpu_setup,
                on_surface_error: std::sync::Arc::new(|err| {
                    tracing::error!("wgpu surface error: {:?}", err);
                    eframe::egui_wgpu::SurfaceErrorAction::SkipFrame
                }),
                ..Default::default()
            },
            ..Default::default()
        }
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
