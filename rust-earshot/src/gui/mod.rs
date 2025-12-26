//! GUI module for Earshot voice assistant

mod app;
mod state;
mod widgets;

pub use app::EarshotApp;
pub use state::{GuiCommand, GuiState, PipelineState};
