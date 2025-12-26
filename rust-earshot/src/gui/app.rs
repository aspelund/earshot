//! Main egui application

use super::state::{GuiCommand, GuiState, PipelineState};
use super::widgets::{LevelMeter, StatusIndicator};
use eframe::egui;
use std::sync::Arc;

pub struct EarshotApp {
    state: Arc<GuiState>,
    /// Smoothed input level for display
    display_input_level: f32,
    /// Smoothed output level for display
    display_output_level: f32,
}

impl EarshotApp {
    pub fn new(state: Arc<GuiState>) -> Self {
        Self {
            state,
            display_input_level: 0.0,
            display_output_level: 0.0,
        }
    }

    /// Smooth level changes for visual appeal (fast attack, slow decay)
    fn smooth_level(current: f32, target: f32, decay: f32) -> f32 {
        if target > current {
            target // Fast attack
        } else {
            current * decay + target * (1.0 - decay) // Slow decay
        }
    }
}

impl eframe::App for EarshotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Read current levels from atomic state
        let input_level = self.state.input_level.load();
        let output_level = self.state.output_level.load();
        let vad_prob = self.state.vad_probability.load();
        let pipeline_state = self.state.state();

        // Smooth levels for display
        self.display_input_level =
            Self::smooth_level(self.display_input_level, input_level, 0.85);
        self.display_output_level =
            Self::smooth_level(self.display_output_level, output_level, 0.85);

        // Request repaint at ~30Hz for smooth meters
        ctx.request_repaint_after(std::time::Duration::from_millis(33));

        // Central panel
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(20.0);
                ui.label(egui::RichText::new("Earshot").size(32.0).strong());
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new("Voice Assistant")
                        .size(16.0)
                        .color(egui::Color32::from_gray(140)),
                );
            });

            ui.add_space(25.0);
            ui.separator();
            ui.add_space(20.0);

            // Status section
            ui.horizontal(|ui| {
                ui.add_space(10.0);
                let (status_text, status_color) = match pipeline_state {
                    PipelineState::Idle => ("Listening", egui::Color32::from_rgb(76, 175, 80)),
                    PipelineState::Listening => {
                        ("Speech Detected", egui::Color32::from_rgb(33, 150, 243))
                    }
                    PipelineState::Processing => {
                        ("Processing", egui::Color32::from_rgb(255, 152, 0))
                    }
                    PipelineState::Speaking => ("Speaking", egui::Color32::from_rgb(156, 39, 176)),
                    PipelineState::Stopped => ("Stopped", egui::Color32::from_gray(100)),
                };
                ui.add(StatusIndicator::new(status_text, status_color));
            });

            ui.add_space(25.0);

            // VAD probability bar
            ui.horizontal(|ui| {
                ui.add_space(20.0);
                ui.label(egui::RichText::new("VAD:").size(16.0));
                let bar_width = (ui.available_width() - 80.0).max(150.0);
                let (rect, _) =
                    ui.allocate_exact_size(egui::Vec2::new(bar_width, 16.0), egui::Sense::hover());
                if ui.is_rect_visible(rect) {
                    // Background
                    ui.painter()
                        .rect_filled(rect, 4.0, egui::Color32::from_gray(40));
                    // Filled portion
                    let filled = egui::Rect::from_min_size(
                        rect.min,
                        egui::Vec2::new(rect.width() * vad_prob, rect.height()),
                    );
                    let color = if vad_prob > 0.5 {
                        egui::Color32::from_rgb(76, 175, 80)
                    } else {
                        egui::Color32::from_rgb(80, 80, 80)
                    };
                    ui.painter().rect_filled(filled, 4.0, color);
                    // Border
                    ui.painter().rect_stroke(
                        rect,
                        4.0,
                        egui::Stroke::new(1.0, egui::Color32::from_gray(60)),
                        egui::StrokeKind::Outside,
                    );
                }
                ui.label(egui::RichText::new(format!("{:.0}%", vad_prob * 100.0)).size(16.0));
            });

            ui.add_space(40.0);

            // Audio level meters - centered with fixed spacing between them
            ui.horizontal(|ui| {
                let meter_width = 50.0;
                let spacing = 60.0;
                let total_width = meter_width * 2.0 + spacing;
                let left_margin = (ui.available_width() - total_width) / 2.0;
                ui.add_space(left_margin.max(0.0));
                ui.add(LevelMeter::new(self.display_input_level, "MIC"));
                ui.add_space(spacing);
                ui.add(LevelMeter::new(self.display_output_level, "SPK"));
            });

            ui.add_space(40.0);

            // Control buttons
            ui.vertical_centered(|ui| {
                let is_stopped = pipeline_state == PipelineState::Stopped;
                let button_text = if is_stopped {
                    "Start Listening"
                } else {
                    "Stop Listening"
                };

                let button = egui::Button::new(egui::RichText::new(button_text).size(18.0))
                    .min_size(egui::Vec2::new(200.0, 48.0));

                if ui.add(button).clicked() {
                    let cmd = if is_stopped {
                        GuiCommand::StartListening
                    } else {
                        GuiCommand::StopListening
                    };
                    let _ = self.state.command_tx.try_send(cmd);
                }
            });

            ui.add_space(20.0);
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let _ = self.state.command_tx.try_send(GuiCommand::Shutdown);
    }
}
