//! Main egui application with JARVIS-style visualization

use super::state::{GuiCommand, GuiState, PipelineState};
use super::visualization::{JarvisRenderResources, JarvisVisualizer};
use eframe::egui;
use eframe::egui_wgpu;
use std::sync::Arc;

/// Main application state
pub struct EarshotApp {
    state: Arc<GuiState>,
    /// JARVIS visualizer state
    visualizer: Option<JarvisVisualizer>,
    /// Whether we're using wgpu backend
    using_wgpu: bool,
}

impl EarshotApp {
    /// Create a new app with wgpu support (called from CreationContext)
    pub fn new(cc: &eframe::CreationContext<'_>, mut state: Arc<GuiState>) -> Self {
        // Check if we have wgpu
        let using_wgpu = cc.wgpu_render_state.is_some();

        // Take the FFT receiver from state
        let fft_receiver = if using_wgpu {
            // We need mutable access to take the receiver
            // This is a bit awkward with Arc, so we'll make this work
            Arc::get_mut(&mut state).and_then(|s| s.take_fft_receiver())
        } else {
            None
        };

        // Initialize visualizer if we have FFT receiver
        let visualizer = fft_receiver.map(JarvisVisualizer::new);

        // Initialize wgpu resources if available
        if let Some(wgpu_state) = &cc.wgpu_render_state {
            // Get the target format from the renderer
            let target_format = wgpu_state.target_format;

            // Create JARVIS render resources
            let resources =
                JarvisRenderResources::new(&wgpu_state.device, target_format);

            // Insert into callback resources
            wgpu_state
                .renderer
                .write()
                .callback_resources
                .insert(resources);
        }

        Self {
            state,
            visualizer,
            using_wgpu,
        }
    }

    /// Fallback constructor without wgpu (for testing)
    #[allow(dead_code)]
    pub fn new_simple(state: Arc<GuiState>) -> Self {
        Self {
            state,
            visualizer: None,
            using_wgpu: false,
        }
    }

    /// Draw a simple fallback visualization using egui painter (when wgpu unavailable)
    fn draw_fallback_visualization(
        &self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        pipeline_state: PipelineState,
    ) {
        let painter = ui.painter();
        let center = rect.center();
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f32();

        // Colors
        let cyan = egui::Color32::from_rgb(0, 255, 255);
        let teal = egui::Color32::from_rgb(0, 128, 128);
        let dark_cyan = egui::Color32::from_rgb(0, 80, 80);

        // Outer decorative rings
        for i in 0..3 {
            let radius = 180.0 + i as f32 * 25.0;
            let alpha = 60 - i as u8 * 15;
            painter.circle_stroke(
                center,
                radius,
                egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(0, 200, 200, alpha)),
            );
        }

        // Rotating dashed ring
        let ring_radius = 160.0;
        let num_dashes = 24;
        let rotation = time * 0.5;
        for i in 0..num_dashes {
            let angle = (i as f32 / num_dashes as f32) * std::f32::consts::TAU + rotation;
            let dash_len = 0.08 * std::f32::consts::TAU;
            let start_angle = angle;
            let end_angle = angle + dash_len;

            let start = center + egui::vec2(start_angle.cos(), start_angle.sin()) * ring_radius;
            let end = center + egui::vec2(end_angle.cos(), end_angle.sin()) * ring_radius;

            painter.line_segment([start, end], egui::Stroke::new(2.0, dark_cyan));
        }

        // Central orb glow (multiple circles for gradient effect)
        let base_radius = 80.0;
        let pulse = (time * 2.0).sin() * 0.1 + 1.0;
        for i in (0..5).rev() {
            let r = base_radius * pulse + i as f32 * 15.0;
            let alpha = (255 - i as u8 * 50).max(20);
            painter.circle_filled(
                center,
                r,
                egui::Color32::from_rgba_unmultiplied(0, 128, 128, alpha / 3),
            );
        }

        // Central orb core
        painter.circle_filled(center, base_radius * 0.6 * pulse, teal);
        painter.circle_filled(center, base_radius * 0.3 * pulse, cyan);

        // Simulated frequency bars (radial)
        let num_bars = 32;
        let inner_radius = 100.0;
        let max_bar_height = 50.0;
        for i in 0..num_bars {
            let angle = (i as f32 / num_bars as f32) * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;

            // Fake frequency response based on time
            let freq = (time * 3.0 + i as f32 * 0.3).sin().abs();
            let bar_height = freq * max_bar_height * 0.5 + max_bar_height * 0.1;

            let start = center + egui::vec2(angle.cos(), angle.sin()) * inner_radius;
            let end = center + egui::vec2(angle.cos(), angle.sin()) * (inner_radius + bar_height);

            let alpha = (freq * 200.0) as u8 + 55;
            painter.line_segment(
                [start, end],
                egui::Stroke::new(4.0, egui::Color32::from_rgba_unmultiplied(0, 255, 255, alpha)),
            );
        }

        // Status indicator in center
        let status_text = match pipeline_state {
            PipelineState::Idle => "IDLE",
            PipelineState::Listening => "LISTENING",
            PipelineState::Processing => "PROCESSING",
            PipelineState::Speaking => "SPEAKING",
            PipelineState::Stopped => "STOPPED",
        };
        painter.text(
            center + egui::vec2(0.0, base_radius + 40.0),
            egui::Align2::CENTER_CENTER,
            status_text,
            egui::FontId::proportional(12.0),
            cyan,
        );
    }
}

impl eframe::App for EarshotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let pipeline_state = self.state.state();

        // Request repaint at ~60Hz for smooth animation
        ctx.request_repaint_after(std::time::Duration::from_millis(16));

        // Set dark background
        let mut style = (*ctx.style()).clone();
        style.visuals.panel_fill = egui::Color32::BLACK;
        style.visuals.window_fill = egui::Color32::BLACK;
        style.visuals.extreme_bg_color = egui::Color32::BLACK;
        ctx.set_style(style);

        // Central panel for JARVIS visualization
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::BLACK))
            .show(ctx, |ui| {
                // Get available rect for the visualizer
                let rect = ui.available_rect_before_wrap();
                let aspect_ratio = rect.width() / rect.height();

                // Render JARVIS visualizer if available
                if self.using_wgpu {
                    if let Some(visualizer) = &mut self.visualizer {
                        let callback = visualizer.update(aspect_ratio, pipeline_state as u32);

                        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                            rect,
                            callback,
                        ));
                    }
                } else {
                    // Fallback: Simple egui-based visualization when wgpu not available
                    self.draw_fallback_visualization(ui, rect, pipeline_state);
                }

                // Overlay UI elements on top of the visualization

                // Top-left: Branding
                let branding_rect = egui::Rect::from_min_size(
                    rect.min + egui::vec2(20.0, 15.0),
                    egui::vec2(200.0, 40.0),
                );
                ui.put(branding_rect, |ui: &mut egui::Ui| {
                    ui.vertical(|ui| {
                        ui.label(
                            egui::RichText::new("E.A.R.S.H.O.T.")
                                .size(18.0)
                                .color(egui::Color32::from_rgb(0, 255, 255))
                                .strong(),
                        );
                        ui.label(
                            egui::RichText::new("SYSTEM V1.0 - VOICE ASSISTANT")
                                .size(10.0)
                                .color(egui::Color32::from_rgb(0, 180, 180)),
                        );
                    });
                    ui.response()
                });

                // Top-right: Status indicators
                let status_rect = egui::Rect::from_min_size(
                    rect.min + egui::vec2(rect.width() - 280.0, 15.0),
                    egui::vec2(260.0, 20.0),
                );
                ui.put(status_rect, |ui: &mut egui::Ui| {
                    ui.horizontal(|ui| {
                        // Status based on pipeline state
                        let (status_text, status_color) = match pipeline_state {
                            PipelineState::Idle => ("IDLE", egui::Color32::from_rgb(0, 255, 0)),
                            PipelineState::Listening => {
                                ("LISTENING", egui::Color32::from_rgb(0, 200, 255))
                            }
                            PipelineState::Processing => {
                                ("PROCESSING", egui::Color32::from_rgb(255, 180, 0))
                            }
                            PipelineState::Speaking => {
                                ("SPEAKING", egui::Color32::from_rgb(200, 100, 255))
                            }
                            PipelineState::Stopped => {
                                ("STOPPED", egui::Color32::from_gray(100))
                            }
                        };

                        ui.label(
                            egui::RichText::new("SYS:")
                                .size(11.0)
                                .color(egui::Color32::from_rgb(0, 200, 200)),
                        );
                        ui.label(
                            egui::RichText::new(status_text)
                                .size(11.0)
                                .color(status_color),
                        );
                        ui.add_space(15.0);
                        ui.label(
                            egui::RichText::new("NET:")
                                .size(11.0)
                                .color(egui::Color32::from_rgb(0, 200, 200)),
                        );
                        ui.label(
                            egui::RichText::new("CONNECTED")
                                .size(11.0)
                                .color(egui::Color32::from_rgb(0, 255, 0)),
                        );
                    });
                    ui.response()
                });

                // Bottom center: TERMINATE button
                let button_width = 160.0;
                let button_height = 40.0;
                let button_rect = egui::Rect::from_min_size(
                    egui::pos2(
                        rect.center().x - button_width / 2.0,
                        rect.max.y - button_height - 30.0,
                    ),
                    egui::vec2(button_width, button_height),
                );

                let is_stopped = pipeline_state == PipelineState::Stopped;
                let button_text = if is_stopped { "ACTIVATE" } else { "TERMINATE" };
                let button_color = if is_stopped {
                    egui::Color32::from_rgb(0, 100, 0) // Dark green
                } else {
                    egui::Color32::from_rgb(139, 0, 0) // Dark red
                };
                let button_hover_color = if is_stopped {
                    egui::Color32::from_rgb(0, 150, 0)
                } else {
                    egui::Color32::from_rgb(200, 0, 0)
                };
                let text_color = if is_stopped {
                    egui::Color32::from_rgb(0, 255, 0)
                } else {
                    egui::Color32::from_rgb(255, 100, 100)
                };

                ui.put(button_rect, |ui: &mut egui::Ui| {
                    let button = egui::Button::new(
                        egui::RichText::new(button_text)
                            .size(16.0)
                            .color(text_color)
                            .strong(),
                    )
                    .fill(button_color)
                    .stroke(egui::Stroke::new(
                        2.0,
                        if is_stopped {
                            egui::Color32::from_rgb(0, 200, 0)
                        } else {
                            egui::Color32::from_rgb(200, 50, 50)
                        },
                    ))
                    .min_size(egui::vec2(button_width, button_height));

                    let response = ui.add(button);

                    if response.clicked() {
                        let cmd = if is_stopped {
                            GuiCommand::StartListening
                        } else {
                            GuiCommand::StopListening
                        };
                        let _ = self.state.command_tx.try_send(cmd);
                    }

                    response
                });

                // Bottom: Copyright/footer
                let footer_rect = egui::Rect::from_min_size(
                    egui::pos2(rect.center().x - 150.0, rect.max.y - 20.0),
                    egui::vec2(300.0, 15.0),
                );
                ui.put(footer_rect, |ui: &mut egui::Ui| {
                    ui.label(
                        egui::RichText::new("EARSHOT VOICE ASSISTANT - 2025")
                            .size(9.0)
                            .color(egui::Color32::from_gray(80)),
                    );
                    ui.response()
                });
            });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let _ = self.state.command_tx.try_send(GuiCommand::Shutdown);
    }
}
