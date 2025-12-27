//! Main egui application with JARVIS-style visualization

use super::fft_data::FFTReceiver;
use super::state::{GuiCommand, GuiState, PipelineState};
use super::visualization::{JarvisRenderResources, JarvisVisualizer};
use crate::audio::fft::FFTSnapshot;
use eframe::egui;
use eframe::egui_wgpu;
use std::sync::Arc;

/// Main application state
pub struct EarshotApp {
    state: Arc<GuiState>,
    /// JARVIS visualizer state
    visualizer: Option<JarvisVisualizer>,
    /// FFT receiver for fallback (Glow) renderer
    fft_receiver: Option<FFTReceiver>,
    /// Last FFT snapshot for fallback rendering
    last_fft: FFTSnapshot,
    /// Whether we're using wgpu backend
    using_wgpu: bool,
    /// Start time for animation timing
    start_time: std::time::Instant,
}

impl EarshotApp {
    /// Create a new app with wgpu support (called from CreationContext)
    pub fn new(cc: &eframe::CreationContext<'_>, state: Arc<GuiState>) -> Self {
        // Check if we have wgpu
        let using_wgpu = cc.wgpu_render_state.is_some();

        // Take the FFT receiver from state (uses Mutex internally, so no Arc::get_mut needed)
        let fft_receiver = state.take_fft_receiver();

        // Initialize visualizer if we have FFT receiver
        // Handle both cases to avoid moving fft_receiver twice
        let (visualizer, fft_receiver) = if using_wgpu {
            (fft_receiver.map(JarvisVisualizer::new), None)
        } else {
            (None, fft_receiver)
        };

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
            fft_receiver,
            last_fft: FFTSnapshot::default(),
            using_wgpu,
            start_time: std::time::Instant::now(),
        }
    }

    /// Fallback constructor without wgpu (for testing)
    #[allow(dead_code)]
    pub fn new_simple(state: Arc<GuiState>) -> Self {
        Self {
            state,
            visualizer: None,
            fft_receiver: None,
            last_fft: FFTSnapshot::default(),
            using_wgpu: false,
            start_time: std::time::Instant::now(),
        }
    }

    /// Draw a simple fallback visualization using egui painter (when wgpu unavailable)
    fn draw_fallback_visualization(
        &self,
        ui: &mut egui::Ui,
        rect: egui::Rect,
        pipeline_state: PipelineState,
        fft_snapshot: &FFTSnapshot,
        input_level: f32,
    ) {
        let painter = ui.painter();
        let center = rect.center();
        let time = self.start_time.elapsed().as_secs_f32();

        // Design System Colors
        let teal = egui::Color32::from_rgb(0, 212, 170);       // #00D4AA
        let blue = egui::Color32::from_rgb(59, 130, 246);      // #3B82F6
        let indigo = egui::Color32::from_rgb(99, 102, 241);    // #6366F1
        let violet = egui::Color32::from_rgb(139, 92, 246);    // #8B5CF6
        let _fuchsia = egui::Color32::from_rgb(217, 70, 239);  // #D946EF

        // Direction based on who is speaking (user vs AI)
        let user_speaking = input_level > 0.05;
        let direction = if user_speaking { -1.0 } else { 1.0 };

        // Outer decorative ring (single) - violet
        painter.circle_stroke(
            center,
            180.0,
            egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(139, 92, 246, 60)),
        );

        // Rotating dashed ring (reduced dashes for performance) - indigo
        let ring_radius = 160.0;
        let num_dashes = 12;
        let rotation = time * 0.5 * direction;
        for i in 0..num_dashes {
            let angle = (i as f32 / num_dashes as f32) * std::f32::consts::TAU + rotation;
            let dash_len = 0.12 * std::f32::consts::TAU;
            let start_angle = angle;
            let end_angle = angle + dash_len;

            let start = center + egui::vec2(start_angle.cos(), start_angle.sin()) * ring_radius;
            let end = center + egui::vec2(end_angle.cos(), end_angle.sin()) * ring_radius;

            painter.line_segment([start, end], egui::Stroke::new(2.0, indigo));
        }

        // Central orb glow (reduced circles for performance) - blue glow
        let base_radius = 80.0;
        let pulse = (time * 2.0).sin() * 0.1 + 1.0;
        for i in (0..3).rev() {
            let r = base_radius * pulse + i as f32 * 20.0;
            let alpha = (200 - i as u8 * 60).max(20);
            painter.circle_filled(
                center,
                r,
                egui::Color32::from_rgba_unmultiplied(59, 130, 246, alpha / 3),
            );
        }

        // Central orb core - teal
        painter.circle_filled(center, base_radius * 0.6 * pulse, blue);
        painter.circle_filled(center, base_radius * 0.3 * pulse, teal);

        // Frequency bars (radial, from FFT snapshot) - gradient from blue to violet
        let num_bars = 32u32;
        let inner_radius = 100.0;
        let max_bar_height = 60.0;
        for i in 0..num_bars {
            // Sample every other magnitude for 32 bars from 64
            let mag_idx = (i as usize * 2).min(fft_snapshot.magnitudes.len() - 1);
            let mag = fft_snapshot.magnitudes[mag_idx];

            // Skip drawing if magnitude is effectively zero
            if mag < 0.01 {
                continue;
            }

            let angle = (i as f32 / num_bars as f32) * std::f32::consts::TAU
                - std::f32::consts::FRAC_PI_2;

            let shaped = mag.powf(1.6);
            let bar_height = max_bar_height * shaped;

            let start = center + egui::vec2(angle.cos(), angle.sin()) * inner_radius;
            let end = center + egui::vec2(angle.cos(), angle.sin()) * (inner_radius + bar_height);

            // Gradient color based on magnitude: blue -> indigo -> violet
            let t = shaped;
            let r = (59.0 + t * (139.0 - 59.0)) as u8;
            let g = (130.0 + t * (92.0 - 130.0)) as u8;
            let b = (246.0) as u8;
            let alpha = (shaped * 200.0) as u8 + 55;
            painter.line_segment(
                [start, end],
                egui::Stroke::new(5.0, egui::Color32::from_rgba_unmultiplied(r, g, b, alpha)),
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
            teal,
        );
    }
}

impl eframe::App for EarshotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let pipeline_state = self.state.state();

        // Request repaint at ~60Hz for smooth animation (GPU handles this well)
        ctx.request_repaint_after(std::time::Duration::from_millis(16));

        // Set dark background using design system bgBase #0A1628
        let bg_base = egui::Color32::from_rgb(10, 22, 40);
        let mut style = (*ctx.style()).clone();
        style.visuals.panel_fill = bg_base;
        style.visuals.window_fill = bg_base;
        style.visuals.extreme_bg_color = bg_base;
        ctx.set_style(style);

        // Central panel for JARVIS visualization
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(bg_base))
            .show(ctx, |ui| {
                // Get available rect for the visualizer
                let rect = ui.available_rect_before_wrap();
                let aspect_ratio = rect.width() / rect.height();

                // Render JARVIS visualizer if available
                if self.using_wgpu {
                    if let Some(visualizer) = &mut self.visualizer {
                        let input_level = self.state.input_level.load();
                        let callback = visualizer.update(aspect_ratio, pipeline_state as u32, input_level);

                        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                            rect,
                            callback,
                        ));
                    }
                } else {
                    if let Some(receiver) = self.fft_receiver.as_mut() {
                        if let Some(snapshot) = receiver.try_recv() {
                            self.last_fft = snapshot.clone();
                        }
                    }
                    // Fallback: Simple egui-based visualization when wgpu not available
                    let input_level = self.state.input_level.load();
                    self.draw_fallback_visualization(ui, rect, pipeline_state, &self.last_fft, input_level);
                }

                // Overlay UI elements on top of the visualization

                // Top-left: Branding - using teal (#00D4AA) and text secondary (#94A3B8)
                let branding_rect = egui::Rect::from_min_size(
                    rect.min + egui::vec2(20.0, 15.0),
                    egui::vec2(200.0, 40.0),
                );
                ui.put(branding_rect, |ui: &mut egui::Ui| {
                    ui.vertical(|ui| {
                        ui.label(
                            egui::RichText::new("E.A.R.S.H.O.T.")
                                .size(18.0)
                                .color(egui::Color32::from_rgb(0, 212, 170)) // Teal
                                .strong(),
                        );
                        ui.label(
                            egui::RichText::new("SYSTEM V1.0 - VOICE ASSISTANT")
                                .size(10.0)
                                .color(egui::Color32::from_rgb(148, 163, 184)), // textSecondary
                        );
                    });
                    ui.response()
                });

                // Top-right: Status indicators - using design system colors
                let status_rect = egui::Rect::from_min_size(
                    rect.min + egui::vec2(rect.width() - 280.0, 15.0),
                    egui::vec2(260.0, 20.0),
                );
                ui.put(status_rect, |ui: &mut egui::Ui| {
                    ui.horizontal(|ui| {
                        // Status based on pipeline state
                        let (status_text, status_color) = match pipeline_state {
                            PipelineState::Idle => ("IDLE", egui::Color32::from_rgb(0, 212, 170)), // Teal
                            PipelineState::Listening => {
                                ("LISTENING", egui::Color32::from_rgb(59, 130, 246)) // Blue
                            }
                            PipelineState::Processing => {
                                ("PROCESSING", egui::Color32::from_rgb(234, 88, 12)) // Orange accent
                            }
                            PipelineState::Speaking => {
                                ("SPEAKING", egui::Color32::from_rgb(139, 92, 246)) // Violet
                            }
                            PipelineState::Stopped => {
                                ("STOPPED", egui::Color32::from_rgb(100, 116, 139)) // textMuted
                            }
                        };

                        ui.label(
                            egui::RichText::new("SYS:")
                                .size(11.0)
                                .color(egui::Color32::from_rgb(148, 163, 184)), // textSecondary
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
                                .color(egui::Color32::from_rgb(148, 163, 184)), // textSecondary
                        );
                        ui.label(
                            egui::RichText::new("CONNECTED")
                                .size(11.0)
                                .color(egui::Color32::from_rgb(0, 212, 170)), // Teal
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
                // Using design system colors - teal for activate, orange for terminate
                let button_color = if is_stopped {
                    egui::Color32::from_rgb(0, 85, 68) // Dark teal
                } else {
                    egui::Color32::from_rgb(117, 44, 6) // Dark orange
                };
                let _button_hover_color = if is_stopped {
                    egui::Color32::from_rgb(0, 170, 136) // Lighter teal
                } else {
                    egui::Color32::from_rgb(234, 88, 12) // Orange
                };
                let text_color = if is_stopped {
                    egui::Color32::from_rgb(0, 212, 170) // Teal
                } else {
                    egui::Color32::from_rgb(234, 88, 12) // Orange
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
                            egui::Color32::from_rgb(0, 212, 170) // Teal
                        } else {
                            egui::Color32::from_rgb(234, 88, 12) // Orange
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

                // Bottom: Copyright/footer - using textMuted
                let footer_rect = egui::Rect::from_min_size(
                    egui::pos2(rect.center().x - 150.0, rect.max.y - 20.0),
                    egui::vec2(300.0, 15.0),
                );
                ui.put(footer_rect, |ui: &mut egui::Ui| {
                    ui.label(
                        egui::RichText::new("EARSHOT VOICE ASSISTANT - 2025")
                            .size(9.0)
                            .color(egui::Color32::from_rgb(100, 116, 139)), // textMuted
                    );
                    ui.response()
                });
            });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let _ = self.state.command_tx.try_send(GuiCommand::Shutdown);
    }
}
