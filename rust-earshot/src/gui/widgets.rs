//! Custom egui widgets for audio visualization

use egui::{Color32, Rect, Response, Sense, Ui, Vec2, Widget};

/// Vertical audio level meter widget
pub struct LevelMeter {
    level: f32,
    width: f32,
    height: f32,
    label: String,
}

impl LevelMeter {
    pub fn new(level: f32, label: impl Into<String>) -> Self {
        Self {
            level: level.clamp(0.0, 1.0),
            width: 50.0,
            height: 200.0,
            label: label.into(),
        }
    }

    #[allow(dead_code)]
    pub fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    #[allow(dead_code)]
    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }
}

impl Widget for LevelMeter {
    fn ui(self, ui: &mut Ui) -> Response {
        let desired_size = Vec2::new(self.width, self.height + 32.0);
        let (rect, response) = ui.allocate_exact_size(desired_size, Sense::hover());

        if ui.is_rect_visible(rect) {
            let painter = ui.painter();

            // Meter background
            let meter_rect = Rect::from_min_size(rect.min, Vec2::new(self.width, self.height));
            painter.rect_filled(meter_rect, 4.0, Color32::from_gray(30));

            // Level bar (from bottom up)
            let level_height = self.height * self.level;
            let level_rect = Rect::from_min_max(
                egui::pos2(meter_rect.min.x + 2.0, meter_rect.max.y - level_height - 2.0),
                egui::pos2(meter_rect.max.x - 2.0, meter_rect.max.y - 2.0),
            );

            let color = Color32::from_rgb(76, 175, 80); // Green

            painter.rect_filled(level_rect, 2.0, color);

            // Tick marks
            for i in 1..4 {
                let y = meter_rect.min.y + (self.height * i as f32 / 4.0);
                painter.line_segment(
                    [
                        egui::pos2(meter_rect.min.x, y),
                        egui::pos2(meter_rect.min.x + 4.0, y),
                    ],
                    egui::Stroke::new(1.0, Color32::from_gray(60)),
                );
            }

            // Border
            painter.rect_stroke(
                meter_rect,
                4.0,
                egui::Stroke::new(1.0, Color32::from_gray(60)),
                egui::StrokeKind::Outside,
            );

            // Label
            let text_pos = egui::pos2(rect.center().x, rect.max.y - 12.0);
            painter.text(
                text_pos,
                egui::Align2::CENTER_CENTER,
                &self.label,
                egui::FontId::proportional(16.0),
                Color32::from_gray(180),
            );
        }

        response
    }
}

/// Status indicator with colored dot
pub struct StatusIndicator {
    label: String,
    color: Color32,
}

impl StatusIndicator {
    pub fn new(label: impl Into<String>, color: Color32) -> Self {
        Self {
            label: label.into(),
            color,
        }
    }
}

impl Widget for StatusIndicator {
    fn ui(self, ui: &mut Ui) -> Response {
        ui.horizontal(|ui| {
            let (rect, response) = ui.allocate_exact_size(Vec2::splat(20.0), Sense::hover());
            if ui.is_rect_visible(rect) {
                ui.painter().circle_filled(rect.center(), 8.0, self.color);
                ui.painter().circle_stroke(
                    rect.center(),
                    8.0,
                    egui::Stroke::new(1.5, self.color.linear_multiply(0.5)),
                );
            }
            ui.label(egui::RichText::new(&self.label).size(18.0));
            response
        })
        .inner
    }
}
