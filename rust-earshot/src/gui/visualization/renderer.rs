//! WGPU renderer for JARVIS visualization

use super::shaders;
use crate::audio::fft::NUM_DISPLAY_BINS;
use crate::gui::fft_data::FFTReceiver;
use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu;
use eframe::wgpu;
use wgpu::util::DeviceExt;

/// Uniform data for the visualization shader
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct JarvisUniforms {
    pub time: f32,
    pub bass_energy: f32,
    pub mid_energy: f32,
    pub high_energy: f32,
    pub dominant_bin: u32,
    pub pipeline_state: u32,
    pub aspect_ratio: f32,
    pub _padding: f32,
}

impl Default for JarvisUniforms {
    fn default() -> Self {
        Self {
            time: 0.0,
            bass_energy: 0.0,
            mid_energy: 0.0,
            high_energy: 0.0,
            dominant_bin: 0,
            pipeline_state: 0,
            aspect_ratio: 1.0,
            _padding: 0.0,
        }
    }
}

/// Frequency bar data for the shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BarData {
    pub magnitudes: [f32; NUM_DISPLAY_BINS],
}

impl Default for BarData {
    fn default() -> Self {
        Self {
            magnitudes: [0.0; NUM_DISPLAY_BINS],
        }
    }
}

/// GPU resources for the JARVIS visualization
pub struct JarvisRenderResources {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bar_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl JarvisRenderResources {
    /// Create new render resources
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        // Create shader module
        let shader_source = shaders::get_combined_shader();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("JARVIS Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JARVIS Uniform Buffer"),
            contents: bytemuck::cast_slice(&[JarvisUniforms::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bar data buffer (storage buffer)
        let bar_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("JARVIS Bar Data Buffer"),
            contents: bytemuck::cast_slice(&[BarData::default()]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("JARVIS Bind Group Layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bar data
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("JARVIS Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bar_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("JARVIS Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("JARVIS Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            bar_buffer,
            bind_group,
        }
    }

    /// Update uniform buffer
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &JarvisUniforms) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[*uniforms]));
    }

    /// Update bar data buffer
    pub fn update_bars(&self, queue: &wgpu::Queue, bars: &BarData) {
        queue.write_buffer(&self.bar_buffer, 0, bytemuck::cast_slice(&[*bars]));
    }

    /// Render the visualization
    /// Note: Uses 'static because the resources are stored in CallbackResources
    /// which persists across frames
    pub fn render(&self, render_pass: &mut wgpu::RenderPass<'static>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..6, 0..1); // Fullscreen quad (2 triangles)
    }
}

/// Paint callback data for egui integration
pub struct JarvisVisualizerCallback {
    pub uniforms: JarvisUniforms,
    pub bars: BarData,
}

impl egui_wgpu::CallbackTrait for JarvisVisualizerCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Get or create resources
        let jarvis_resources: &JarvisRenderResources = resources.get().unwrap();

        // Update buffers
        jarvis_resources.update_uniforms(queue, &self.uniforms);
        jarvis_resources.update_bars(queue, &self.bars);

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let jarvis_resources: &JarvisRenderResources = resources.get().unwrap();
        // SAFETY: The resources stored in CallbackResources are created once and persist
        // for the entire lifetime of the renderer. The pipeline and bind_group references
        // we're passing to the render_pass are valid for the duration of the frame.
        // We use raw pointers to convince Rust these references are 'static.
        let pipeline: &'static wgpu::RenderPipeline = unsafe {
            std::mem::transmute(&jarvis_resources.pipeline)
        };
        let bind_group: &'static wgpu::BindGroup = unsafe {
            std::mem::transmute(&jarvis_resources.bind_group)
        };
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..6, 0..1); // Fullscreen quad (2 triangles)
    }
}

/// Visualizer state that manages FFT data and animation
pub struct JarvisVisualizer {
    start_time: std::time::Instant,
    fft_receiver: FFTReceiver,
    current_bars: BarData,
    current_uniforms: JarvisUniforms,
}

impl JarvisVisualizer {
    pub fn new(fft_receiver: FFTReceiver) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            fft_receiver,
            current_bars: BarData::default(),
            current_uniforms: JarvisUniforms::default(),
        }
    }

    /// Update from FFT data and return callback for rendering
    pub fn update(&mut self, aspect_ratio: f32, pipeline_state: u32) -> JarvisVisualizerCallback {
        // Update time
        self.current_uniforms.time = self.start_time.elapsed().as_secs_f32();
        self.current_uniforms.aspect_ratio = aspect_ratio;
        self.current_uniforms.pipeline_state = pipeline_state;

        // Try to get latest FFT data
        if let Some(snapshot) = self.fft_receiver.try_recv() {
            self.current_bars.magnitudes = snapshot.magnitudes;
            self.current_uniforms.bass_energy = snapshot.bass_energy;
            self.current_uniforms.mid_energy = snapshot.mid_energy;
            self.current_uniforms.high_energy = snapshot.high_energy;
            self.current_uniforms.dominant_bin = snapshot.dominant_bin as u32;
        }

        JarvisVisualizerCallback {
            uniforms: self.current_uniforms,
            bars: self.current_bars,
        }
    }
}
