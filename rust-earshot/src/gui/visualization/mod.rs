//! JARVIS-like audio visualization module
//!
//! This module provides a sci-fi inspired audio visualization using wgpu
//! for custom rendering with glow/bloom effects.

mod renderer;
mod shaders;

pub use renderer::{JarvisRenderResources, JarvisVisualizer, JarvisVisualizerCallback};
