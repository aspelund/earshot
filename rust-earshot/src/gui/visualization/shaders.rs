//! WGSL shader sources for JARVIS visualization

/// Fullscreen quad vertex shader - renders a quad covering the entire screen
pub const FULLSCREEN_QUAD_VERTEX: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    // Generate fullscreen triangle positions
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0)
    );

    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 0.0)
    );

    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_idx], 0.0, 1.0);
    out.uv = uvs[vertex_idx];
    return out;
}
"#;

/// Uniforms for the JARVIS visualization
pub const JARVIS_UNIFORMS: &str = r#"
struct JarvisUniforms {
    time: f32,
    bass_energy: f32,
    mid_energy: f32,
    high_energy: f32,
    dominant_bin: u32,
    pipeline_state: u32,
    aspect_ratio: f32,
    _padding: f32,
}

struct BarData {
    magnitudes: array<f32, 64>,
}

@group(0) @binding(0) var<uniform> uniforms: JarvisUniforms;
@group(0) @binding(1) var<storage, read> bars: BarData;
"#;

/// Main JARVIS visualization fragment shader
pub const JARVIS_FRAGMENT: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct JarvisUniforms {
    time: f32,
    bass_energy: f32,
    mid_energy: f32,
    high_energy: f32,
    dominant_bin: u32,
    pipeline_state: u32,
    aspect_ratio: f32,
    _padding: f32,
}

struct BarData {
    magnitudes: array<f32, 64>,
}

@group(0) @binding(0) var<uniform> uniforms: JarvisUniforms;
@group(0) @binding(1) var<storage, read> bars: BarData;

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const NUM_BARS: u32 = 64u;

// Colors
const CYAN: vec3<f32> = vec3<f32>(0.0, 1.0, 1.0);
const TEAL: vec3<f32> = vec3<f32>(0.0, 0.5, 0.5);
const BLUE: vec3<f32> = vec3<f32>(0.0, 0.5, 1.0);
const PINK: vec3<f32> = vec3<f32>(1.0, 0.3, 0.5);

// Smooth step with configurable edge
fn smooth_edge(d: f32, edge: f32) -> f32 {
    return smoothstep(edge, edge - 0.002, d);
}

// Draw a ring segment
fn ring_segment(uv: vec2<f32>, inner: f32, outer: f32, start_angle: f32, end_angle: f32) -> f32 {
    let dist = length(uv);
    let angle = atan2(uv.y, uv.x);

    let radial = step(inner, dist) * step(dist, outer);

    // Normalize angles to [0, TAU]
    var a = angle;
    if a < 0.0 { a += TAU; }
    var s = start_angle;
    if s < 0.0 { s += TAU; }
    var e = end_angle;
    if e < 0.0 { e += TAU; }

    let angular = step(s, a) * step(a, e);
    return radial * angular;
}

// Draw a dashed ring
fn dashed_ring(uv: vec2<f32>, radius: f32, thickness: f32, num_dashes: u32, rotation: f32) -> f32 {
    let dist = length(uv);
    let angle = atan2(uv.y, uv.x) + rotation;

    // Radial mask
    let inner = radius - thickness * 0.5;
    let outer = radius + thickness * 0.5;
    let radial = smoothstep(inner - 0.003, inner, dist) * smoothstep(outer + 0.003, outer, dist);

    // Angular dashes
    var a = angle;
    if a < 0.0 { a += TAU; }
    let segment = TAU / f32(num_dashes);
    let dash_ratio = 0.5; // Dash takes 50% of segment
    let in_dash = step(fract(a / segment), dash_ratio);

    return radial * in_dash;
}

// Central glowing orb
fn orb(uv: vec2<f32>, pulse: f32) -> vec3<f32> {
    let dist = length(uv);
    let base_radius = 0.12;
    let radius = base_radius + pulse * 0.02;

    // Core glow (bright center)
    let core = smoothstep(radius, 0.0, dist);

    // Outer glow
    let glow_radius = radius + 0.08 + pulse * 0.04;
    let glow = smoothstep(glow_radius, radius, dist) * 0.6;

    // Color gradient
    let core_color = CYAN * core * 1.5;
    let glow_color = TEAL * glow;

    return core_color + glow_color;
}

// Radial frequency bars
fn radial_bars(uv: vec2<f32>) -> vec3<f32> {
    let dist = length(uv);
    let angle = atan2(uv.y, uv.x);

    // Normalize angle to [0, TAU]
    var a = angle;
    if a < 0.0 { a += TAU; }

    // Find which bar we're in
    let bar_angle = TAU / f32(NUM_BARS);
    let bar_idx = u32(a / bar_angle);
    let bar_center_angle = (f32(bar_idx) + 0.5) * bar_angle;

    // Get magnitude for this bar
    let magnitude = bars.magnitudes[bar_idx];

    // Bar geometry
    let inner_radius = 0.22;
    let max_bar_height = 0.25;
    let bar_height = inner_radius + magnitude * max_bar_height;
    let bar_width = bar_angle * 0.4; // 40% of segment width

    // Check if we're inside the bar
    let angle_diff = abs(a - bar_center_angle);
    let in_bar_angle = step(angle_diff, bar_width * 0.5);
    let in_bar_radial = step(inner_radius, dist) * step(dist, bar_height);
    let in_bar = in_bar_angle * in_bar_radial;

    // Color based on magnitude and position
    let t = (dist - inner_radius) / max_bar_height;
    var color = mix(BLUE, CYAN, t);
    color = mix(color, PINK, magnitude * t);

    // Highlight dominant frequency
    let is_dominant = f32(bar_idx == uniforms.dominant_bin);
    color += vec3<f32>(0.3) * is_dominant;

    // Glow effect (soft edge)
    let glow_strength = smoothstep(bar_height + 0.02, bar_height, dist) * in_bar_angle * magnitude;
    let glow = CYAN * glow_strength * 0.3;

    return color * in_bar + glow;
}

// Decorative rotating rings
fn decorative_rings(uv: vec2<f32>, time: f32, mid_energy: f32) -> vec3<f32> {
    var result = vec3<f32>(0.0);

    // Ring 1 - outer, slow rotation
    let ring1_speed = 0.1 + mid_energy * 0.1;
    let ring1 = dashed_ring(uv, 0.55, 0.008, 12u, time * ring1_speed);
    result += TEAL * ring1 * 0.5;

    // Ring 2 - middle, medium rotation (opposite direction)
    let ring2_speed = 0.15 + mid_energy * 0.15;
    let ring2 = dashed_ring(uv, 0.48, 0.006, 24u, -time * ring2_speed);
    result += CYAN * ring2 * 0.4;

    // Ring 3 - inner, fast rotation
    let ring3_speed = 0.2 + mid_energy * 0.2;
    let ring3 = dashed_ring(uv, 0.52, 0.004, 36u, time * ring3_speed);
    result += BLUE * ring3 * 0.3;

    // Solid thin ring
    let solid_ring = smoothstep(0.58, 0.578, length(uv)) * smoothstep(0.573, 0.575, length(uv));
    result += TEAL * solid_ring * 0.4;

    return result;
}

// Simple particle effect (pseudo-random dots drifting outward)
fn particles(uv: vec2<f32>, time: f32, high_energy: f32) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    let num_particles = 20u;

    for (var i = 0u; i < num_particles; i++) {
        // Pseudo-random position based on index
        let seed = f32(i) * 1.618033988749895; // Golden ratio
        let base_angle = fract(seed) * TAU;
        let phase = fract(seed * 2.718281828) * TAU;

        // Animate outward
        let lifetime = fract(time * 0.3 + phase);
        let radius = 0.15 + lifetime * 0.35;
        let angle = base_angle + time * 0.05;

        let particle_pos = vec2<f32>(cos(angle), sin(angle)) * radius;
        let dist_to_particle = length(uv - particle_pos);

        // Particle size and brightness
        let size = 0.008 * (1.0 - lifetime * 0.5);
        let brightness = (1.0 - lifetime) * (0.5 + high_energy * 0.5);

        let particle = smoothstep(size, size * 0.3, dist_to_particle) * brightness;
        result += CYAN * particle * 0.5;
    }

    return result;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Center UV and correct aspect ratio
    var uv = in.uv * 2.0 - 1.0;
    uv.x *= uniforms.aspect_ratio;

    // Background (pure black)
    var color = vec3<f32>(0.0);

    // Add layers (back to front)
    // 1. Decorative rings
    color += decorative_rings(uv, uniforms.time, uniforms.mid_energy);

    // 2. Radial frequency bars
    color += radial_bars(uv);

    // 3. Particles
    color += particles(uv, uniforms.time, uniforms.high_energy);

    // 4. Central orb (on top)
    color += orb(uv, uniforms.bass_energy);

    // Simple bloom approximation - brighten already bright areas
    let luminance = dot(color, vec3<f32>(0.299, 0.587, 0.114));
    color += color * smoothstep(0.5, 1.0, luminance) * 0.3;

    return vec4<f32>(color, 1.0);
}
"#;

/// Combined shader module with both vertex and fragment
pub fn get_combined_shader() -> String {
    format!("{}\n{}", FULLSCREEN_QUAD_VERTEX, JARVIS_FRAGMENT)
}
