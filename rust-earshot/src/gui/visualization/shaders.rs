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
    input_level: f32,
}

struct BarData {
    magnitudes: array<f32, 64>,
}

@group(0) @binding(0) var<uniform> uniforms: JarvisUniforms;
@group(0) @binding(1) var<storage, read> bars: BarData;
"#;

/// Main JARVIS visualization fragment shader
/// Note: VertexOutput, JarvisUniforms, BarData, and bindings are defined in the
/// vertex shader portion and shared when shaders are combined
pub const JARVIS_FRAGMENT: &str = r#"
const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;
const NUM_BARS: u32 = 64u;

// Design System Colors
// Primary palette from brand guidelines
const TEAL: vec3<f32> = vec3<f32>(0.0, 0.831, 0.667);      // #00D4AA - main brand color
const BLUE: vec3<f32> = vec3<f32>(0.231, 0.51, 0.965);     // #3B82F6
const INDIGO: vec3<f32> = vec3<f32>(0.388, 0.4, 0.945);    // #6366F1
const VIOLET: vec3<f32> = vec3<f32>(0.545, 0.361, 0.965);  // #8B5CF6
const FUCHSIA: vec3<f32> = vec3<f32>(0.851, 0.275, 0.937); // #D946EF
const ORANGE: vec3<f32> = vec3<f32>(0.918, 0.345, 0.047);  // #EA580C - accent

// Background colors
const BG_BASE: vec3<f32> = vec3<f32>(0.039, 0.086, 0.157); // #0A1628
const BG_SUBTLE: vec3<f32> = vec3<f32>(0.075, 0.137, 0.216); // #132337

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

    // Color gradient - teal core with blue glow
    let core_color = TEAL * core * 1.5;
    let glow_color = mix(TEAL, BLUE, 0.5) * glow;

    return core_color + glow_color;
}

fn bin_distance(a: u32, b: u32) -> f32 {
    let diff = abs(i32(a) - i32(b));
    let wrap = i32(NUM_BARS) - diff;
    return f32(min(diff, wrap));
}

// Radial frequency bars
fn radial_bars(uv: vec2<f32>) -> vec3<f32> {
    let dist = length(uv);
    let angle = atan2(uv.y, uv.x);

    // Normalize angle to [0, TAU] and rotate so low bins sit near the bottom.
    var a = angle + PI * 0.5;
    if a < 0.0 { a += TAU; }

    // Find which bar we're in
    let bar_angle = TAU / f32(NUM_BARS);
    let bar_idx = u32(a / bar_angle);
    let bar_center_angle = (f32(bar_idx) + 0.5) * bar_angle;

    // Get magnitude for this bar - return black if effectively zero
    let magnitude = bars.magnitudes[bar_idx];
    if magnitude < 0.01 {
        return vec3<f32>(0.0);
    }
    let shaped = pow(magnitude, 1.6);

    // Directional focus around dominant bin for a "burst" arc
    let dist_bins = bin_distance(bar_idx, uniforms.dominant_bin);
    let focus = exp(-0.5 * (dist_bins * dist_bins) / (6.0 * 6.0));
    let beam = pow(focus, 5.0) * (0.12 + uniforms.high_energy * 0.3);

    // Bar geometry - no base height so zero magnitude = no bar
    let inner_radius = 0.22;
    let max_bar_height = 0.25 + beam;
    let bar_height = inner_radius + shaped * max_bar_height * (0.4 + 0.9 * focus);
    let bar_width = bar_angle * 0.4; // 40% of segment width

    // Check if we're inside the bar
    let angle_diff = abs(a - bar_center_angle);
    let in_bar_angle = step(angle_diff, bar_width * 0.5);
    let in_bar_radial = step(inner_radius, dist) * step(dist, bar_height);
    let in_bar = in_bar_angle * in_bar_radial;

    // Spray accent for the dominant cluster
    let spray_width = bar_width * 0.35;
    let spray_height = bar_height + beam * 0.9;
    let in_spray_angle = step(angle_diff, spray_width * 0.5);
    let in_spray_radial = step(inner_radius, dist) * step(dist, spray_height);
    let spray = in_spray_angle * in_spray_radial * pow(focus, 3.0);

    // Color based on magnitude and position - gradient through palette
    let t = (dist - inner_radius) / max_bar_height;
    // Base gradient: Blue -> Indigo -> Violet based on height
    var color = mix(BLUE, INDIGO, t);
    // Add Fuchsia for high energy bars
    color = mix(color, FUCHSIA, shaped * t * 0.7);
    // Teal tint for the spray/burst effect
    color = mix(color, TEAL, spray * 0.6);

    // Highlight dominant frequency with orange accent
    let is_dominant = f32(bar_idx == uniforms.dominant_bin);
    color = mix(color, ORANGE, is_dominant * 0.4);

    // Glow effect (soft edge) - teal glow
    let glow_strength = smoothstep(bar_height + 0.02, bar_height, dist) * in_bar_angle * shaped;
    let glow = TEAL * glow_strength * (0.25 + 0.25 * focus);

    return color * in_bar + glow + color * spray * 0.6;
}

// Decorative rotating rings
// Direction reverses based on who is speaking (user vs AI)
fn decorative_rings(uv: vec2<f32>, time: f32, mid_energy: f32, input_level: f32) -> vec3<f32> {
    var result = vec3<f32>(0.0);

    // Direction multiplier: positive when AI speaks, negative when user speaks
    // Use threshold to detect when user is speaking
    let user_speaking = step(0.05, input_level);
    let direction = mix(1.0, -1.0, user_speaking);

    // Speed boost when someone is actively speaking
    let activity = max(mid_energy, input_level);

    // Ring 1 - outer, slow rotation (indigo)
    let ring1_speed = 0.1 + activity * 0.15;
    let ring1 = dashed_ring(uv, 0.55, 0.008, 12u, time * ring1_speed * direction);
    result += INDIGO * ring1 * 0.5;

    // Ring 2 - middle, medium rotation (opposite direction, blue)
    let ring2_speed = 0.15 + activity * 0.2;
    let ring2 = dashed_ring(uv, 0.48, 0.006, 24u, -time * ring2_speed * direction);
    result += BLUE * ring2 * 0.5;

    // Ring 3 - inner, fast rotation (teal)
    let ring3_speed = 0.2 + activity * 0.25;
    let ring3 = dashed_ring(uv, 0.52, 0.004, 36u, time * ring3_speed * direction);
    result += TEAL * ring3 * 0.4;

    // Solid thin ring (violet)
    let solid_ring = smoothstep(0.58, 0.578, length(uv)) * smoothstep(0.573, 0.575, length(uv));
    result += VIOLET * solid_ring * 0.4;

    return result;
}

// Particle effect - direction and count based on who is speaking
// User speaking: particles move inward (fewer particles)
// AI speaking: particles move outward (more particles)
fn particles(uv: vec2<f32>, time: f32, high_energy: f32, input_level: f32) -> vec3<f32> {
    var result = vec3<f32>(0.0);

    // Determine who is speaking
    let user_speaking = step(0.05, input_level);
    let ai_speaking = step(0.05, high_energy);

    // More particles when AI speaks (40), fewer when user speaks (15)
    let max_particles = 40u;

    for (var i = 0u; i < max_particles; i++) {
        // Pseudo-random position based on index
        let seed = f32(i) * 1.618033988749895; // Golden ratio
        let base_angle = fract(seed) * TAU;
        let phase = fract(seed * 2.718281828) * TAU;

        // Only render subset of particles when user is speaking
        // When AI speaking: show all 40
        // When user speaking: show first 15
        // When idle: show first 20
        let particle_limit = select(select(20u, 40u, ai_speaking > 0.5), 15u, user_speaking > 0.5);
        if i >= particle_limit {
            continue;
        }

        // Animate based on who is speaking
        let lifetime = fract(time * 0.3 + phase);

        // User speaking: particles move inward (from outer to inner)
        // AI speaking: particles move outward (from inner to outer)
        let inner_radius = 0.15;
        let outer_radius = 0.50;
        var radius: f32;
        if user_speaking > 0.5 {
            // Inward: start at outer, move to inner
            radius = outer_radius - lifetime * 0.35;
        } else {
            // Outward: start at inner, move to outer
            radius = inner_radius + lifetime * 0.35;
        }

        let angle = base_angle + time * 0.05;

        let particle_pos = vec2<f32>(cos(angle), sin(angle)) * radius;
        let dist_to_particle = length(uv - particle_pos);

        // Particle size and brightness
        let activity = max(high_energy, input_level);
        let size = 0.008 * (1.0 - lifetime * 0.5);
        let brightness = (1.0 - lifetime) * (0.5 + activity * 0.5);

        let particle = smoothstep(size, size * 0.3, dist_to_particle) * brightness;
        result += TEAL * particle * 0.6;
    }

    return result;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Center UV and correct aspect ratio
    var uv = in.uv * 2.0 - 1.0;
    uv.x *= uniforms.aspect_ratio;

    // Background - dark navy from design system
    var color = BG_BASE;

    // Add layers (back to front)
    // 1. Decorative rings
    color += decorative_rings(uv, uniforms.time, uniforms.mid_energy, uniforms.input_level);

    // 2. Radial frequency bars
    color += radial_bars(uv);

    // 3. Particles
    color += particles(uv, uniforms.time, uniforms.high_energy, uniforms.input_level);

    // 4. Central orb (on top)
    color += orb(uv, uniforms.bass_energy);

    // Simple bloom approximation - brighten already bright areas
    let luminance = dot(color, vec3<f32>(0.299, 0.587, 0.114));
    color += color * smoothstep(0.5, 1.0, luminance) * 0.3;

    return vec4<f32>(color, 1.0);
}
"#;

/// Combined shader module with vertex, uniforms, and fragment
pub fn get_combined_shader() -> String {
    format!("{}\n{}\n{}", FULLSCREEN_QUAD_VERTEX, JARVIS_UNIFORMS, JARVIS_FRAGMENT)
}
