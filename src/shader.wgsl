struct Camera {
    view_proj: mat4x4<f32>,
};

struct Object {
    model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(2) @binding(0) var<uniform> object: Object;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_pos = object.model * vec4<f32>(input.position, 1.0);
    output.position = camera.view_proj * world_pos;
    output.uv = input.uv;
    return output;
}

@group(1) @binding(0) var base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var base_color_sampler: sampler;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(base_color_texture, base_color_sampler, input.uv);
}