// main.rs - Complete working version with transforms
use anyhow::{Context, Result};
use clap::Parser;
use gltf::image::Source;
use image::{GenericImageView, ImageBuffer, Rgba};
use std::path::{Path, PathBuf};
use wgpu::*;
use wgpu::util::DeviceExt;

use cgmath::{Matrix4, SquareMatrix, Vector3};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = "models/Avocado.glb")]
    model: PathBuf,
    #[arg(short, long, default_value = "output.png")]
    output: PathBuf,
    #[arg(short = 'W', long, default_value_t = 800)]
    width: u32,
    #[arg(short = 'H', long, default_value_t = 600)]
    height: u32,
}

struct RenderContext {
    device: Device,
    queue: Queue,
    width: u32,
    height: u32,
    camera_buffer: Buffer,
    camera_bind_group: BindGroup,
    camera_layout: BindGroupLayout,
    object_layout: BindGroupLayout,
    depth_view: TextureView,
    material_layout: BindGroupLayout,
    sampler: Sampler,
}

struct Model {
    primitives: Vec<Primitive>,
    materials: Vec<Material>,
    render_items: Vec<RenderItem>,
    bounds_min: Vector3<f32>,
    bounds_max: Vector3<f32>,
}

struct Primitive {
    vertex: Buffer,
    index: Option<Buffer>,
    index_count: u32,
    material: usize,
}

struct RenderItem {
    primitive: usize,
    transform: Matrix4<f32>,
}

struct Material {
    bind_group: BindGroup,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectUniform {
    model: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    uv: [f32; 2],
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let ctx = init(args.width, args.height).await?;
    let model = load_gltf(&ctx, &args.model).await?;
    let pipeline = create_pipeline(&ctx);

    let pixels = render(&ctx, &model, &pipeline).await?;
    save(&args.output, pixels, args.width, args.height)?;

    println!("Saved image to {}", args.output.display());
    Ok(())
}

async fn init(width: u32, height: u32) -> Result<RenderContext> {
    let instance = Instance::default();

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| anyhow::anyhow!("Failed to find a suitable GPU adapter"))?;

    let (device, queue) = adapter
        .request_device(&DeviceDescriptor::default(), None)
        .await?;

    let camera_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("camera"),
        size: std::mem::size_of::<CameraUniform>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let camera_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("camera_layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
        layout: &camera_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
        label: Some("camera_bind_group"),
    });

    let depth_texture = device.create_texture(&TextureDescriptor {
        label: Some("depth_texture"),
        size: Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth24Plus,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let object_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("object_layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let material_layout = create_material_layout(&device);
    let sampler = device.create_sampler(&SamplerDescriptor::default());

    Ok(RenderContext {
        device,
        queue,
        width,
        height,
        camera_buffer,
        camera_bind_group,
        camera_layout,
        object_layout,
        depth_view: depth_texture.create_view(&Default::default()),
        material_layout,
        sampler,
    })
}

fn node_transform(node: &gltf::Node) -> cgmath::Matrix4<f32> {
    use cgmath::*;

    match node.transform() {
        gltf::scene::Transform::Matrix { matrix } => Matrix4::from(matrix),
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => {
            let t = Matrix4::from_translation(Vector3::from(translation));
            let r = Matrix4::from(Quaternion::from(rotation));
            let s = Matrix4::from_nonuniform_scale(scale[0], scale[1], scale[2]);
            t * r * s
        }
    }
}

fn traverse_node(
    node: gltf::Node,
    parent_transform: Matrix4<f32>,
    mesh_primitive_map: &[Vec<usize>],
    render_items: &mut Vec<RenderItem>,
) {
    let local_transform = node_transform(&node);
    let world_transform = parent_transform * local_transform;

    if let Some(mesh) = node.mesh() {
        if let Some(primitive_indices) = mesh_primitive_map.get(mesh.index()) {
            for &prim_idx in primitive_indices {
                render_items.push(RenderItem {
                    primitive: prim_idx,
                    transform: world_transform,
                });
            }
        }
    }

    for child in node.children() {
        traverse_node(child, world_transform, mesh_primitive_map, render_items);
    }
}

async fn load_gltf(ctx: &RenderContext, path: &Path) -> Result<Model> {
    //use cgmath::{Matrix4, Vector3, Transform, Point3};
    use cgmath::{Matrix4, Vector3, Transform, Point3, EuclideanSpace};
    //use cgmath::{Matrix4, Vector3, Point3, EuclideanSpace, SquareMatrix};
    
    let (doc, buffers, _) = gltf::import(path)?;
    let base = path.parent().context("Model has no parent directory")?;

    // ---- Images -> GPU textures ----
    let mut images = Vec::new();
    for img in doc.images() {
        let data = match img.source() {
            Source::Uri { uri, .. } => std::fs::read(base.join(uri))?,
            Source::View { view, .. } => {
                let buf = &buffers[view.buffer().index()].0;
                buf[view.offset()..view.offset() + view.length()].to_vec()
            }
        };
        let image = image::load_from_memory(&data).context("Failed to decode image")?;
        images.push(upload_texture(ctx, &image));
    }

    // ---- Default texture ----
    let default_tex = {
        let data = [255u8, 255, 255, 255];
        let tex = ctx.device.create_texture(&TextureDescriptor {
            size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            label: Some("default_white_texture"),
            view_formats: &[],
        });
        ctx.queue.write_texture(
            tex.as_image_copy(),
            &data,
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        tex
    };

    // ---- glTF textures -> GPU textures ----
    let mut textures = Vec::new();
    for tex in doc.textures() {
        textures.push(&images[tex.source().index()]);
    }

    // ---- Materials ----
    let mut materials = Vec::new();
    for mat in doc.materials() {
        let tex = mat.pbr_metallic_roughness()
            .base_color_texture()
            .map(|t| t.texture().index())
            .and_then(|i| textures.get(i).copied())
            .unwrap_or(&default_tex);
        let view = tex.create_view(&Default::default());
        let bind = ctx.device.create_bind_group(&BindGroupDescriptor {
            layout: &ctx.material_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&view) },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&ctx.sampler) },
            ],
            label: None,
        });
        materials.push(Material { bind_group: bind });
    }

    // ---- Read all primitive data (CPU side) ----
    struct CpuPrimitive {
        vertices: Vec<Vertex>,
        indices: Option<Vec<u32>>,
        material: usize,
    }
    let mut cpu_primitives = Vec::new();

    for mesh in doc.meshes() {
        for prim in mesh.primitives() {
            let reader = prim.reader(|b| Some(&buffers[b.index()].0));
            let positions: Vec<_> = reader.read_positions().context("No positions")?.collect();
            let uvs: Vec<_> = reader
                .read_tex_coords(0)
                .map(|c| c.into_f32().collect())
                .unwrap_or(vec![[0.0, 0.0]; positions.len()]);
            let vertices: Vec<Vertex> = positions
                .into_iter()
                .zip(uvs)
                .map(|(p, uv)| Vertex { pos: p, uv })
                .collect();
            let indices = reader.read_indices().map(|indices| indices.into_u32().collect());
            cpu_primitives.push(CpuPrimitive {
                vertices,
                indices,
                material: prim.material().index().unwrap_or(0),
            });
        }
    }

    // ---- Build mapping from (mesh, primitive) to cpu_primitives index ----
    let mut primitive_index = 0;
    let mut mesh_primitive_map = Vec::new();
    for mesh in doc.meshes() {
        let mut prim_indices = Vec::new();
        for _ in mesh.primitives() {
            prim_indices.push(primitive_index);
            primitive_index += 1;
        }
        mesh_primitive_map.push(prim_indices);
    }

    // ---- Traverse scene to build render_items and collect world positions for bounds ----
    let mut render_items = Vec::new();
    let mut world_positions = Vec::new();

    fn traverse_and_collect(
        node: gltf::Node,
        parent_transform: Matrix4<f32>,
        mesh_primitive_map: &[Vec<usize>],
        cpu_primitives: &[CpuPrimitive],
        render_items: &mut Vec<RenderItem>,
        world_positions: &mut Vec<Vector3<f32>>,
    ) {
        let local_transform = node_transform(&node);
        let world_transform = parent_transform * local_transform;

        if let Some(mesh) = node.mesh() {
            if let Some(prim_indices) = mesh_primitive_map.get(mesh.index()) {
                for &prim_idx in prim_indices {
                    render_items.push(RenderItem {
                        primitive: prim_idx,
                        transform: world_transform,
                    });
                    // Add transformed vertices for bounds
                    let prim = &cpu_primitives[prim_idx];
                    for vertex in &prim.vertices {
                        let pos = Vector3::new(vertex.pos[0], vertex.pos[1], vertex.pos[2]);
                        let world_pos = world_transform.transform_point(Point3::new(pos.x, pos.y, pos.z));
                        world_positions.push(world_pos.to_vec());
                    }
                }
            }
        }

        for child in node.children() {
            traverse_and_collect(
                child,
                world_transform,
                mesh_primitive_map,
                cpu_primitives,
                render_items,
                world_positions,
            );
        }
    }

    let scene = doc.default_scene().unwrap_or_else(|| doc.scenes().next().unwrap());
    for node in scene.nodes() {
        traverse_and_collect(
            node,
            Matrix4::identity(),
            &mesh_primitive_map,
            &cpu_primitives,
            &mut render_items,
            &mut world_positions,
        );
    }

    // ---- Compute bounding box ----
    let mut bounds_min = Vector3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut bounds_max = Vector3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for pos in &world_positions {
        bounds_min.x = bounds_min.x.min(pos.x);
        bounds_min.y = bounds_min.y.min(pos.y);
        bounds_min.z = bounds_min.z.min(pos.z);
        bounds_max.x = bounds_max.x.max(pos.x);
        bounds_max.y = bounds_max.y.max(pos.y);
        bounds_max.z = bounds_max.z.max(pos.z);
    }
    if world_positions.is_empty() {
        bounds_min = Vector3::new(-1.0, -1.0, -1.0);
        bounds_max = Vector3::new(1.0, 1.0, 1.0);
    }

    println!("Bounds min: {:?}, max: {:?}", bounds_min, bounds_max);
    println!("Center: {:?}", (bounds_min + bounds_max) / 2.0);
    println!("Size: {:?}", bounds_max - bounds_min);

    // ---- Upload primitives to GPU ----
    let mut primitives = Vec::new();
    for cpu_prim in cpu_primitives {
        let vertex = ctx.device.create_buffer_init(&util::BufferInitDescriptor {
            contents: bytemuck::cast_slice(&cpu_prim.vertices),
            usage: BufferUsages::VERTEX,
            label: None,
        });
        let (index, count) = if let Some(indices) = cpu_prim.indices {
            let buf = ctx.device.create_buffer_init(&util::BufferInitDescriptor {
                contents: bytemuck::cast_slice(&indices),
                usage: BufferUsages::INDEX,
                label: None,
            });
            (Some(buf), indices.len() as u32)
        } else {
            (None, cpu_prim.vertices.len() as u32)
        };
        primitives.push(Primitive {
            vertex,
            index,
            index_count: count,
            material: cpu_prim.material,
        });
    }

    Ok(Model {
        primitives,
        materials,
        render_items,
        bounds_min,
        bounds_max,
    })
}

fn upload_texture(ctx: &RenderContext, img: &image::DynamicImage) -> Texture {
    let rgba = img.to_rgba8();
    let (w, h) = img.dimensions();

    let tex = ctx.device.create_texture(&TextureDescriptor {
        size: Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        label: None,
        view_formats: &[],
    });

    ctx.queue.write_texture(
        tex.as_image_copy(),
        &rgba,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: Some(h),
        },
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );

    tex
}

fn create_pipeline(ctx: &RenderContext) -> RenderPipeline {
    let shader = ctx.device.create_shader_module(include_wgsl!("shader.wgsl"));

    let layout = ctx.device.create_pipeline_layout(&PipelineLayoutDescriptor {
        bind_group_layouts: &[&ctx.camera_layout, &ctx.material_layout, &ctx.object_layout],
        push_constant_ranges: &[],
        label: Some("pipeline_layout"),
    });

    ctx.device.create_render_pipeline(&RenderPipelineDescriptor {
        layout: Some(&layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as _,
                step_mode: VertexStepMode::Vertex,
                attributes: &[
                    VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: VertexFormat::Float32x3,
                    },
                    VertexAttribute {
                        offset: 12,
                        shader_location: 1,
                        format: VertexFormat::Float32x2,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(ColorTargetState {
                format: TextureFormat::Rgba8UnormSrgb,
                blend: Some(BlendState::REPLACE),
                write_mask: ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth24Plus,
            depth_write_enabled: true,
            depth_compare: CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        primitive: Default::default(),
        multisample: Default::default(),
        multiview: None,
        cache: None,
        label: Some("render_pipeline"),
    })
}

async fn render(ctx: &RenderContext, model: &Model, pipeline: &RenderPipeline) -> Result<Vec<u8>> {
    use cgmath::*;
    
        // Auto-frame camera based on model bounds
    let center = (model.bounds_min + model.bounds_max) / 2.0;
    let size = model.bounds_max - model.bounds_min;
    let radius = size.magnitude() / 2.0; // half diagonal

    // Compute distance needed for vertical FOV
    let fov_rad = 45.0_f32.to_radians();
    let distance_vertical = radius / (fov_rad / 2.0).tan();

    // Also consider horizontal FOV based on aspect ratio
    let aspect = ctx.width as f32 / ctx.height as f32;
    let horizontal_fov = 2.0 * (aspect * (fov_rad / 2.0).tan()).atan();
    let distance_horizontal = radius / (horizontal_fov / 2.0).tan();

    // Take the larger distance to ensure both axes fit
    let distance = distance_vertical.max(distance_horizontal) * 1.2; // 20% margin

    // Place camera at a direction that gives a good view (from +X, +Y, +Z)
    let direction = Vector3::new(1.0, 1.0, 1.0).normalize();
    let eye = center + direction * distance;
    let target = center;

    let cam = Matrix4::look_at_rh(Point3::from_vec(eye), Point3::from_vec(target), Vector3::unit_y());
    let proj = perspective(Deg(45.0), aspect, 0.01, distance * 3.0);
    let vp = proj * cam;

    ctx.queue.write_buffer(
        &ctx.camera_buffer,
        0,
        bytemuck::cast_slice(&[CameraUniform {
            view_proj: vp.into(),
        }]),
    );

    // Prepare transforms for all render items
    let mut transforms = Vec::new();
    for item in &model.render_items {
        transforms.push(ObjectUniform {
            model: item.transform.into(),
        });
    }

    // Create a single buffer large enough for all transforms
    let transform_size = std::mem::size_of::<ObjectUniform>() as u64;
    let total_size = transform_size * transforms.len() as u64;
    let transform_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("transform_buffer"),
        size: total_size,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Write all transforms
    if !transforms.is_empty() {
        ctx.queue.write_buffer(&transform_buffer, 0, bytemuck::cast_slice(&transforms));
    }

    // Create one bind group that references the whole buffer
    let object_bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        layout: &ctx.object_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &transform_buffer,
                offset: 0,
                size: None,
            }),
        }],
        label: Some("object_bind_group"),
    });

    // Create output texture
    let tex = ctx.device.create_texture(&TextureDescriptor {
        size: Extent3d {
            width: ctx.width,
            height: ctx.height,
            depth_or_array_layers: 1,
        },
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        label: Some("output_texture"),
        view_formats: &[],
    });

    let view = tex.create_view(&Default::default());

    let mut enc = ctx.device.create_command_encoder(&Default::default());

    {
        let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &ctx.depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            label: Some("render_pass"),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &ctx.camera_bind_group, &[]);

        // Draw each primitive with its transform
        for (i, item) in model.render_items.iter().enumerate() {
            let prim = &model.primitives[item.primitive];
            let dynamic_offset = (i * transform_size as usize) as u32;

            pass.set_bind_group(2, &object_bind_group, &[dynamic_offset]);
            pass.set_bind_group(1, &model.materials[prim.material].bind_group, &[]);
            pass.set_vertex_buffer(0, prim.vertex.slice(..));

            if let Some(idx) = &prim.index {
                pass.set_index_buffer(idx.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..prim.index_count, 0, 0..1);
            } else {
                pass.draw(0..prim.index_count, 0..1);
            }
        }
    }

    // Read back pixels
    let padded_bytes_per_row = align_to(4 * ctx.width, 256);
    let buffer_size = (padded_bytes_per_row * ctx.height) as u64;

    let buffer = ctx.device.create_buffer(&BufferDescriptor {
        size: buffer_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
        label: Some("readback_buffer"),
    });

    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(ctx.height),
            },
        },
        Extent3d {
            width: ctx.width,
            height: ctx.height,
            depth_or_array_layers: 1,
        },
    );

    ctx.queue.submit(Some(enc.finish()));

    let slice = buffer.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    ctx.device.poll(Maintain::Wait);

    let data = slice.get_mapped_range();

    let mut pixels = Vec::with_capacity((ctx.width * ctx.height * 4) as usize);

    let padded = align_to(4 * ctx.width, 256) as usize;
    let row_size = (4 * ctx.width) as usize;

    for y in 0..ctx.height as usize {
        let start = y * padded;
        let end = start + row_size;
        pixels.extend_from_slice(&data[start..end]);
    }

    drop(data);
    buffer.unmap();

    Ok(pixels)
}

fn save(path: &Path, pixels: Vec<u8>, w: u32, h: u32) -> Result<()> {
    ImageBuffer::<Rgba<u8>, _>::from_raw(w, h, pixels)
        .context("Invalid image buffer")?
        .save(path)?;
    Ok(())
}

fn create_material_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("material_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    view_dimension: TextureViewDimension::D2,
                    sample_type: TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn align_to(value: u32, alignment: u32) -> u32 {
    ((value + alignment - 1) / alignment) * alignment
}