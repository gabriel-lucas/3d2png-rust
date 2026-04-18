# AGENTS.md

## Quick start
```
cargo run                          # default: models/Avocado.glb -> output.png
cargo run -- -m <path> -o <path>   # custom model/output
cargo run -- -W 1024 -H 1024       # custom resolution
```

## Architecture
- Single binary: `src/main.rs` (entrypoint `main()`)
- Shader: `src/shader.wgsl` (embedded via `include_str!`)
- Flow: `initialize_wgpu` → `load_gltf` → `create_render_pipeline` → `render_frame` → `save_image`
- `render_frame` uses a **fixed 135° camera angle** (hardcoded)

## Gotchas
- **No tests, no CI** — verify by running `cargo run`
- Uses `edition = "2024"` (unstable; may require recent Rust toolchain)
- Texture `bytes_per_row` is aligned to 256 via `align_to()` — don't remove this
- `force_fallback_adapter: true` in adapter request — works without GPU
- Dead code: `load_embedded_texture`, `create_material_bind_group2`, `create_material_bind_group` (material-arg variant) — safe to delete
- Only first primitive per mesh is used; multi-primitive meshes emit a warning
- glTF extensions are not supported — `extensions_used()` check fails if any present
- Textures loaded relative to glTF file's parent directory
