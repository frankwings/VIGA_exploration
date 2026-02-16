# SAM3D / TRELLIS Deep Dive Study

**Date:** 2026-02-15
**Author:** kingy + Claude (Opus 4.6)

## 1. Pipeline Verification

**Confirmed:** SAM3D is a two-stage pipeline — SAM segmentation followed by TRELLIS 3D reconstruction.

### Stage 1: SAM Segmentation (`tools/sam3d/sam_worker.py`)

- Uses **SAM ViT-H** (`sam_vit_h_4b8939.pth`) for automatic mask generation
- `SamAutomaticMaskGenerator` produces raw masks for all objects
- **Panic filtering** (`panic_filtering_process()`): removes masks < 1% of image area, rejects masks > 50% coverage (background), requires 70% unique area contribution, keeps max 10 masks
- Each mask saved as transparent PNG + `.npy` binary
- **VLM naming** (`get_object_name_from_vlm()`): GPT-4o identifies each masked object and assigns a descriptive name (e.g., `green_tea_bottle`, `wooden_table`)

### Stage 2: TRELLIS 3D Reconstruction (`tools/sam3d/sam3d_worker.py`)

- Runs per-object in a subprocess (conda env: `sam3d_py311`)
- Uses **MoGe** (depth/pointmap estimation) internally to compute camera intrinsics
- TRELLIS inference: Sparse Structure → SLAT → Dual Decode (Gaussian + Mesh)
- Outputs per object:
  - `.glb` file (3D mesh with textures)
  - `translation` (3D vector)
  - `rotation` (quaternion)
  - `scale` (scalar)

### Orchestrator (`tools/sam3d/init.py`)

- Calls SAM worker → iterates objects → calls SAM3D worker per object (sequential)
- Aggregates all object transforms into `object_transforms.json`
- Optionally imports all GLBs into a Blender scene via `glb_import.py`

### Data Flow

```
Input Image
    │
    ▼
SAM ViT-H (sam_worker.py)
    │  → raw masks → panic filter → VLM naming
    │  → {object_name}.png + {object_name}.npy per object
    ▼
TRELLIS (sam3d_worker.py) × N objects [sequential]
    │  → MoGe pointmap → camera intrinsics (computed but NOT exported)
    │  → Sparse Structure (2 steps) → SLAT (12 steps) → Dual Decode
    │  → {object_name}.glb + {translation, rotation, scale}
    ▼
Blender Import (glb_import.py)
    │  → Camera hardcoded at origin (0,0,0), facing -Y, 32mm lens
    │  → GLBs imported but NOT positioned using transforms
    ▼
state.blend + Camera.png
```

## 2. Speed Analysis

### Current Timing (RTX 5080 16GB, TRELLIS 1)

| Stage | Time per Object | Notes |
|---|---|---|
| Model loading | ~26 sec | DINOv2 + TRELLIS checkpoints |
| Sparse Structure | 8-14 sec | ShortCut solver, 2 steps |
| SLAT | 20-74 sec | Euler solver, 12 steps |
| Dual Decode | ~7 min | 32 Gaussians/voxel, 64³ resolution |
| **Total per object** | **9-33 min** | Depends on complexity |

5 objects sequential = **1.5-2 hours total**

### Bottleneck Ranking

1. **Decoder (50%)** — 32 Gaussians/voxel at 64³ resolution = ~300K vertices. Hardcoded in checkpoint weights, cannot reduce without retraining.
2. **SLAT sampling (30%)** — 12 Euler steps. Tuneable via `slat_generator.yaml` → `inference_steps` (try 6-8 with quality tradeoff).
3. **Sequential processing (15%)** — For-loop in `init.py`. Could parallelize if VRAM allows.
4. **Model loading (5%)** — DINOv2 reloaded per object. Could cache across objects.

### TRELLIS 2 (Released Dec 2025)

| Feature | TRELLIS 1 | TRELLIS 2 |
|---|---|---|
| Architecture | Sparse Structure + SLAT | O-Voxel (unified) |
| Parameters | ~1B | **4B** |
| Speed (H100) | ~10 min/object | **~3 sec/object** |
| Speed (consumer GPU) | 9-33 min | Unknown (not benchmarked) |
| Decoder | Dual (Gaussian + Mesh) | SC-VAE (Sparse Convolutional VAE) |
| Output formats | GLB, Gaussian, Radiance Field | GLB, Gaussian, Radiance Field |
| Repo | `microsoft/TRELLIS` | `microsoft/TRELLIS` (v2 branch) |

**Key concern:** 4B params may not fit on 16GB VRAM. Needs benchmarking.

### Fast-SAM3D (Training-Free Speedup)

- **2.67x speedup** on existing TRELLIS 1 without retraining
- Reduces SLAT inference steps via distilled solver scheduling
- Drop-in replacement for the TRELLIS inference call
- Repo: `https://github.com/wlfeng0509/Fast-SAM3D`

## 3. Output Format & Camera Position

### Per-Object Output

Each reconstructed object produces:
```json
{
    "glb_path": "path/to/object.glb",
    "translation": [x, y, z],
    "rotation": [w, x, y, z],
    "scale": [sx, sy, sz]
}
```

### Coordinate Transforms Applied (`transform_mesh_vertices()`)

The mesh vertices go through 6 transforms:
1. Z-flip (`R_flip_z`)
2. Y-up → Z-up conversion (`R_yup_to_zup`)
3. Scale from model output
4. Rotation from model output (quaternion → matrix)
5. Translation from model output
6. PyTorch3D → camera conversion (`R_pytorch3d_to_cam`)
7. Y-flip + X-flip corrections

### Camera Position Gap

**Critical finding:** Camera intrinsics ARE computed inside the TRELLIS pipeline but are NOT exported.

- In `inference_pipeline_pointmap.py`, MoGe calls `infer_intrinsics_from_pointmap()` which produces focal length and principal point
- These values are used internally for the pointmap-to-3D conversion
- But the output dict only contains `{glb, translation, rotation, scale}` — no camera data
- `glb_import.py` uses a **hardcoded camera** at `(0, 0, 0)` facing `-Y` with 32mm lens

**To get aligned renders matching the input image, we need to:**
1. Export camera intrinsics from the TRELLIS pipeline
2. Use those intrinsics to set the Blender camera (focal length, position)
3. Apply per-object transforms when importing GLBs (currently ignored)

## 4. Aligned Render Feasibility

### What exists today
- Per-object GLBs with world-space transforms (translation, rotation, scale)
- MoGe-computed camera intrinsics (focal length, principal point) — computed but discarded

### What needs to change

| Change | File | Effort |
|---|---|---|
| Export camera intrinsics from MoGe | `sam3d_worker.py` + `inference.py` | Small — add fields to output JSON |
| Pass camera data through orchestrator | `init.py` | Small — forward the JSON fields |
| Apply transforms when importing GLBs | `glb_import.py` | Medium — apply translation/rotation/scale per object |
| Set Blender camera from intrinsics | `glb_import.py` | Medium — compute FOV from focal length, set camera location |
| Render aligned view | Existing render pipeline | None — already works once camera is correct |

### Expected Result
With these changes, running the SAM3D pipeline on a target image would produce a Blender scene where:
- Each object is positioned at its estimated world location
- The camera is positioned and configured to match the original viewpoint
- Rendering from this camera produces an image closely matching the input

## 5. Optimization Roadmap

### Quick Wins (no retraining)
1. **Fast-SAM3D** — 2.67x speedup, drop-in replacement
2. **Reduce SLAT steps** — `slat_generator.yaml` → `inference_steps: 8` (from 12)
3. **Cache model loading** — Load DINOv2 + TRELLIS once, reuse across objects
4. **Parallel objects** — Process 2 objects simultaneously if VRAM allows

### Medium Term
5. **TRELLIS 2** — ~3 sec/object but needs VRAM benchmarking on consumer GPU
6. **Export camera intrinsics** — Enable aligned rendering

### Long Term
7. **Distilled decoder** — Would need retraining or waiting for official release
8. **Server-grade GPU** — A100/H100 for production workloads
