# Architecture Comparison: SAM3D Objects vs TRELLIS v1 vs TRELLIS.2

**Date**: 2026-02-24
**Updated**: 2026-02-24 (corrected with verified sources)

## Sources

- **TRELLIS v1** (Microsoft): [github.com/microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)
- **TRELLIS.2** (Microsoft): [github.com/microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2), [project page](https://microsoft.github.io/TRELLIS.2/)
- **SAM3D Objects** (Meta): `utils/third_party/sam3d/` in this repo (Meta's fork of TRELLIS v1)
- **Timing data**: Measured on GCP VM `genesisforge-gpu` (g2-standard-8, NVIDIA L4 24GB)

---

## Terminology

### SS (Sparse Structure)

The first stage of the pipeline. A **flow-matching generative model** that predicts
which voxels in a 64^3 grid are occupied by the object. Think of it as converting
a 2D image into a coarse 3D voxel skeleton.

- **"SS Flow"** = the original TRELLIS v1 architecture for this stage. Uses
  `ModulatedTransformerCrossBlock` (cross-attention between voxel latents and image
  features). Class: `SparseStructureFlowModel` in `trellis/models/sparse_structure_flow.py`
  ([source](https://github.com/microsoft/TRELLIS/blob/main/trellis/models/sparse_structure_flow.py)).
- **"SS MoT"** = Meta's upgraded version for SAM3D. Uses `MOTModulatedTransformerCrossBlock`
  (Mixture-of-Transformers), which adds **multiple latent heads** so the model can jointly
  predict voxel occupancy AND 6D object pose (rotation, translation, scale) in one pass.
  Same class name but in `sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py`.
- **"SS DiT"** = TRELLIS.2's version. A vanilla DiT (Diffusion Transformer) that predicts
  O-Voxel occupancy at 64^3 resolution. Config: `ss_flow_img_dit_1_3B_64_bf16.json` (~1.3B params).

In all three systems, SS is a **rectified flow model** — it learns a velocity field that
maps noise to structure.

### SLAT (Structured LATent)

The second stage in SAM3D and TRELLIS v1 only. A **sparse 3D U-Net flow model** that
takes the occupied voxel coordinates from SS and predicts an 8-dimensional latent feature
vector per voxel. These latent features encode the local geometry and appearance.

Architecture: `SLatFlowModel` in `structured_latent_flow.py`
([TRELLIS v1 source](https://github.com/microsoft/TRELLIS/blob/main/trellis/models/structured_latent_flow.py)):
- Input/output blocks: `SparseResBlock3d` with `sp.SparseConv3d` (sparse 3D convolutions)
- Middle blocks: `ModulatedSparseTransformerCrossBlock` (sparse attention + cross-attention to image)
- U-Net structure: downsample stages → transformer → upsample with skip connections
- Euler solver, configurable inference steps

**TRELLIS.2 does NOT use SLAT.** It replaces it with two separate DiT stages
(Geometry + Material).

### SS Decoder (D_S)

A small **VAE decoder** (`SparseStructureDecoder` in `sparse_structure_vae.py`) that
converts the SS generator's latent output into a binary occupancy grid. Standard 3D CNN
with `Conv3d` + `ResBlock3d` + `UpsampleBlock3d` layers. NOT a generative model —
just a deterministic decoder. ~10MB.

---

## Quick Summary Table

| | **SAM3D Objects** (Meta) | **TRELLIS v1** (Microsoft) | **TRELLIS.2** (Microsoft) |
|---|---|---|---|
| **Source** | Meta fork of TRELLIS v1 | [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) |
| **Total params** | ~1.8B | ~1.2B (large) | ~4.7B |
| **Stages** | 2 (SS + SLAT) | 2 (SS + SLAT) | 3 (SS + Geometry + Material) |
| **Pose prediction** | Yes (built-in via MoT) | No | No |
| **3D representation** | SLAT (64^3 voxels) | SLAT (64^3 voxels) | O-Voxel (octree, up to 1536^3) |
| **Mesh decoder** | FlexiCubes + GS | FlexiCubes + GS + RF | Direct from O-Voxels |
| **Texture** | UV-mapped 1024^2 | UV-mapped 1024^2 | PBR 2048^2 (albedo+normal+roughness) |
| **Speed (GCP L4)** | ~55s/obj batch, ~8 min/8 obj | Same as SAM3D (shared code) | ~80s/obj, ~12 min/10 obj |
| **Speed (H100)** | N/A | N/A | ~3s (512^3), ~17s (1024^3) |

---

## Detailed Pipeline: SAM3D Objects / TRELLIS v1

Both share the same core codebase. SAM3D **is** Meta's fork of TRELLIS v1 with three
additions: (1) MoT for pose prediction, (2) pointmap conditioning via MoGe,
(3) progressive training for scene reconstruction.

The TRELLIS v1 model definitions live at
[github.com/microsoft/TRELLIS/tree/main/trellis/models](https://github.com/microsoft/TRELLIS/tree/main/trellis/models).
Meta's modified copies are in `sam3d_objects/model/backbone/tdfy_dit/models/` in this repo.

### Input

```
RGBA image (mask baked into alpha channel)
  + MoGe ViT-L pointmap (3D coordinates per pixel, PyTorch3D camera space)
```

### Models (7 loaded, each a separate checkpoint)

```
Pipeline: InferencePipelinePointMap (inference_pipeline_pointmap.py)

MODEL 1: ss_condition_embedder          [~400MB]
├── DINOv2 ViT-B/14 backbone (facebookresearch/dinov2)
├── embed_dim: 768 → projected to cond_channels
├── SAM3D adds: PointPatchEmbed (pointmap.py) — converts MoGe pointmap to patch tokens
└── Output: (N_patches, cond_dim) conditioning tensor

MODEL 2: ss_generator                   [~500MB-1GB]
├── SAM3D (MoT variant):
│   └── SparseStructureFlowModel (mot_sparse_structure_flow.py)
│   └── MOTModulatedTransformerCrossBlock × N blocks
│   └── Multi-latent heads: "shape" + "quaternion" + "translation" + "scale"
├── TRELLIS v1 (standard):
│   └── SparseStructureFlowModel (sparse_structure_flow.py)
│   └── ModulatedTransformerCrossBlock × N blocks
│   └── Single latent head ("shape" only — no pose)
│   └── Parameters: resolution, model_channels, cond_channels, num_blocks, pe_mode (ape/rope)
└── Output: shape latent + pose latents (SAM3D only)

MODEL 3: ss_decoder                     [~10MB]
├── SparseStructureDecoder (sparse_structure_vae.py)
├── 3D CNN: Conv3d → ResBlock3d × N → UpsampleBlock3d
├── latent → occupancy (B, 1, 64, 64, 64)
└── Output: binary voxel grid, threshold > 0 → occupied coordinates

MODEL 4: slat_condition_embedder        [~400MB]
├── Same DINOv2 architecture as MODEL 1
└── Output: (N_patches, cond_dim) conditioning for SLAT stage

MODEL 5: slat_generator                 [~1GB]
├── SLatFlowModel (structured_latent_flow.py)
├── Sparse 3D U-Net:
│   ├── Input: sp.SparseLinear → hierarchical channels
│   ├── Down: SparseResBlock3d with SparseDownsample
│   ├── Middle: ModulatedSparseTransformerCrossBlock × N blocks
│   ├── Up: SparseResBlock3d with SparseUpsample + skip connections
│   └── Output: sp.SparseLinear → 8-channel SLAT features per voxel
├── Euler solver, configurable inference steps
└── Output: SparseTensor(coords, 8D features) for each occupied voxel

MODEL 6: slat_decoder_gs                [~200MB]
├── SLatGaussianDecoder (decoder_gs.py)
├── Sparse transformer (SparseTransformerBase)
├── Per-voxel output: num_gaussians × 14 channels
│   (3 xyz + 3 color + 3 scaling + 4 rotation + 1 opacity per gaussian)
├── Hammersley sequence perturbation for gaussian offset diversity
└── Output: 3D Gaussian Splat representation

MODEL 7: slat_decoder_mesh              [~200MB]
├── SLatMeshDecoder (decoder_mesh.py)
├── Sparse transformer + SparseSubdivideBlock3d (4× resolution upsample)
├── SparseFeatures2Mesh → FlexiCubes isosurface extraction
├── Texture baking: nvdiffrast → 1024×1024 UV atlas
└── Output: trimesh.Trimesh with UV-mapped texture
```

Note: TRELLIS v1 also supports a radiance field decoder (`slat_decoder_rf`) not used in SAM3D.

### Pose Prediction (SAM3D only)

```
pose_decoder() in inference_utils.py — pure function, no weights

Input:  ss_generator output dict containing:
        - "quaternion" or "6drotation_normalized" latent
        - "translation" latent
        - "scale" latent (log-space)
        - pointmap_scale, pointmap_shift from preprocessor

Steps:  1. 6D rotation → orthogonalize → 3×3 matrix → quaternion (wxyz)
        2. Scale: exp(log_scale)
        3. PoseTargetConverter: normalize by pointmap_scale/shift (ScaleShiftInvariant)

Output: rotation (1,4) quaternion, translation (1,3), scale (1,3)
```

### Output

| Format | Contents |
|---|---|
| GLB | Mesh + UV texture (FlexiCubes) |
| PLY | Gaussian splat point cloud |
| NPZ (checkpoint) | vertices, faces, vertex_colors, rotation, translation, scale |
| NPZ (mesh) | vertices, faces, vertex_colors (for downstream alignment) |

### Measured Timing (GCP L4 24GB)

| Metric | Value | Source |
|---|---|---|
| Model load (per-object) | ~26s | `output/sam3d_dining/summary.json` |
| Model load (batch, once) | ~27s | `output/sam3d_dining_batch/batch_summary.json` |
| Per-object (sequential) | ~120s avg | 8 obj in 1002s |
| Per-object (batch, cached) | ~55s avg | 8 obj in 442s inference |
| **Total 8 objects (batch)** | **491s (8.2 min)** | `sam3d_dining_batch` |
| Total 8 objects (sequential) | 1002s (16.7 min) | `sam3d_dining` |
| Flash-attn improvement | 2.4% (not significant) | `sam3d_dining_flash` |

---

## Detailed Pipeline: TRELLIS.2

### Source

[github.com/microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2),
model weights: [huggingface.co/microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)

### Input

```
RGBA image (mask baked into alpha channel)
No pointmap — TRELLIS.2 does not use MoGe
```

### Models

```
Pipeline: Trellis2ImageTo3DPipeline (from trellis2.pipelines)
Model:    microsoft/TRELLIS.2-4B (~4.7B parameters total)

Uses a Sparse 3D VAE with 16× spatial downsampling to encode up to 1536^3
assets into ~9.6K latent tokens.

Three internal flow-matching DiT stages:

STAGE 1: Sparse Structure DiT            [~1.3B params]
├── Config: ss_flow_img_dit_1_3B_64_bf16.json
├── Vanilla DiT (Diffusion Transformer)
├── Predicts O-Voxel occupancy (octree-based, NOT dense 64^3)
├── 12 inference steps
└── No pose prediction

STAGE 2: Geometry DiT (Shape SLat)       [~1.3B params]
├── Config: slat_flow_img2shape_dit_1_3B_512_bf16.json
├── Generates geometry attributes per occupied O-Voxel
├── Conditioned on sparse structure + image
├── 12 inference steps
└── Outputs shape latents

STAGE 3: Material DiT (Texture SLat)     [~1.3B params]
├── Config: slat_flow_imgshape2tex_dit_1_3B_512_bf16.json
├── Predicts PBR material properties:
│   ├── Albedo (base color)
│   ├── Normal map
│   └── Roughness + Metallic + Opacity
├── 12 inference steps
└── Conditioned on shape + image

Shape VAE:  shape_vae_next_dc_f16c32_fp16.json
Texture VAE: tex_vae_next_dc_f16c32_fp16.json

GLB Export: o_voxel.postprocess.to_glb()
├── Decimation: configurable (e.g. max_faces=10000)
├── Texture: 2048×2048 PBR, WebP compressed
├── Remesh available
└── Output: GLB binary
```

### O-Voxel vs SLAT

| | SLAT (SAM3D / TRELLIS v1) | O-Voxel (TRELLIS.2) |
|---|---|---|
| Structure | Dense 64^3 grid | Octree (hierarchical) |
| Max resolution | 64^3 = 262K voxels | Up to 1536^3 |
| Per-voxel data | 8D latent vector | Direct geometry + PBR material |
| Decoder required | Yes (GS + Mesh separately) | No (mesh extracted from O-Voxels) |
| Memory | O(N^3) = fixed | O(occupied) = adaptive |

### Output

| Format | Contents |
|---|---|
| GLB | Mesh + PBR textures (albedo + normal + roughness), WebP compressed |
| NPZ (mesh) | vertices (float32), faces (int32) — raw canonical frame |

### No Pose Prediction

TRELLIS.2 outputs meshes in a **canonical frame** (centered, Y-up in GLB convention).
For scene alignment, we tested **Module 4b** (SS Pose Estimation) which runs SAM3D's
SS model to predict rotation/translation/scale, but this did NOT improve alignment
(see Results section below).

### Measured Timing (GCP L4 24GB)

| Metric | Value | Source |
|---|---|---|
| Model load | 169s | `output/sam3d_dining_t2/trellis2_run.log` |
| Per-object inference | 27-118s (varies by complexity) | same log |
| Per-object GLB export | 15-65s (varies by vertex count) | same log |
| Per-object total | 43-141s | same log |
| **Total 10 objects** | **724s (12.1 min)** inference | `modular_dining_t2_sspose` |
| Total with model load | ~893s (14.9 min) | same |

Reported H100 timings from Microsoft:
- 512^3: ~3s, 1024^3: ~17s, 1536^3: ~60s

---

## Side-by-Side: Every Distinct Model

| # | Model Name | Architecture | Used In | Size | Purpose |
|---|---|---|---|---|---|
| 1 | **ss_condition_embedder** | DINOv2 ViT-B/14 + PointPatchEmbed | SAM3D, TRELLIS v1 | ~400MB | Image + pointmap → conditioning tokens |
| 2 | **ss_generator** (MoT) | MoT Flow Transformer | SAM3D only | ~500MB-1GB | Voxel occupancy + 6D pose |
| 2' | **ss_generator** (standard) | Flow Transformer | TRELLIS v1 only | ~500MB | Voxel occupancy (no pose) |
| 3 | **ss_decoder** | 3D CNN VAE decoder | SAM3D, TRELLIS v1 | ~10MB | Latent → 64^3 binary occupancy |
| 4 | **slat_condition_embedder** | DINOv2 ViT-B/14 | SAM3D, TRELLIS v1 | ~400MB | Image → conditioning for SLAT |
| 5 | **slat_generator** | Sparse 3D U-Net | SAM3D, TRELLIS v1 | ~1GB | 8D SLAT features per voxel |
| 6 | **slat_decoder_gs** | Sparse Transformer | SAM3D, TRELLIS v1 | ~200MB | SLAT → Gaussians |
| 7 | **slat_decoder_mesh** | Sparse Transformer + FlexiCubes | SAM3D, TRELLIS v1 | ~200MB | SLAT → mesh + UV texture |
| 8 | **MoGe ViT-L** (depth) | Monocular depth model | SAM3D, TRELLIS v1 | ~300MB | Image → 3D pointmap |
| 9 | **pose_decoder** | Pure function (no weights) | SAM3D only | 0 | SS latents → quaternion/T/S |
| 10 | **TRELLIS.2-4B** | 3-stage DiT (SS+Geo+Mat) + 2 VAEs | TRELLIS.2 only | ~4.7B | Image → O-Voxel → mesh + PBR |

**Total VRAM loaded:**
- SAM3D / TRELLIS v1: Models 1-8 = ~3-4GB (+ ~300MB MoGe)
- TRELLIS.2: Model 10 alone = ~9-10GB (fp16)
- Module 4b (SS pose for TRELLIS.2): Models 1-3 + 5-7 loaded (~3-4GB), only 1-3 used

---

## Architecture Diagrams

### SAM3D Objects (Meta) — with pose prediction

```
                    Image (RGBA)
                         │
                    ┌────┴────┐
                    ▼         ▼
            [MODEL 1]    [MODEL 8]
            DINOv2       MoGe ViT-L
            (ss_cond)    (depth)
                │             │
                │         pointmap (3,H,W)
                │             │
                ├─────────────┘
                ▼
           [MODEL 2]
           SS Generator (MoT)
                │
        ┌───────┼───────┐
        ▼       ▼       ▼
    shape    rotation  translation
    latent   latent    + scale latent
        │       │           │
        ▼       └─────┬─────┘
   [MODEL 3]    [pose_decoder]
   SS Decoder     (pure fn)
        │              │
   64^3 voxel     quaternion,
   occupancy      translation,
        │          scale
        ▼              │
   occupied            │  ← SS Pose: initial guess
   coordinates         │     for pose alignment
        │              │
        ▼              │
   [MODEL 4+5]         │
   DINOv2 + SLAT       │
   Generator            │
        │              │
   8D features         │
   per voxel           │
        │              │
   ┌────┴────┐         │
   ▼         ▼         │
[MODEL 6] [MODEL 7]   │
GS Decoder Mesh Dec.   │
   │       (FlexiCubes │
   │       + 1024^2    │
   │       texture)    │
   ▼         ▼         ▼
  PLY       GLB     Pose NPZ
```

### TRELLIS v1 (Microsoft) — no pose

Same architecture as SAM3D but:
- MODEL 2 uses standard `ModulatedTransformerCrossBlock` (no MoT, no pose heads)
- No pose_decoder, no MoGe pointmap conditioning
- DINOv2 used for conditioning (same as SAM3D)
- Output: canonical-frame mesh only (no scene placement)
- Also has radiance field decoder (not in SAM3D)
- Source: [github.com/microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)

### TRELLIS.2 (Microsoft)

```
             Image (RGBA)
                  │
                  ▼
          ┌──────────────┐
          │  TRELLIS.2   │
          │  (4.7B)      │
          │              │
          │ Stage 1: SS  │──→ O-Voxel occupancy (64^3)
          │   DiT ~1.3B  │
          │              │
          │ Stage 2:     │──→ Shape latents
          │   Geometry   │
          │   DiT ~1.3B  │
          │              │
          │ Stage 3:     │──→ PBR materials
          │   Material   │    (albedo, normal, roughness)
          │   DiT ~1.3B  │
          │              │
          │ Shape VAE    │──→ Decode shape latents → geometry
          │ Texture VAE  │──→ Decode material latents → PBR
          └──────┬───────┘
                 │
                 ▼
          o_voxel.postprocess
          .to_glb()
                 │
                 ▼
          GLB (2048^2 PBR)
          + NPZ (raw mesh)

          ⚠ No pose prediction
```

---

## Key Differences Explained

### 1. Why does SAM3D predict pose but TRELLIS v1/v2 don't?

SAM3D is designed for **scene reconstruction** — placing multiple objects in their
correct 3D positions from a single photo. Microsoft's TRELLIS models are designed
for **single-object generation** in a canonical frame (centered, normalized scale).

Meta added pose prediction by replacing the standard transformer blocks with MoT
(Mixture of Transformers) that have **multiple latent heads**: one for voxel shape
and others for rotation/translation/scale.

### 2. Speed comparison (GCP L4 24GB)

| System | 8-10 objects | Per-object avg | Model load |
|---|---|---|---|
| SAM3D/TRELLIS v1 (batch) | 491s (8.2 min) | ~55s | 27s (once) |
| SAM3D/TRELLIS v1 (sequential) | 1002s (16.7 min) | ~120s | ~26s (each) |
| TRELLIS.2 | 724s (12.1 min) | ~72s | 169s (once) |

TRELLIS.2 is slower per-object on L4 GPU despite the architectural improvements
because the 4.7B model is much larger. On H100, TRELLIS.2 is dramatically faster
(~3s per object at 512^3).

### 3. Texture quality

- **TRELLIS.2**: Dedicated ~1.3B Material DiT, 2048^2, full PBR (albedo+normal+roughness+metallic+opacity)
- **SAM3D/TRELLIS v1**: Texture baked from Gaussian splat colors via nvdiffrast, 1024^2, albedo only

### 4. Alignment quality (IoU on dining scene, GCP L4)

| System | Avg IoU | Best | Worst | Source |
|---|---|---|---|---|
| **TRELLIS v1 (SAM3D)** | **0.451** | 0.791 | 0.077 | `modular_dining_v4` |
| TRELLIS.2 + multi-start | 0.157 | 0.257 | 0.096 | `test_multistart_t2_v2` |
| TRELLIS.2 + SS Pose (Module 4b) | 0.157 | 0.233 | 0.053 | `modular_dining_t2_sspose` |

TRELLIS v1 alignment is **3x better** because:
1. SAM3D's MoT predicts pose trained end-to-end with the TRELLIS v1 decoder
2. TRELLIS v1's FlexiCubes mesh has clean topology suited for ICP
3. TRELLIS.2's O-Voxel mesh geometry is fundamentally different from what the SS model expects

---

## Module 4b: SS Pose for TRELLIS.2 — Negative Result

We tested using SAM3D's SS model as a standalone pose estimator for TRELLIS.2 objects.
**Result: No improvement.** The SS model's pose predictions are trained end-to-end with
TRELLIS v1's decoder output space and do not transfer to TRELLIS.2's O-Voxel mesh geometry.

| Object | T2+SS Pose | T2 Baseline | T1 Baseline |
|---|---|---|---|
| table | 0.053 | 0.187 | 0.411 |
| sofa | 0.186 | 0.191 | 0.269 |
| tablecloth | 0.116 | 0.121 | 0.492 |
| chair_cushion | 0.136 | 0.105 | 0.688 |
| armchair | 0.176 | 0.211 | 0.077 |
| neck_pillow | 0.233 | 0.257 | 0.791 |
| newspaper | 0.142 | 0.096 | 0.696 |
| plant_in_pot | 0.175 | 0.099 | 0.129 |
| placemat | 0.194 | 0.146 | 0.420 |
| **AVG** | **0.157** | **0.157** | **0.441** |

Files: `modules/ss_pose.py`, `tools/sam3d/ss_pose_worker.py`, `modules/run_all.py`
