# Architecture Comparison: SAM3D Objects vs TRELLIS v1 vs TRELLIS.2

**Date**: 2026-02-24

## Terminology

### SS (Sparse Structure)

The first stage of the pipeline. A **flow-matching generative model** that predicts
which voxels in a 64^3 grid are occupied by the object. Think of it as converting
a 2D image into a coarse 3D voxel skeleton.

- **"SS Flow"** = the original TRELLIS v1 architecture for this stage. It uses a
  standard `ModulatedTransformerCrossBlock` (cross-attention between voxel latents
  and image features). Class: `SparseStructureFlowModel` in `sparse_structure_flow.py`.
- **"SS MoT"** = Meta's upgraded version for SAM3D. Uses `MOTModulatedTransformerCrossBlock`
  (Mixture-of-Transformers), which adds **multiple latent heads** so the model can jointly
  predict voxel occupancy AND 6D object pose (rotation, translation, scale) in one pass.
  Same class name `SparseStructureFlowModel` but in `mot_sparse_structure_flow.py`.
- **"SS DiT"** = TRELLIS.2's version. A vanilla DiT (Diffusion Transformer) that predicts
  O-Voxel occupancy. Different architecture entirely.

In all three systems, SS is a **rectified flow model** — it learns a velocity field that
maps noise to structure in a small number of steps (typically 2 via ShortCut solver).

### SLAT (Structured LATent)

The second stage. A **sparse 3D U-Net flow model** that takes the occupied voxel coordinates
from SS and predicts an 8-dimensional latent feature vector per voxel. These latent
features encode the local geometry and appearance at each voxel position.

Architecture: `SLatFlowModel` in `structured_latent_flow.py`:
- Input/output blocks: `SparseResBlock3d` with `sp.SparseConv3d` (sparse 3D convolutions)
- Middle blocks: `ModulatedSparseTransformerCrossBlock` (sparse attention + cross-attention to image)
- U-Net structure: downsample (8 -> 16 -> 32 resolution) -> transformer -> upsample
- Skip connections between down/up stages

SLAT is used in **SAM3D** and **TRELLIS v1** only. TRELLIS.2 replaces it with a
geometry DiT + material DiT.

### SS Decoder (D_S)

A small **VAE decoder** (`SparseStructureDecoder`) that converts the SS generator's
8-channel latent output into a binary occupancy grid. It is a standard 3D CNN with
`Conv3d` + `ResBlock3d` + `UpsampleBlock3d` layers. This is NOT a generative model
— just a deterministic decoder.

---

## Quick Summary Table

| | **SAM3D Objects** (Meta) | **TRELLIS v1** (Microsoft) | **TRELLIS.2** (Microsoft) |
|---|---|---|---|
| **Relationship** | Fork of TRELLIS v1 + Meta additions | Original | New architecture |
| **Total params** | ~1.8B | ~1-2B | ~4.7B |
| **Stages** | 2 (SS + SLAT) | 2 (SS + SLAT) | 3 (SS + Geometry + Material) |
| **Pose prediction** | Yes (built-in) | No | No |
| **3D representation** | SLAT (64^3 voxels) | SLAT (64^3 voxels) | O-Voxel (octree, up to 1536^3) |
| **Mesh decoder** | FlexiCubes + Gaussian Splatting | FlexiCubes + Gaussian Splatting | Direct mesh from O-Voxels |
| **Texture** | UV-mapped 1024^2 | UV-mapped 1024^2 | PBR 2048^2 (albedo+normal+roughness) |
| **Speed (per object)** | 9-33 min (RTX 5080) | 9-33 min (RTX 5080) | ~2 min (RTX 5080) |

---

## Detailed Pipeline: SAM3D Objects / TRELLIS v1

Both share the same codebase in this project. SAM3D **is** Meta's fork of TRELLIS v1
with three additions: (1) MoT for pose prediction, (2) pointmap conditioning, (3) progressive training.

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
├── embed_dim: 768 → projected to cond_channels: 4096
├── PointPatchEmbed (pointmap.py) — converts MoGe pointmap to patch tokens
└── Output: (N_patches, 4096) conditioning tensor

MODEL 2: ss_generator                   [~500MB-1GB]
├── SAM3D:  SparseStructureFlowModel (mot_sparse_structure_flow.py)
│   └── MOTModulatedTransformerCrossBlock × 20 blocks
│   └── model_channels: 1024, cond_channels: 4096, 16 heads
│   └── Multi-latent heads: "shape" (voxel) + "quaternion" + "translation" + "scale"
│   └── ShortCut solver, 2 inference steps
├── TRELLIS v1: SparseStructureFlowModel (sparse_structure_flow.py)
│   └── ModulatedTransformerCrossBlock × 20 blocks (standard, no MoT)
│   └── Same dims, but single latent head ("shape" only, no pose)
└── Output: shape latent (B, 4096, 8) + pose latents (SAM3D only)

MODEL 3: ss_decoder                     [~10MB]
├── SparseStructureDecoder (sparse_structure_vae.py)
├── 3D CNN: Conv3d → ResBlock3d × N → UpsampleBlock3d
├── latent (B, 8, 16, 16, 16) → occupancy (B, 1, 64, 64, 64)
└── Output: binary voxel grid, threshold > 0 → occupied coordinates

MODEL 4: slat_condition_embedder        [~400MB]
├── Same DINOv2 architecture as MODEL 1 (shared weights in practice)
└── Output: (N_patches, 4096) conditioning for SLAT stage

MODEL 5: slat_generator                 [~1GB]
├── SLatFlowModel (structured_latent_flow.py)
├── Sparse 3D U-Net:
│   ├── Input: sp.SparseLinear → io_block_channels [256, 512, 768]
│   ├── Down: SparseResBlock3d × 3 stages with SparseDownsample
│   ├── Middle: ModulatedSparseTransformerCrossBlock × 24 blocks
│   │   └── model_channels: 1024, cond_channels: 4096
│   ├── Up: SparseResBlock3d × 3 stages with SparseUpsample + skip connections
│   └── Output: sp.SparseLinear → 8-channel SLAT features per voxel
├── Euler solver, 12 inference steps
└── Output: SparseTensor(coords, 8D features) for each occupied voxel

MODEL 6: slat_decoder_gs                [~200MB]
├── SLatGaussianDecoder (decoder_gs.py)
├── Sparse transformer (SparseTransformerBase) + swin attention
├── Per-voxel output layout:
│   ├── _xyz:         32 gaussians × 3 positions    = 96 channels
│   ├── _features_dc: 32 gaussians × 1 × 3 colors   = 96 channels
│   ├── _scaling:     32 gaussians × 3 scales        = 96 channels
│   ├── _rotation:    32 gaussians × 4 quaternion     = 128 channels
│   ├── _opacity:     32 gaussians × 1 alpha          = 32 channels
│   └── Total: 448 output channels per sparse position
├── Hammersley sequence perturbation for gaussian offset diversity
└── Output: List[Gaussian] — 3D Gaussian Splat representation

MODEL 7: slat_decoder_mesh              [~200MB]
├── SLatMeshDecoder (decoder_mesh.py)
├── Sparse transformer + SparseSubdivideBlock3d (progressive subdivision)
├── FlexiCubes isosurface extraction (cube2mesh.py)
├── Texture baking: nvdiffrast → 1024×1024 UV atlas
└── Output: trimesh.Trimesh with UV-mapped texture
```

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

---

## Detailed Pipeline: TRELLIS.2

### Input

```
RGBA image (mask baked into alpha channel)
No pointmap — TRELLIS.2 does not use MoGe
```

### Models

```
Pipeline: Trellis2ImageTo3DPipeline (from trellis2.pipelines)
Model:    microsoft/TRELLIS.2-4B (~4.7B parameters)

The TRELLIS.2 pipeline is a single model with 3 internal stages:

STAGE 1: Sparse Structure DiT
├── Vanilla DiT (Diffusion Transformer)
├── Predicts O-Voxel occupancy (octree-based, not dense 64^3)
├── ~1.3B parameters
└── No pose prediction

STAGE 2: Geometry DiT
├── Generates geometry attributes per occupied O-Voxel
├── Outputs vertex positions for mesh extraction
└── ~1.5B parameters

STAGE 3: Material DiT
├── Predicts PBR material properties:
│   ├── Albedo (base color)
│   ├── Normal map
│   └── Roughness
└── ~1.9B parameters

GLB Export: o_voxel.postprocess.to_glb()
├── Decimation target: 100K faces
├── Texture: 2048×2048
├── Remesh: True
├── Format: WebP-compressed textures
└── Output: trimesh.Trimesh → GLB binary
```

### Output

| Format | Contents |
|---|---|
| GLB | Mesh + PBR textures (albedo + normal + roughness), WebP compressed |
| NPZ (mesh) | vertices (float32), faces (int32) — raw canonical frame |

### No Pose Prediction

TRELLIS.2 outputs meshes in a **canonical frame** (centered, Y-up in GLB convention).
For scene alignment, we use **Module 4b** (SS Pose Estimation) which runs SAM3D's
SS model (MODEL 1 + MODEL 2 + MODEL 3 from above) to predict rotation/translation/scale,
then Module 5 (ICP registration) refines the alignment.

---

## Side-by-Side: Every Distinct Model

| # | Model Name | Architecture | Used In | Parameters | Purpose |
|---|---|---|---|---|---|
| 1 | **ss_condition_embedder** | DINOv2 ViT-B/14 + PointPatchEmbed | SAM3D, TRELLIS v1 | ~400MB | Encode image + pointmap → 4096-dim tokens |
| 2 | **ss_generator** (MoT) | MoT Flow Transformer, 20 blocks, 1024-dim | SAM3D only | ~500MB-1GB | Predict voxel occupancy + 6D pose |
| 2' | **ss_generator** (standard) | Flow Transformer, 20 blocks, 1024-dim | TRELLIS v1 only | ~500MB | Predict voxel occupancy (no pose) |
| 3 | **ss_decoder** | 3D CNN VAE decoder | SAM3D, TRELLIS v1 | ~10MB | Latent → 64^3 binary occupancy |
| 4 | **slat_condition_embedder** | DINOv2 ViT-B/14 (shared arch with #1) | SAM3D, TRELLIS v1 | ~400MB | Encode image → 4096-dim tokens for SLAT |
| 5 | **slat_generator** | Sparse 3D U-Net, 24 blocks, 1024-dim | SAM3D, TRELLIS v1 | ~1GB | Predict 8D SLAT features per voxel |
| 6 | **slat_decoder_gs** | Sparse Transformer + swin attn | SAM3D, TRELLIS v1 | ~200MB | SLAT features → 32 Gaussians/voxel |
| 7 | **slat_decoder_mesh** | Sparse Transformer + FlexiCubes | SAM3D, TRELLIS v1 | ~200MB | SLAT features → mesh + UV texture |
| 8 | **MoGe ViT-L** (depth model) | MoGe monocular depth | SAM3D, TRELLIS v1 | ~300MB | Image → 3D pointmap + intrinsics |
| 9 | **pose_decoder** | Pure function (no weights) | SAM3D only | 0 | SS pose latents → quaternion/T/S |
| 10 | **TRELLIS.2-4B** | 3-stage DiT (SS + Geometry + Material) | TRELLIS.2 only | ~4.7B | Image → O-Voxel → mesh + PBR |

**Total VRAM loaded:**
- SAM3D / TRELLIS v1: Models 1-8 = ~3-4GB (+ ~300MB MoGe)
- TRELLIS.2: Model 10 alone = ~9-10GB (fp16)
- Module 4b (SS pose for TRELLIS.2): Models 1-3 + 8 = ~1.2GB needed (currently loads all 1-8)

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
           20 blocks, 2 steps
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
   occupied            │  ← SS Pose: used as initial guess
   coordinates         │     for layout_post_optimization
        │              │
        ▼              │
   [MODEL 4]           │
   DINOv2              │
   (slat_cond)         │
        │              │
        ▼              │
   [MODEL 5]           │
   SLAT Generator      │
   24 blocks, 12 steps │
        │              │
   8D features         │
   per voxel           │
        │              │
   ┌────┴────┐         │
   ▼         ▼         │
[MODEL 6] [MODEL 7]   │
GS Decoder Mesh Dec.   │
(32 gauss/ (FlexiCubes │
 voxel)    + 1024^2    │
   │       texture)    │
   │         │         │
   ▼         ▼         ▼
  PLY       GLB     Pose NPZ
```

### TRELLIS v1 (Microsoft) — no pose

Same as above but:
- MODEL 2 uses standard `ModulatedTransformerCrossBlock` (no MoT, no pose heads)
- No pose_decoder
- Output: canonical-frame mesh only (requires external alignment)

### TRELLIS.2 (Microsoft)

```
             Image (RGBA)
                  │
                  ▼
          ┌──────────────┐
          │  TRELLIS.2   │
          │  (4.7B)      │
          │              │
          │ Stage 1: SS  │──→ O-Voxel occupancy
          │   (DiT)      │
          │              │
          │ Stage 2:     │──→ Geometry attributes
          │   Geometry   │
          │   (DiT)      │
          │              │
          │ Stage 3:     │──→ PBR materials
          │   Material   │    (albedo, normal, roughness)
          │   (DiT)      │
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
          → Requires Module 4b (SS Pose)
            to borrow SAM3D's SS model
```

---

## Key Differences Explained

### 1. Why does SAM3D predict pose but TRELLIS v1/v2 don't?

SAM3D is designed for **scene reconstruction** — placing multiple objects in their
correct 3D positions from a single photo. Microsoft's TRELLIS models are designed
for **single-object generation** in a canonical frame (centered, normalized scale).

Meta added pose prediction by replacing the standard transformer blocks with MoT
(Mixture of Transformers) that have **multiple latent heads**: one for voxel shape
and others for rotation/translation/scale. The MoT was trained with pose supervision
from multi-view datasets.

### 2. Why is TRELLIS.2 faster?

- **O-Voxel representation**: Octree-based sparse voxels scale hierarchically (only
  subdivide where needed), vs TRELLIS v1's dense 64^3 grid
- **No dual decoder**: TRELLIS.2 extracts mesh directly from O-Voxels, while
  TRELLIS v1 runs a Gaussian Splatting decoder (32 gaussians/voxel × all voxels)
  which dominates ~50% of per-object time
- **Unified model**: All 3 stages in one forward pass, vs TRELLIS v1's separate
  models with independent loading and inference

### 3. Why is TRELLIS.2 texture quality better?

- **Stage 3 (Material DiT)**: A dedicated ~1.9B parameter model for PBR prediction.
  TRELLIS v1 bakes texture from Gaussian splat colors via nvdiffrast — no explicit
  material model.
- **2048^2 vs 1024^2**: Double the texture resolution
- **Full PBR**: albedo + normal + roughness vs just albedo

### 4. SLAT vs O-Voxel

| | SLAT (SAM3D / TRELLIS v1) | O-Voxel (TRELLIS.2) |
|---|---|---|
| Structure | Dense 64^3 grid | Octree (hierarchical) |
| Max resolution | 64^3 = 262K voxels | Up to 1536^3 |
| Per-voxel features | 8D latent vector | Direct geometry + material |
| Decoder required | Yes (GS + Mesh) | No (mesh extracted directly) |
| Memory | O(N^3) = fixed 262K positions | O(occupied) = adaptive |

---

## Module 4b: Bridging the Pose Gap

Since TRELLIS.2 has no pose prediction, we use SAM3D's SS model as a standalone
pose estimator:

```
Module 3 output:  pointmap.npz (MoGe depth)
Module 1 output:  per-object masks

     ┌──────────────────────────────┐
     │     Module 4b: SS Pose       │
     │                              │
     │  Load: MODEL 1 + 2 + 3 + 8  │
     │  (ss_cond + ss_gen + ss_dec  │
     │   + MoGe depth model)        │
     │                              │
     │  Run: stage1_only=True       │
     │  → pose_decoder()            │
     │                              │
     │  Output: rotation (quat),    │
     │          translation (3D),   │
     │          scale (3D)          │
     └──────────────┬───────────────┘
                    │
                    ▼
            Module 5: ICP Registration
            (uses SS pose as initial guess
             instead of 4 discrete Y-rotations)
```

This replaces the heuristic multi-start rotation (4 discrete Y-axis angles, avg IoU 0.16)
with learned continuous rotation (avg IoU expected to improve significantly).
