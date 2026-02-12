# SAM3D Internals

**Pipeline Architecture, Tunable Parameters & VRAM Optimization**

`SAM3D / TRELLIS`

---

## Overview

SAM3D is a single-image to 3D reconstruction tool built on Microsoft's **TRELLIS** architecture. It takes an RGB image + mask and outputs a 3D mesh (GLB/PLY) with vertex colors.

> **Key Insight:** SAM3D uses monocular depth estimation (MoGe) to get partial 3D from the image, then a generative model "hallucinates" the unseen parts (back, sides) based on training data.

### How Single-Image to 3D Works

```
Single Image --> MoGe Depth Model --> 2.5D Pointmap (front view only)
                                        |
                              TRELLIS Generative Model
                                        |
                              Full 3D Voxel Grid (including back/sides)
```

The back and sides are *generated*, not observed. Running the same image multiple times may produce slightly different results.

---

## Pipeline Stages

### Stage 0: Preprocessing

**Models:** MoGe + DINOv2

Depth estimation --> Pointmap generation --> Image/mask resize to 518x518 --> Feature extraction

| Input | Output | VRAM |
|---|---|---|
| RGB Image + Mask | Condition embeddings | ~2-3 GB |

### Stage 1: Sparse Structure Sampling

**Model:** `ss_generator`

Determines which voxels in a 16^3 grid are occupied (where the object exists in 3D space).

| Resolution | Output | VRAM |
|---|---|---|
| 16^3 = 4,096 voxels | N occupied voxels | +2-4 GB |

### Stage 2: Sparse Latent Sampling

**Model:** `slat_generator`

For each occupied voxel, generates an 8-dimensional latent vector encoding appearance/geometry.

| Latent Dim | Output | VRAM |
|---|---|---|
| 8 channels | N x 8 feature matrix | +2-3 GB |

### Stage 3: Decode (Dual Decoder) -- PEAK VRAM

**Models:** `slat_decoder_gs` + `slat_decoder_mesh`

Two parallel decoders: one generates Gaussian Splatting (colors), one generates Mesh (geometry). Results are merged.

| GS Decoder | Mesh Decoder | VRAM |
|---|---|---|
| N x num_gaussians points | Vertices + Faces | +4-6 GB (PEAK) |

### Export

Mesh geometry + GS vertex colors --> GLB file with vertex colors

> **Warning: OOM typically occurs in Stage 3 (Decode)**
>
> This is the peak VRAM stage. A 3200x2082 image on RTX 5080 (16GB) peaked at ~14.9GB.

---

## Tunable Parameters

| Parameter | Location | Default | Tunable? | Effect |
|---|---|---|---|---|
| `num_gaussians` | slat_decoder_gs.yaml | 32 | No* | Gaussians per voxel (weights tied) |
| `resolution` (ss) | ss_generator.yaml | 16 | No* | Voxel grid resolution |
| `resolution` (decoder) | slat_decoder_gs.yaml | 64 | No* | Output resolution |
| `input image size` | runtime | any | Yes | Condition quality |
| `dtype` | pipeline.yaml | float16 | Careful | Precision (float16/float32) |

\* Resolution parameters are tied to learned position embeddings. Changing them requires retraining.

### Understanding num_gaussians

This is the most important tunable parameter for VRAM optimization:

```
Total Gaussian points = Occupied_Voxels x num_gaussians

Example (bottle test):
- Occupied voxels: ~580
- num_gaussians: 32 (default)
- Total points: 580 x 32 = 18,560 Gaussians

With num_gaussians: 8
- Total points: 580 x 8 = 4,640 Gaussians (75% reduction)
```

> **Quality vs VRAM tradeoff:**
> Lower `num_gaussians` = fewer points = less detail per region, but same overall shape (voxel structure unchanged).

---

## VRAM Optimization

> **Important Discovery (2026-02-03):**
>
> `num_gaussians` CANNOT be changed at runtime! The checkpoint weights are trained for `num_gaussians=32` specifically. Changing the config causes shape mismatch errors:
>
> ```
> size mismatch for offset_perturbation: checkpoint [32, 3] vs model [8, 3]
> ```

### What Actually Works

| Method | Works? | Notes |
|---|---|---|
| Reduce input image size | Yes | Smaller image = less VRAM in all stages |
| Post-processing downsample | Yes | Generate full, then reduce points |
| ~~Change num_gaussians~~ | No | Weights tied to 32 |
| ~~Change resolution~~ | No | Weights tied to 16^3/64^3 |

### Post-processing Downsample (Recommended)

If you need to reduce point count after generation:

```bash
# Voxel downsample
python tools/downsample_pointcloud.py output.glb --voxel 0.02 -o output_down.ply

# Uniform downsample (keep every 2nd point)
python tools/downsample_pointcloud.py output.glb --uniform 2 -o output_down.ply

# Random downsample (keep 50%)
python tools/downsample_pointcloud.py output.glb --random 0.5 -o output_down.ply
```

---

## File Structure

```
GenesisVIGA/
├── tools/
│   └── sam3d/
│       └── sam3d_worker.py          # Entry point (called by VIGA)
│
└── utils/third_party/sam3d/
    ├── sam3d_objects/
    │   ├── pipeline/
    │   │   └── inference_pipeline.py # Main inference logic
    │   └── models/
    │       ├── sparse_structure/     # Stage 1 model
    │       ├── structured_latent/    # Stage 2 model
    │       └── slat_decoder_*/       # Stage 3 decoders
    │
    └── checkpoints/hf/checkpoints/
        ├── pipeline.yaml             # Main config (only usable one)
        ├── ss_generator.yaml         # Stage 1 config
        ├── slat_generator.yaml       # Stage 2 config
        ├── slat_decoder_gs.yaml      # GS decoder (num_gaussians=32, fixed)
        └── slat_decoder_mesh.yaml    # Mesh decoder
```

> **Key Point:** All SAM3D configs are in `utils/third_party/sam3d/`. VIGA just calls `sam3d_worker.py` with a config path.

---

## Relationship to VIGA

```
VIGA Workflow:
┌─────────────────────────────────────────────────────────────────┐
│  Video --> Frames --> SAM2 Segmentation --> Masks               │
│                                          |                      │
│                              ┌───────────────────────┐          │
│                              │       SAM3D           │          │
│                              │  (This document)      │          │
│                              │                       │          │
│                              │  Image + Mask --> GLB  │          │
│                              └───────────────────────┘          │
│                                          |                      │
│                              Scene Composition --> Final Output  │
└─────────────────────────────────────────────────────────────────┘
```

SAM3D is a **self-contained module** within VIGA. All the parameters and configs discussed in this document are SAM3D-specific, not VIGA-specific.

---

## Quick Reference

#### To reduce VRAM usage:

1. **Reduce input image size** (target <2M pixels for 16GB VRAM)
2. Post-process with downsampling tool after generation

#### To improve quality:

1. Use higher resolution input images (if VRAM allows)
2. Ensure clean masks (no noise)
3. Use Windows native instead of WSL (better VRAM efficiency)

#### Cannot change without retraining:

- `num_gaussians` (fixed at 32, weights tied)
- Voxel grid resolution (16^3 for sparse structure)
- Decoder resolution (64^3)
- Latent dimensions (8 channels)

---

Created: 2026-02-03 15:47 PST
Author: Yuna
Category: SAM3D / TRELLIS (separate from VIGA workflow docs)
