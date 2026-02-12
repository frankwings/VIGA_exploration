# VIGA Architecture

**Hierarchical relationship between VIGA, SAM3D, and TRELLIS**

---

## Architecture Hierarchy

```
VIGA          -- Your complete project: Video --> 3D reconstruction pipeline
  |
  v
SAM3D         -- Single-image to 3D tool (core module of VIGA)
  |
  v
TRELLIS       -- Microsoft's 3D generation model architecture (underlying engine)
```

> **Simple analogy:**
> VIGA is your car, SAM3D is the engine, TRELLIS is the core technology inside the engine.

---

## Component Details

### VIGA (Video-to-3D Gaussian Avatar)

Your complete workflow, including:

- Video input & frame extraction
- SAM2 object segmentation
- SAM3D 3D reconstruction
- Scene composition & export

### SAM3D (Segment Anything Model 3D)

VIGA's 3D reconstruction module:

- Input: RGB image + Mask
- Output: GLB/PLY 3D model
- Uses TRELLIS architecture underneath

### TRELLIS (Microsoft)

SAM3D's neural network engine:

- Sparse Structure sampling
- Sparse Latent representation
- Dual decoder architecture (GS + Mesh)

---

## TRELLIS Inference Pipeline

Internal flow when SAM3D invokes TRELLIS:

```
Image + Mask --> Condition Embedder --> Sparse Structure --> Sparse Latent --> Dual Decoder
```

### Dual Decoder Architecture

| Decoder | Output | Purpose |
|---|---|---|
| `slat_decoder_gs` | Gaussian Splatting | Color & texture information |
| `slat_decoder_mesh` | Mesh geometry | Vertex & face structure |

Both decoders run **in parallel**, then merge: Mesh provides geometry, GS provides vertex colors --> export as GLB.

---

## Code Path Mapping

```
GenesisVIGA/
├── tools/
│   └── sam3d/
│       └── sam3d_worker.py          # VIGA entry point
│
└── utils/third_party/sam3d/
    └── sam3d_objects/
        ├── pipeline/
        │   └── inference_pipeline.py # TRELLIS inference flow
        │
        └── models/
            ├── sparse_structure/     # Sparse structure model
            ├── structured_latent/    # Sparse latent model
            └── slat_decoder_*/       # Dual decoders
```

---

## VRAM Usage Distribution

| Stage | TRELLIS Component | VRAM Usage | Duration |
|---|---|---|---|
| Model Loading | All models | ~4-6 GB | ~20-30s |
| Stage 1 | `sparse_structure_sampler` | +2-4 GB | ~10-60s |
| Stage 2 | `slat_sampler` | +2-3 GB | ~15-80s |
| Stage 3 | `slat_decoder_*` | +4-6 GB (peak) | ~1-7min |

> **Warning: OOM typically occurs in Stage 3 (Decode)**
>
> This is the peak VRAM stage. A 3200x2082 image on RTX 5080 (16GB) peaked at ~14.9GB.

---

## VRAM Optimization Configs

Pre-configured pipeline variants for different VRAM budgets:

| Config File | num_gaussians | Point Count | VRAM |
|---|---|---|---|
| `pipeline.yaml` | 32 | 100% | High (default) |
| `pipeline_midvram.yaml` | 16 | 50% | Medium |
| `pipeline_lowvram.yaml` | 8 | 25% | Low |

### Usage

```bash
# Low VRAM mode
python sam3d_worker.py --config ".../pipeline_lowvram.yaml" ...

# Medium VRAM mode
python sam3d_worker.py --config ".../pipeline_midvram.yaml" ...
```

---

## Related Documentation

- [SAM3D Bottle Test Report](20260203_SAM3D_BOTTLE_TEST.html)
- [SAM3D WSL Testing Notes](20260201_SAM3D_WSL_TESTING.html)
- [TRELLIS GitHub (Microsoft)](https://github.com/microsoft/TRELLIS)

---

Created: 2026-02-03 15:05 PST
Updated: 2026-02-03 15:24 PST
Author: Yuna
