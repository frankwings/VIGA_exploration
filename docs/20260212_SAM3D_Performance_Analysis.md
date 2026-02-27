# SAM3D Performance Analysis — Why 1.5-2 Hours?

**Date:** 2026-02-12
**Author:** kingyy (win/vscode/opus/hum)

Breaking down the SAM3D local runtime vs Meta's 1-2 minute online demo.

---

## 1. Local Pipeline Timing Breakdown (Per Object)

The pipeline uses **TRELLIS** (Microsoft's flow-matching generative model), NOT SDS optimization.

| Stage | What Happens | Time | Steps |
|---|---|---|---|
| Model Loading | Load DINOv2 + TRELLIS checkpoints to GPU | ~26 sec | once per object |
| Stage 1: Sparse Structure | Flow matching → 16³ voxel occupancy grid | ~8-14 sec | **2 steps** (Euler solver, ShortCut) |
| Stage 2: Sparse Latent (SLAT) | Flow matching → 8D latent per occupied voxel | ~20-74 sec | **12 steps** (Euler solver) |
| Stage 3: Dual Decoding | Gaussian Splatting + Mesh decoder | **~420 sec (7 min)** | per-voxel expansion |
| Post-processing | Export GLB | <1 sec | — |

**Total per object: ~9-33 minutes** (depends on object complexity)
**5 objects × sequential processing = 1.5-2 hours**

---

## 2. Three Biggest Bottlenecks

### 2.1 Mesh + Gaussian Decoding (50% of time)

The dual decoder generates:
- ~300K+ vertices and ~750K+ faces per object
- 32 Gaussians per voxel across a 64³ resolution space
- Both GS (appearance) and Mesh (geometry) decoders run per voxel

This is **hardcoded in the checkpoint weights** — `num_gaussians=32` and `resolution=64` cannot be reduced without retraining.

### 2.2 Flow Matching SLAT Sampling (30% of time)

- 12 Euler solver steps × number of occupied voxels (e.g., 18,578 for a bottle)
- Each step requires a full forward pass through a 24-block DIT model (1024 channels, 16 attention heads)
- Total neural network evaluations per object: 12 + 2 = **14 DIT forward passes minimum**

### 2.3 Sequential Object Processing (15% of time)

From `tools/sam3d/init.py`:
```python
for task in tasks:
    success, glb_path, object_transform, error_msg = process_single_object(task)
```
Objects are processed one-at-a-time. No parallelism.

---

## 3. Meta's 1-2 Minute Demo

The Meta SAM3D online demo achieves 1-2 minutes likely through:

| Factor | Demo | Local Run |
|---|---|---|
| **Solver** | Distilled / ShortCut (1-4 steps) | Standard Euler (14 steps total) |
| **Decoder** | Possibly lighter (fewer Gaussians) | 32 Gaussians/voxel, 64³ resolution |
| **Objects** | 1 object | 5 objects (sequential) |
| **GPU** | A100/H100 (server-grade) | RTX 5080 (16GB consumer) |
| **Precision** | Likely optimized kernels | float16 (already optimized) |

Note: Stage 1 (sparse structure) already uses a `ShortCut` solver class locally — that's why it only takes 2 steps. But Stage 2 (SLAT) still uses standard flow matching, and Stage 3 (decoding) is the dominant bottleneck.

---

## 4. Common Misconception: SDS vs Flow Matching

### What is NOT happening (SDS / Score Distillation Sampling):
```
❌ Freeze 3D model → Render → Compare to image → Adjust mesh → Repeat 2,000-10,000×
```

SDS is a **test-time optimization** loop. The 3D model is iteratively refined by comparing renders against the input. This takes thousands of gradient steps.

### What IS happening (Flow Matching — Feed-Forward):
```
✅ Image → DINOv2 embedding → Flow Matching (14 steps) → Voxel Latents → Decode → Mesh + Colors
```

TRELLIS uses **feed-forward inference**. There is no render-compare-adjust loop. The model directly predicts the 3D shape from the image in a fixed number of flow steps. It's slow because of:
- The number of flow steps (14 vs 1-4 for distilled models)
- The heavy decoder architecture
- Sequential object processing

### Corrected Comparison

| | Demo (Fast) | Local Run (Slow) |
|---|---|---|
| **Category** | Feed-forward (distilled) | Feed-forward (standard) |
| **Method** | Distilled flow matching (shortcut solver) | Standard flow matching (Euler solver) |
| **Flow steps** | 1-4 steps | 14 steps (2 + 12) |
| **Decoder** | Possibly lighter | 32 Gaussians/voxel, 64³, ~7 min decode |
| **Objects** | 1 | 5 (sequential) |
| **Result** | 1-2 min | 1.5-2 hours |

Both are feed-forward inference. The difference is solver efficiency and decoder weight, not a fundamentally different algorithm.

---

## 5. Key Configuration Files

| File | Parameter | Value | Tuneable? |
|---|---|---|---|
| `ss_generator.yaml` | `inference_steps` | 2 | Yes |
| `slat_generator.yaml` | `inference_steps` | 12 | Yes (quality tradeoff) |
| `pipeline.yaml` | `dtype` | float16 | Already optimized |
| `slat_decoder_gs.yaml` | `num_gaussians` | 32 | **No** (checkpoint weights) |
| `slat_decoder_gs.yaml` | `resolution` | 64 | **No** (checkpoint weights) |
| `slat_decoder_mesh.yaml` | mesh decoder | ~300K vertices | **No** (checkpoint weights) |

---

## 6. Potential Optimizations

| Optimization | Impact | Feasibility |
|---|---|---|
| Reduce `slat_inference_steps` 12 → 6-8 | ~15-25% faster | Easy, but quality loss |
| Parallel object processing | ~3-4x faster for multi-object | Moderate (VRAM limited) |
| Cache DINOv2 embeddings | ~26 sec saved per object | Easy |
| Smaller input images | Minor speedup | Easy |
| Use distilled solver for SLAT | Major speedup (like demo) | Hard (need distilled weights) |
| Lighter decoder (4 Gaussians/voxel) | Major speedup | Hard (weight mismatch with current checkpoint) |

---

## 7. Subprocess Commands

**SAM Segmentation:**
```bash
python tools/sam3d/sam_worker.py --image <input> --out <masks> --vlm-model gpt-4o
```

**SAM3D Reconstruction (per object):**
```bash
python tools/sam3d/sam3d_worker.py \
  --image <input_image> \
  --mask <mask.npy> \
  --config utils/third_party/sam3d/checkpoints/hf/checkpoints/pipeline.yaml \
  --glb <output.glb> \
  --info <transform.json>
```

---

**Hardware:** RTX 5080 16GB | Ryzen 9 9900X | 32GB DDR5-6000 | Windows 11
**Pipeline:** TRELLIS (Microsoft) via SAM3D integration in GenesisVIGA
