# SAM3D Pipeline: Image to 3D GLB Objects

**Date:** 2026-02-16
**Code:** `utils/third_party/sam3d/` (Meta's original), `tools/sam3d/` (VIGA wrappers)

## Overview

SAM3D takes a single scene image and produces per-object 3D meshes (GLB files) positioned in camera space. It combines three models — SAM (segmentation), MoGe (monocular geometry), and TRELLIS (3D reconstruction) — with a multi-stage pose alignment pipeline.

## Pipeline Steps

```
Input Image (e.g., dining scene photo)
    |
    v
[Step 1] SAM Segmentation ─────────── N binary masks
    |
    v
[Step 2] MoGe Depth Estimation ────── (H,W,3) pointmap + intrinsics
    |
    v
[Step 3] TRELLIS 3D Reconstruction ── per-object GLB mesh (model space)
    |
    v
[Step 4] Pose Decoding ────────────── initial S, R, T per object
    |
    v
[Step 5] Layout Post-Optimization ─── refined S, R, T (3 sub-stages)
    |
    v
[Step 6] Export ────────────────────── GLBs with baked transforms
```

---

## Step 1: SAM Segmentation

**Conda env:** `sam` (Python 3.10)
**Model:** SAM ViT-H (Meta's Segment Anything Model)

Takes the full scene image and produces per-object binary masks. Each mask is a (H, W) uint8 array with values 0 (background) or 255 (object).

**Input:** RGB image
**Output:** N binary masks (one per detected object)

---

## Step 2: MoGe Depth Estimation

**Conda env:** `sam3d_py311` (Python 3.11)
**Model:** MoGe (Monocular Geometry estimator)

Runs on the **full scene image** (not per-object). Produces a dense **pointmap** — a (H, W, 3) array where each pixel (x, y) maps to a 3D point (X, Y, Z) in camera space.

**What is a pointmap?**
A pointmap is NOT an RGB image. It is a per-pixel 3D coordinate map:
- `pointmap[y, x] = (X, Y, Z)` — the 3D camera-space position of pixel (x, y)
- The Z channel is depth, but the full (X, Y, Z) provides complete spatial information
- This is richer than a depth map alone because it includes the lateral (X, Y) positions

MoGe also estimates camera intrinsics (focal length, principal point).

**Input:** RGB image
**Output:**
- `points` — (H, W, 3) dense pointmap in camera space
- `intrinsics_px` — (3, 3) camera intrinsic matrix in pixel coordinates
- `depth` — (H, W) depth map (Z channel of pointmap)

---

## Step 3: TRELLIS 3D Reconstruction

**Conda env:** `sam3d_py311` (Python 3.11)
**Model:** TRELLIS (feed-forward flow matching)

For **each** object mask:
1. Crop the image to the mask bounding box, apply alpha from the mask
2. Feed the RGBA image to TRELLIS

**What TRELLIS receives:**
- **RGBA image only** (RGB + alpha mask from segmentation)
- TRELLIS does NOT receive the MoGe depth/pointmap for shape reconstruction
- The MoGe pointmap is used only as conditioning for the sparse structure generator (Stage 1) to help with pose prediction, but the actual shape reconstruction (SLAT / Stage 2) uses only the image

**TRELLIS internal stages:**
1. **Sparse Structure (Stage 1):** Flow matching with ShortCut solver (2 steps). Receives image + pointmap conditioning. Outputs shape latent + pose heads (rotation, translation, scale)
2. **Sparse Latent / SLAT (Stage 2):** Flow matching with Euler solver (12 steps). Receives image ONLY (no pointmap). Outputs detailed shape latent
3. **Dual Decode (Stage 3):** Decodes latent to 3D Gaussians (32 per voxel, 64^3 resolution) then extracts mesh (~300K vertices)

**Input:** RGBA image (per object)
**Output:** Mesh in TRELLIS model space (Z-up, ~[-0.5, 0.5] range)

---

## Step 4: Pose Decoding

**Conda env:** `sam3d_py311`
**Component:** Deterministic math function (NOT a neural network)

The sparse structure generator (Step 3, Stage 1) is an MM-DiT model with **multiple output heads** that jointly predict:
- Shape latent
- 6D rotation
- Translation (in log-space)
- Scale (in log-space)

The "pose decoder" is a pure Python function that converts these raw outputs into usable pose parameters:
1. Converts 6D rotation to quaternion (via Gram-Schmidt orthogonalization)
2. Exponentiates scale and translation (predicted in log-space)
3. Undoes the MoGe pointmap normalization (scale/shift)
4. Applies the pose target convention (e.g., `ScaleShiftInvariant`)

**Input:** Raw network outputs from sparse structure generator + MoGe scale/shift
**Output:** Per-object quaternion (wxyz), translation (3D), scale (uniform)

---

## Step 5: Layout Post-Optimization

**Conda env:** `sam3d_py311`
**Component:** Part of Meta's original SAM3D code (`layout_post_optimization_utils.py`)

This is a **3-stage refinement** that adjusts each object's pose to better match the scene. It is NOT something we added — it has Meta's copyright headers.

### Stage 5a: Manual Alignment

Aligns the mesh to the MoGe pointmap within the object's mask:
1. Extract 3D points from the MoGe pointmap within the mask (with outlier filtering at 90th depth percentile)
2. Scale the mesh to match the height of the target point cloud
3. Translate the mesh centroid to match the target centroid
4. Compute silhouette IoU as a quality check

**Space:** 3D camera space
**Target:** Masked MoGe pointmap (NOT full image — only pixels within the object mask)

### Stage 5b: ICP (Iterative Closest Point)

Open3D point-to-point ICP between mesh vertices and MoGe pointmap points:
- Operates in **3D space** (not 2D projections)
- Uses the **masked** MoGe pointmap only (points within the object mask, outlier-filtered)
- Threshold: 0.05 (units in camera space)
- **Accepted only if** silhouette IoU improves compared to pre-ICP state

**Space:** 3D camera space
**Target:** Masked MoGe pointmap

### Stage 5c: Differentiable Rendering Optimization

Uses PyTorch3D to render the mesh and optimize the pose against the ground-truth segmentation mask.

**What is differentiable rendering?**
A rendering technique where gradients can flow backward through the render operation. This allows optimizing 3D parameters (rotation, translation, scale) by comparing the rendered output to a target image using gradient descent.

**Renderer:** PyTorch3D `MeshRenderer` with `SoftSilhouetteShader`
**What it renders:** Soft silhouette ONLY (alpha channel). It does NOT render depth or color.

**Parameters optimized:**
- Quaternion (rotation)
- Translation (3D)
- Scale (uniform)

**Two-stage optimization:**
- Stage 1 (5 iterations, lr=1e-2): Translation + Scale only
- Stage 2 (25 iterations, lr=5e-3): Quaternion + Translation + Scale

**Loss function:**
```
L = 200 * MSE(rendered_silhouette, gt_mask)
  + 0.1 * MSE(quaternion, identity)
  + 0.05 * ||translation||^2
  + 0.05 * (scale - 1)^2
```

**Accepted only if:** Final IoU > 0.5 AND better than pre-optimization IoU.

**Critical limitation:** The loss is silhouette-only. There is **no depth loss term**. This means objects can achieve good 2D alignment (high mask IoU) while having incorrect depth placement. This is the root cause of the depth alignment issues we diagnosed.

---

## Step 6: Export

1. Apply Z-up to Y-up rotation to TRELLIS mesh
2. Bake the final S, R, T into vertex positions (PyTorch3D camera space)
3. Export as `.glb` per object
4. Save metadata:
   - `object_transforms.json` — all objects' translation, rotation (wxyz quaternion), scale
   - `target_moge.npz` — MoGe pointmap, intrinsics, depth
   - `{name}.npy` — per-object binary masks

---

## Conda Environment Summary

| Conda Env | Python | Used By | Contains |
|-----------|--------|---------|----------|
| `sam3d` | 3.11 | Orchestrator (`tools/sam3d/init.py`) | Subprocess management |
| `sam` | 3.10 | SAM segmentation (`sam_worker.py`) | SAM ViT-H, PyTorch |
| `sam3d_py311` | 3.11 | TRELLIS + MoGe + layout opt (`sam3d_worker.py`) | TRELLIS, MoGe, PyTorch3D, Open3D |

Three separate envs are needed because SAM and TRELLIS have incompatible PyTorch/CUDA dependencies. The orchestrator spawns each as a subprocess in its own env.

---

## Coordinate Systems

```
TRELLIS model space:   Z-up, range ~[-0.5, 0.5]
PyTorch3D camera:      X-left, Y-up, Z-forward
OpenCV camera:         X-right, Y-down, Z-forward

Conversion (PT3D -> OpenCV): X_cv = -X_pt3d, Y_cv = -Y_pt3d, Z_cv = Z_pt3d
Conversion (TRELLIS -> PT3D): Z-up to Y-up rotation, then apply S/R/T

Projection: u = fx * X_cv / Z_cv + cx
            v = fy * Y_cv / Z_cv + cy

Quaternion convention: wxyz (w = real part first)
```

---

## Key Files

| File | Role |
|------|------|
| `tools/sam3d/init.py` | MCP tool server (orchestrator) |
| `tools/sam3d/sam_worker.py` | SAM segmentation subprocess |
| `tools/sam3d/sam3d_worker.py` | TRELLIS + MoGe + layout subprocess |
| `tools/sam3d/adapters/trellis2_adapter.py` | TRELLIS.2 adapter (simpler, no layout opt) |
| `utils/third_party/sam3d/sam3d_objects/pipeline/inference_pipeline_pointmap.py` | Main inference pipeline |
| `utils/third_party/sam3d/sam3d_objects/pipeline/inference_utils.py` | Pose decoder + layout orchestration |
| `utils/third_party/sam3d/sam3d_objects/pipeline/layout_post_optimization_utils.py` | ICP + differentiable rendering |
| `utils/third_party/sam3d/sam3d_objects/pipeline/depth_models/moge.py` | MoGe wrapper |
