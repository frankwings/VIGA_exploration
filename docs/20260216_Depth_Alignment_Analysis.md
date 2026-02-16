# Depth Alignment Diagnostic Analysis

**Date:** 2026-02-16
**Scene:** `output/sam3d_dining/` (9 objects)
**Tools:** `diagnose_depth_alignment.py` (numpy + trimesh), `render_depth_pass.py` (Blender 4.5)

## Executive Summary

MoGe estimates a depth map from the full dining scene image. SAM3D reconstructs 9 individual objects as GLB meshes with baked S, R, T transforms in PyTorch3D camera space. This analysis diagnoses why the reconstructed object depths don't perfectly match MoGe's global depth.

**Key findings:**
1. **Square pixel forcing (Test 1):** Not an issue — fx == fy exactly.
2. **Per-vertex depth error (Test 2):** 3 objects have >25% relative depth error; 4 objects are well-aligned (<5%).
3. **Depth consistency (Test 4):** Sofa and table centroid projections land on wrong MoGe depth regions (55% and 18% off).
4. **2D vs depth trade-off (Test 5):** Several objects have good 2D mask coverage but poor depth — the SAM3D layout optimizer prioritizes silhouette fit over depth accuracy.

## Test 1: Square Pixel Forcing

| Parameter | Value |
|-----------|-------|
| fx (px) | 701.1085 |
| fy (px) | 701.1085 |
| fx − fy | 0.0000 |
| cx, cy | 385.5, 512.0 |
| Image size | 771 × 1024 |

**Result:** fx == fy exactly. MoGe produced isotropic intrinsics for this image. No square-pixel forcing issue exists.

**Note:** The normalized intrinsics differ (fx_norm=0.909, fy_norm=0.685) because the image is not square, but this is correct — they're `fx/W` and `fy/H` respectively.

## Test 2: MoGe Depth vs GLB Vertex Depth

For each object, GLB vertices (PyTorch3D space → OpenCV) are projected to pixel coordinates. MoGe depth is sampled at those pixels and compared to vertex Z.

| Object | Median vZ | Median mZ | Median Err | \|Err\| Mean | Rel Error | Scale Ratio |
|--------|-----------|-----------|------------|------------|-----------|-------------|
| **newspaper** | 1.121 | 1.125 | −0.009 | 0.022 | **1.9%** | 0.996 |
| **placemat** | 1.447 | 1.443 | +0.006 | 0.020 | **1.4%** | 1.003 |
| **travel_pillow** | 2.068 | 2.070 | −0.014 | 0.097 | **5.3%** | 0.999 |
| chair_cushion | 1.289 | 1.255 | +0.025 | 0.147 | 11.7% | 1.027 |
| round_table | 1.441 | 1.411 | −0.017 | 0.197 | 12.7% | 1.022 |
| strainer | 1.682 | 1.500 | +0.121 | 0.174 | 10.7% | 1.121 |
| wooden_chair | 1.268 | 1.333 | −0.007 | 0.376 | **25.1%** | 0.951 |
| sofa | 2.084 | 1.883 | +0.284 | 0.689 | **51.2%** | 1.107 |
| **chair_legs** | 1.847 | 1.286 | +0.377 | 0.641 | **55.4%** | 1.436 |

### Observations

**Well-aligned objects (rel error < 5%):** newspaper, placemat, travel_pillow. These are compact, flat-ish objects where TRELLIS reconstruction closely matches the MoGe depth surface.

**Moderately misaligned (5–15%):** chair_cushion, round_table, strainer. The depth errors are dominated by the object's 3D extent — vertices at the front/back of the object span a depth range that the flat MoGe depth surface doesn't capture. The median error is small, suggesting the centroid placement is good but the shape thickness causes spread.

**Severely misaligned (>25%):** wooden_chair, sofa, chair_legs. These are large, complex 3D objects where:
- **chair_legs** has a scale ratio of 1.44 — it's placed 44% deeper than MoGe expects. This is the worst offender.
- **sofa** is 10.7% deeper at median but has 51% mean relative error due to its large depth extent.
- **wooden_chair** has low median error but 25% mean — it's well-centered but its back/legs extend far from the MoGe surface.

### Root Cause Analysis

The depth errors fall into two categories:

1. **Translation offset (systematic bias):** The SAM3D layout optimizer places the object centroid at the wrong depth. Visible in chair_legs (ratio=1.44) and sofa (ratio=1.11). This is a **layout optimization bug** — the optimizer converges on a 2D-correct but depth-incorrect solution.

2. **Shape thickness (spread around centroid):** Even well-placed objects show vertex-level errors because MoGe depth is a 2D surface while GLBs have full 3D extent. The back face of a sofa or underside of a table will always deviate from MoGe. This is **expected and not fixable** without switching to volumetric depth.

## Test 3: Blender Rendered Depth vs MoGe

The Blender render required a PyTorch3D → Blender coordinate conversion. Results:

| Metric | Value |
|--------|-------|
| Valid pixels | 789,504 / 789,504 (100%) |
| Mean error | −0.266 |
| Median error | −0.213 |
| \|Err\| mean | 0.342 |
| \|Err\| max | 1.858 |
| Rel mean | 19.8% |
| Scale ratio | 0.815 |

**Note:** The 100% valid pixel count indicates the background isn't being filtered (Blender writes clip_end distance for empty pixels). The scale ratio of 0.815 suggests the Blender coordinate transform may introduce a systematic shift. The per-vertex Test 2 results are more reliable for analysis since they bypass the Blender transform chain.

**Rendered depth range:** [0.734, 1.953] vs MoGe [0.698, 3.011]. The rendered depth is compressed — likely because the coordinate transform doesn't perfectly preserve the PyTorch3D→OpenCV→Blender mapping for all objects equally.

## Test 4: Per-Object Depth Consistency

Compares each object's translation Z (PyTorch3D→OpenCV centroid) vs MoGe depth at the projected centroid pixel.

| Object | Trans Z | MoGe Z | Offset | Ratio | Status |
|--------|---------|--------|--------|-------|--------|
| newspaper | 1.116 | 1.124 | −0.008 | 0.993 | OK |
| placemat | 1.447 | 1.439 | +0.008 | 1.006 | OK |
| chair_cushion | 1.290 | 1.280 | +0.009 | 1.007 | OK |
| chair_legs | 1.665 | 1.703 | −0.038 | 0.978 | OK |
| travel_pillow | 2.076 | 2.164 | −0.088 | 0.959 | OK |
| wooden_chair | 1.206 | 1.090 | +0.116 | 1.107 | Marginal |
| round_table | 1.408 | 1.185 | +0.223 | **1.188** | **18% off** |
| sofa | 2.067 | 1.324 | +0.743 | **1.561** | **55% off** |
| strainer | 1.427 | NaN | — | — | Out of bounds |

**Median ratio across objects:** 1.007 (nearly perfect overall scale).

**Anomalous objects:**
- **sofa_with_patterned_cover (55% off):** The sofa centroid projects to pixel (563, 341) which falls on the **table region** of the MoGe depth map, not the sofa. The sofa extends behind the table, so its centroid depth (2.07) is much larger than the MoGe depth at that pixel (1.32, which is the table surface). This isn't a scale error — it's a **projection occlusion issue**.
- **round_table_with_tablecloth (18% off):** Similarly, the table centroid at pixel (692, 392) is near the image edge where MoGe depth may be less accurate.
- **strainer:** Projects outside the image bounds (u=846, but image width=771).

## Test 5: Silhouette vs Depth Trade-off

| Object | 2D IOU | Mask Coverage | Depth \|Err\| | Depth Rel Err |
|--------|--------|--------------|-------------|-------------|
| placemat | **0.751** | 0.955 | 0.020 | **1.4%** |
| newspaper | **0.722** | 0.982 | 0.022 | **1.9%** |
| travel_pillow | **0.700** | 0.955 | 0.097 | **5.3%** |
| chair_cushion | **0.698** | 0.970 | 0.147 | 11.7% |
| chair_legs | 0.522 | 0.943 | 0.641 | 55.4% |
| round_table | 0.503 | 0.918 | 0.197 | 12.7% |
| sofa | 0.351 | **0.961** | 0.689 | **51.2%** |
| strainer | 0.327 | 0.356 | 0.174 | 10.7% |
| wooden_chair | **0.125** | 0.548 | 0.376 | 25.1% |

### Key Insight

**The sofa has 96% coverage but only 35% IOU with 51% depth error.** This means the projected vertices cover the mask well (the shape is in roughly the right place) but the IOU is low because the sofa mesh is much larger than its mask in 2D — it "bleeds" past the mask boundary. Meanwhile, its depth is 51% off.

**The wooden chair has only 12.5% IOU and 55% coverage** — it's substantially misplaced in 2D as well as depth. This may be a layout optimization failure.

**The strainer has 33% IOU and 36% coverage** — its centroid projects out of image bounds. The object is positioned incorrectly.

**Well-aligned objects** (placemat, newspaper, travel_pillow) show the expected pattern: high IOU, high coverage, low depth error. The SAM3D layout optimization worked well for these compact objects.

## Summary of Root Causes

| # | Root Cause | Severity | Objects Affected | Fixable? |
|---|-----------|----------|-----------------|----------|
| 1 | **Square pixel forcing** | None | None | N/A — fx == fy |
| 2 | **Layout optimization depth bias** | High | sofa (+10.7%), chair_legs (+43.6%), strainer (+12.1%) | Yes — re-run layout optimizer with depth loss term |
| 3 | **Shape thickness vs flat MoGe depth** | Medium | All thick objects (chair, sofa, table) | Expected — not a bug |
| 4 | **Centroid projection occlusion** | Medium | sofa, round_table | Yes — use object-masked MoGe depth for comparison |
| 5 | **Object extends outside image** | Low | strainer | Yes — clamp or reposition |

## Recommended Fixes

### Fix 1: Add depth loss to layout optimization (Highest Priority)

The SAM3D layout post-optimization (`layout_post_optimization_utils.py`) currently optimizes for 2D silhouette alignment (mask IOU). Adding a depth consistency loss would constrain the Z-translation:

```python
# In the layout optimization loss function:
# Sample MoGe depth at projected vertex locations
z_moge = sample_moge_depth(projected_uv, moge_depth)
z_vertex = vertices_opencv[:, 2]

# Depth loss: L1 on median depth (robust to shape thickness)
depth_loss = torch.abs(torch.median(z_vertex) - torch.median(z_moge))
total_loss = silhouette_loss + lambda_depth * depth_loss
```

**Expected impact:** Would fix sofa, chair_legs, and strainer depth placement. The `lambda_depth` weight should be tuned to not degrade 2D alignment.

### Fix 2: Use masked MoGe depth for centroid comparison

When comparing object centroid depth to MoGe, sample MoGe depth only within the object's mask, not at the centroid's projected pixel (which may land on an occluding surface):

```python
mask_pixels = np.argwhere(object_mask)  # (N, 2) [row, col]
moge_depths_in_mask = moge_depth[mask_pixels[:, 0], mask_pixels[:, 1]]
reference_depth = np.median(moge_depths_in_mask)
```

### Fix 3: Clamp object placement to image bounds

Objects whose centroids project outside the image (strainer: u=846 > width=771) should be flagged and repositioned during layout optimization.

## Appendix: Coordinate System Reference

```
PyTorch3D camera:  X-left,  Y-up,   Z-forward
OpenCV camera:     X-right, Y-down,  Z-forward
Conversion:        X_cv = -X_pt3d,  Y_cv = -Y_pt3d,  Z_cv = Z_pt3d

Projection:  u = fx * X_cv / Z_cv + cx
             v = fy * Y_cv / Z_cv + cy

Quaternion convention: wxyz (w = real part first)
Transform order: v' = v @ R_pre @ S @ R @ T  (right-multiply, homogeneous)
```

## Files

| File | Description |
|------|-------------|
| `diagnose_depth_alignment.py` | Diagnostic script (Tests 1, 2, 4, 5) |
| `render_depth_pass.py` | Blender Z-pass renderer (Test 3) |
| `output/sam3d_dining/depth_alignment_results.json` | Full numeric results |
| `output/sam3d_dining/rendered_depth.npy` | Blender-rendered depth map |
| `output/sam3d_dining/rendered_depth_vis.png` | Depth visualisation |
