# SAM3D Dining Scene — ICP Pose Refinement + 2D Overlay

**Date:** 2026-02-19
**Run:** `output/sam3d_dining_icp/`
**Scene:** Dining (9 objects)
**Script:** `run_icp_dining.py`
**Input:** `output/sam3d_dining_v4/` (v9 ray-cast mask growth GLBs)

---

## What Was Done

For each of the 9 dining-scene GLBs:
1. **V9 mask growth** — normal-consistency + 8-direction ray depth gate (same algorithm as `visualize_convex_hull_growth.py`) to expand the SAM mask toward its convex hull.
2. **ICP** — Open3D point-to-point ICP: source = mesh vertices (OpenCV space), target = MoGe pointmap within the grown mask.
3. **Depth-scale correction** — uniform scale from camera origin (preserves 2D projection, corrects depth offset).
4. **Rejection gate** — keep original if depth error increases after ICP+scale.
5. **Renders** — per-object 3-panel compare (masked photo | before | after) + photo overlay; scene-level overlays.

---

## Results Summary

| Object | Before | After | Delta | Status |
|---|---|---|---|---|
| chair_cushion | 2.7% | 2.7% | +0.0% | ACCEPTED |
| newspaper | 1.5% | 1.5% | +0.0% | kept |
| placemat | 0.5% | 0.5% | +0.0% | ACCEPTED |
| round_table_with_tablecloth | 10.3% | 10.3% | +0.0% | kept |
| sofa_with_patterned_cover | 7.4% | 7.1% | +0.3% | ACCEPTED |
| strainer | 2.5% | 2.5% | +0.0% | kept |
| travel_pillow | 2.5% | 2.5% | +0.0% | ACCEPTED |
| wooden_chair | 19.9% | 19.9% | +0.0% | kept |
| chair_legs | 5.3% | 5.3% | +0.0% | kept |

**4/9 objects accepted** (ICP improved or matched depth error).

The objects that are already well-aligned (`newspaper`, `placemat`, `travel_pillow`) have near-zero depth error — ICP yields negligible change. `sofa_with_patterned_cover` benefits most (+0.3%).

---

## Scene Overlay

### Side-by-side: Original Photo | 3D Objects Overlaid (alpha=0.6)

![scene_photo_comparison](test_results_images/sam3d_dining_icp/scene_photo_comparison.png)

All 9 ICP-aligned objects projected to 2D and blended onto the target photograph at alpha=0.6. Each object is assigned a distinct color. Vertices rendered as 3x3 blocks.

### Depth-Map Overlay: Before vs After ICP

![scene_2d_comparison](test_results_images/sam3d_dining_icp/scene_2d_comparison.png)

Left: before ICP | Right: after ICP. Background = MoGe depth (greyscale). Objects colored by index.

### After-ICP Scene on Depth Map

![scene_depth_after](test_results_images/sam3d_dining_icp/scene_depth_after.png)

---

## Per-Object Comparisons

Panel layout: **Masked photo** (green border = SAM mask) | **Before ICP** (vertex projection, depth-coded) | **After ICP**

### chair_cushion

![chair_cushion_compare](test_results_images/sam3d_dining_icp/chair_cushion_compare.png)

**+0.0%** (2.70%→2.70%). Tiny movement — the v4 GLB was already well-positioned. ICP accepted (scale=1.002).

![chair_cushion_overlay](test_results_images/sam3d_dining_icp/chair_cushion_overlay.png)

### newspaper

![newspaper_compare](test_results_images/sam3d_dining_icp/newspaper_compare.png)

**+0.0%** (1.52%→1.53%). ICP rejected — minor scale mismatch. Kept original (already 1.5% error).

![newspaper_overlay](test_results_images/sam3d_dining_icp/newspaper_overlay.png)

### placemat

![placemat_compare](test_results_images/sam3d_dining_icp/placemat_compare.png)

**+0.0%** (0.54%→0.54%). Already near-perfect alignment. ICP accepted (scale=0.999).

![placemat_overlay](test_results_images/sam3d_dining_icp/placemat_overlay.png)

### round_table_with_tablecloth

![round_table_with_tablecloth_compare](test_results_images/sam3d_dining_icp/round_table_with_tablecloth_compare.png)

**+0.0%** (10.26%→10.51%). ICP rejected. High base error (10.3%) reflects tablecloth geometry complexity — draping fabric has depth variation that the rigid ICP transform cannot resolve.

![round_table_with_tablecloth_overlay](test_results_images/sam3d_dining_icp/round_table_with_tablecloth_overlay.png)

### sofa_with_patterned_cover

![sofa_with_patterned_cover_compare](test_results_images/sam3d_dining_icp/sofa_with_patterned_cover_compare.png)

**+0.3%** (7.36%→7.08%). Best ICP improvement in this run. Large curved sofa — v9 grown mask (+12,639 px) provides good coverage for ICP target, allowing a small but consistent alignment improvement.

![sofa_with_patterned_cover_overlay](test_results_images/sam3d_dining_icp/sofa_with_patterned_cover_overlay.png)

### strainer

![strainer_compare](test_results_images/sam3d_dining_icp/strainer_compare.png)

**+0.0%** (2.52%→2.55%). ICP rejected. Small compact object — mask growth added +4,532 px but ICP converged to a slightly worse pose.

![strainer_overlay](test_results_images/sam3d_dining_icp/strainer_overlay.png)

### travel_pillow

![travel_pillow_compare](test_results_images/sam3d_dining_icp/travel_pillow_compare.png)

**+0.0%** (2.47%→2.46%). Marginal improvement (0.01%). Accepted.

![travel_pillow_overlay](test_results_images/sam3d_dining_icp/travel_pillow_overlay.png)

### wooden_chair

![wooden_chair_compare](test_results_images/sam3d_dining_icp/wooden_chair_compare.png)

**+0.0%** (19.88%→21.16%). ICP rejected — made things worse. High base error (19.9%) is a known issue: the sparse chair frame has many depth-inconsistent vertices (frame slats at different depths). ICP with sparse pointmap target cannot reliably handle this.

![wooden_chair_overlay](test_results_images/sam3d_dining_icp/wooden_chair_overlay.png)

### chair_legs

![chair_legs_compare](test_results_images/sam3d_dining_icp/chair_legs_compare.png)

**+0.0%** (5.29%→5.39%). ICP rejected. Large flat slab (floor area) — ICP fitness low (0.39), indicating poor correspondence between the flat floor geometry and the dense pointmap.

![chair_legs_overlay](test_results_images/sam3d_dining_icp/chair_legs_overlay.png)

---

## Observations

- **Objects already well-positioned** (newspaper 1.5%, placemat 0.5%): v4 ICP already found near-optimal poses. The v9 re-run confirms this — no improvement.
- **sofa_with_patterned_cover** benefits from v9 grown mask: +12,639 px additional target coverage allows ICP to find a slightly better alignment (+0.3%).
- **round_table and wooden_chair** have high persistent depth error due to geometry reasons (draping tablecloth, sparse frame), not pose. ICP with a rigid transform cannot resolve these.
- **chair_cushion** (`rel_err=2.7%`) is the persistent unsolved case from prior analysis — normal-gate bottleneck, and the TRELLIS geometry may not match the actual curved cushion shape.

---

## Algorithm

### V9 Mask Growth (ICP target selection)

```
1. Morphological opening (2x erosion + 2x dilation)
2. Convex hull of cleaned mask
3. Gaussian-smoothed normals (sigma=2.0)
4. Global reference normal = mean of cleaned mask normals
5. Adaptive threshold = clip(median + 2*std, 10 deg, 60 deg)
6. 8-direction ray first-hit precompute (scan-propagation, O(H*W))
7. Accept hull pixel P if:
     angle(normal_P, ref) < threshold  (normal-consistency gate)
     AND dmin <= depth[P] <= dmax       (8-dir ray depth range gate)
```

### ICP + Depth Scale

```
ICP: Open3D point-to-point, max_correspondence=0.3m, 100 iterations
     Source: mesh vertices (PT3D -> OpenCV)
     Target: pointmap[grown_mask] (OpenCV space, top-90th percentile depth)
Depth scale: s = median(z_moge / z_vertex), clamped [0.5, 2.0]
Rejection gate: keep original if rel_err increases after ICP+scale
```

### Overlay Composite

```
Per-object: 3x3 pixel dilation of projected vertices
            alpha=0.6 blend onto original target photo
Scene: all objects, distinct colors, back-to-front z-order
```

---

## Files

| File | Description |
|---|---|
| `run_icp_dining.py` | Main script |
| `output/sam3d_dining_icp/` | Output: GLBs + compare/overlay images |
| `docs/test_results_images/sam3d_dining_icp/` | Images for this doc |

### Key Code

- `grow_mask_v9()` — v9 mask growth (self-contained in script, same algorithm as `visualize_convex_hull_growth.py`)
- `run_icp_alignment()` — ICP in OpenCV space, result converted back to PT3D via `M @ T_cv @ M`
- `render_scene_on_background()` — vectorized vertex projection + 3x3 dilation + alpha blend onto photo
- `make_per_object_compare()` — 3-panel compare image
- `make_scene_photo_comparison()` — side-by-side original | overlay
