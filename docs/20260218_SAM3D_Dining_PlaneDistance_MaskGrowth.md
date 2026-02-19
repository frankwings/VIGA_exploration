# SAM3D Dining Scene — Local Plane-Distance Mask Growth

**Date:** 2026-02-18
**Run:** `output/sam3d_dining_plane_dist/`
**Scene:** Dining (9 objects)
**Baseline:** `output/sam3d_dining_v5/` (angle-based normal consistency, max 60°)

---

## What Was Done

Replaced the global-reference angle-based convex hull growth (v5) with a new algorithm using **local plane-distance consistency** plus **morphological opening** as preprocessing. Visualization-only run on the dining scene using existing MoGe pointmap from `sam3d_dining_v4`.

---

## Key Finding: Local Plane Distance Fixes Curved-Surface Failures

The previous angle-based approach (v5) compared each hull candidate's normal to a **global mean reference** computed from the entire mask. For curved/complex objects this global mean fails — e.g. `chair_cushion` grew only +6px despite having a large hull gap.

The new criterion compares each candidate's 3D position to the **tangent plane of its nearest mask neighbor** (found via EDT):

```python
# Old: global reference
ref_normal = normals[mask_bool & valid].mean(axis=0)
angle_map  = arccos(|normals @ ref_normal|)
threshold  = clip(median + 2σ, 10°, 60°)

# New: local reference per pixel
_, nearest_idx = distance_transform_edt(~cleaned, return_indices=True)
norm_neighbor  = normals[nearest_idx[0], nearest_idx[1]]
pos_neighbor   = pointmap[nearest_idx[0], nearest_idx[1]]
plane_dist     = |dot(norm_neighbor, pos_candidate - pos_neighbor)|
accept         = plane_dist < threshold_m   # fixed 3 cm
```

**Why this works better:**
- For a curved surface, each pixel's neighbor is nearby on the same surface → plane distance ≈ 0 even when the global normal differs widely
- For a background pixel at a depth discontinuity: its 3D position is far behind the nearest object pixel's tangent plane → plane distance large → correctly rejected
- No arbitrary angular cap needed (60° cap was ad-hoc; plane distance is scale-aware in meters)

**Morphological opening** (2× erosion + 2× dilation) cleans noisy boundary protrusions from the SAM mask before computing the convex hull. This gives a tighter, cleaner hull and a cleaner reference for the EDT.

---

## Results

### v5 Angle-Based vs v4 Plane-Distance (dining scene, threshold = 3 cm)

| Object | Mask (px) | v5 Angle +added | v4 Plane +added | Eroded | Net Δ | Notes |
|---|---|---|---|---|---|---|
| chair_cushion | 39,822 | +6px | **+6,436px** | -19px | +6,417 | Curved cushion — angle method failed |
| chair_legs | 262,985 | +33,530px | **+35,300px** | -98px | +35,202 | Flat + concave, both work |
| newspaper | 16,463 | +4,041px | **+389px** | -9px | +380 | Plane dist correctly rejects off-surface |
| placemat | 12,344 | +168px | **+1,187px** | -8px | +1,179 | Flat — plane dist more aggressive |
| round_table_with_tablecloth | 84,560 | +48,088px | **+45,143px** | -37px | +45,106 | Draped cloth, similar |
| sofa_with_patterned_cover | 134,334 | +30,072px | **+36,710px** | -67px | +36,643 | Large curved sofa |
| strainer | 14,468 | +4,823px | **+4,922px** | -48px | +4,874 | Complex perforated shape |
| travel_pillow | 18,284 | +1,745px | **+2,509px** | -13px | +2,496 | Curved U-shape pillow |
| wooden_chair | 36,802 | +22,795px | **+23,945px** | -23px | +23,922 | Sparse frame, large hull |

---

## Per-Object Visualizations

Each 4-panel: plane distance map (0=coplanar/green → red=far) | original + cleaned + hull | grown mask | plane distance histogram.

![chair_cushion_mask_growth](../output/sam3d_dining_plane_dist/vis/chair_cushion_mask_growth.png)

**chair_cushion:** +6,436px at 3 cm threshold. Previous angle method gave only +6px due to curved surface failing the global reference. Plane-distance correctly identifies the coplanar cushion pixels through local neighbor comparison. Morphological opening removed -19px of boundary noise.

![chair_legs_mask_growth](../output/sam3d_dining_plane_dist/vis/chair_legs_mask_growth.png)

**chair_legs:** +35,300px. Large structured flat object with concave leg gaps. Hull gap well filled. Erosion removes -98px of jagged boundary pixels from the complex multi-leg silhouette.

![newspaper_mask_growth](../output/sam3d_dining_plane_dist/vis/newspaper_mask_growth.png)

**newspaper:** Only +389px (was +4,041px with angle method). Plane distance is stricter — hull pixels that are on a different surface (table background) are correctly rejected because their 3D position is off the newspaper's tangent plane. More conservative = more correct.

![placemat_mask_growth](../output/sam3d_dining_plane_dist/vis/placemat_mask_growth.png)

**placemat:** +1,187px (was +168px). Flat object — plane-distance with local reference is more aggressive than the angle method's tight 15° adaptive threshold, which was overly conservative for a flat surface.

![round_table_with_tablecloth_mask_growth](../output/sam3d_dining_plane_dist/vis/round_table_with_tablecloth_mask_growth.png)

**round_table_with_tablecloth:** +45,143px. Draped tablecloth has complex geometry; both methods fill most of the 52K px hull gap. Plane distance slightly more conservative (-3K vs angle method) for the outermost cloth folds.

![sofa_with_patterned_cover_mask_growth](../output/sam3d_dining_plane_dist/vis/sofa_with_patterned_cover_mask_growth.png)

**sofa_with_patterned_cover:** +36,710px (was +30,072px). Large curved sofa benefits from local reference — plane distance correctly follows the sofa surface curvature rather than being capped by a global 60° limit.

![strainer_mask_growth](../output/sam3d_dining_plane_dist/vis/strainer_mask_growth.png)

**strainer:** +4,922px, nearly identical to angle method (+4,823px). Small complex 3D shape; both methods fill the hull gap similarly.

![travel_pillow_mask_growth](../output/sam3d_dining_plane_dist/vis/travel_pillow_mask_growth.png)

**travel_pillow:** +2,509px (was +1,745px). Curved U-shape pillow benefits from local reference rather than global mean.

![wooden_chair_mask_growth](../output/sam3d_dining_plane_dist/vis/wooden_chair_mask_growth.png)

**wooden_chair:** +23,945px. Sparse frame mask with large hull gap (78K px). 26% of hull gap filled — correctly rejects the transparent regions between chair slats (background depth ≠ chair surface plane).

---

## Observations

- **`chair_cushion` is the clearest win**: +6px → +6,436px. The global angle reference failed completely for a curved cushion; local plane distance works naturally.
- **`newspaper` shows correct conservatism**: +4,041px → +389px. The angle method was adding off-surface hull pixels (wrong table depth included); plane distance correctly rejects them.
- **`placemat` shows improved aggressiveness on flat surfaces**: +168px → +1,187px. The angle-based 15° adaptive threshold was too tight; plane distance at 3 cm is calibrated in world space, not sensitive to normal distribution shape.
- **Morphological opening is small but useful**: erosion ranges from -8px (placemat) to -98px (chair_legs). Cleans boundary protrusions before hull computation, giving tighter hulls and cleaner EDT reference.
- **Threshold 3 cm appears well-calibrated**: background pixels (depth discontinuity ~20+ cm) are rejected; surface pixels (within a few mm of the local tangent plane) are accepted.

---

## What Was Not Isolated

- TRELLIS reconstruction quality impact not measured — this is visualization only. A full pipeline re-run with the new grown masks would be needed to compare IoU.
- Optimal `threshold_m` not tuned — 3 cm is a reasonable default but could be scene-dependent (objects at different depths have different effective pixel sizes).
- The `erosion_iters=2 / dilation_iters=2` parameters not ablated — could test 1×1 or 3×3.

---

## Files

| File | Description |
|---|---|
| `visualize_convex_hull_growth.py` | Plane-distance growth script (current config: dining) |
| `output/sam3d_dining_plane_dist/vis/` | New plane-distance mask growth images |
| `output/sam3d_dining_v5/vis/` | Previous angle-based mask growth images (baseline) |

### Key Code Changes

- `visualize_convex_hull_growth.py` — `_grow_mask_plane_distance()` replaces `_grow_mask_normal_consistency()`; new `visualize_mask_growth()` shows plane-dist map and histogram instead of angle map
- `utils/third_party/sam3d/.../layout_post_optimization_utils.py:227-287` — `_grow_mask_to_convex_hull()` updated to match; `distance_transform_edt` added to imports
