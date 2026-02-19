# SAM3D Dining Scene v6 — Normal-Consistency + Depth-Difference Gate

**Date:** 2026-02-18
**Run:** `output/sam3d_dining_v6/`
**Scene:** Dining (9 objects)
**Baseline:** `output/sam3d_dining_v5/` (normal-consistency only, max 60°)

---

## What Was Done

Added a **depth-difference gate** on top of the existing normal-consistency convex hull growth algorithm (v5). The goal: catch cases where hull pixels pass the normal angle test but are at genuinely different depths (i.e., background lying between object parts with similar normals).

```python
# v5 — normal-consistency only:
grown = cleaned | (hull_region & angle < threshold & valid_normal)

# v6 — normal-consistency + depth gate:
depth_neighbor  = depth[EDT_nearest_cleaned_pixel]
depth_ok        = |depth_P - depth_neighbor| < depth_thresh_m   # 5 cm
grown = cleaned | (hull_region & angle < threshold & valid_normal & depth_ok)
```

**Parameters:** `smooth_sigma=2.0` (~13×13), `max_angle=60°`, `depth_thresh=5cm`, `erosion_iters=2`, `dilation_iters=2`.

---

## Key Finding: Depth Gate Filters Background on Flat/Sparse Objects

For objects with complex silhouettes (chair legs, sofa, wooden chair), many hull pixels pass the normal angle gate but are at depths 5+ cm behind the nearest mask pixel — i.e., they are floor/background pixels, not object surface. The depth gate correctly rejects these.

The "Normal+ok+Depth-fail" column below quantifies how many pixels the depth gate removes from what normal-only would have accepted.

---

## Results

### v5 Normal-only vs v6 Normal+Depth (dining scene, depth_thresh = 5 cm)

| Object | Mask (px) | v5 Normal-only | v6 Normal+Depth | Normal+ok+Depth-fail | Normal-fail+Depth+ok | Eroded | Notes |
|---|---|---|---|---|---|---|---|
| chair_cushion | 39,822 | +6px | **+6px** | 0px | 874px | -19px | Normal gate still bottleneck for curved cushion |
| chair_legs | 262,985 | +33,530px | **+2,077px** | 29,897px | 8,334px | -98px | Depth gate filters 90% of normal-passing pixels |
| newspaper | 16,463 | +4,041px | **+321px** | 0px | 68px | -9px | Depth gate strict — hull overlaps table surface |
| placemat | 12,344 | +168px | **+127px** | 36px | 175px | -8px | Mostly unchanged |
| round_table_with_tablecloth | 84,560 | +48,088px | **+30,315px** | 17,676px | 1,373px | -37px | Cloth folds have depth discontinuities |
| sofa_with_patterned_cover | 134,334 | +30,072px | **+5,681px** | 20,591px | 4,076px | -67px | Large curved sofa — many background hull pixels |
| strainer | 14,468 | +4,823px | **+4,094px** | 769px | 80px | -48px | Small object, compact hull, minimal change |
| travel_pillow | 18,284 | +1,745px | **+545px** | 1,194px | 333px | -13px | Depth gate removes pillow-side pixels |
| wooden_chair | 36,802 | +22,795px | **+2,938px** | 15,310px | 17,613px | -23px | Chair frame gaps — background at different depth |

---

## Per-Object Visualizations

Each image is a 2×2 grid:

| Panel | Contents | Color legend |
| --- | --- | --- |
| **Top-left** | Depth map + original mask | Green (semi-transparent) = original mask pixels |
| **Top-right** | Mask + convex hull | Green = cleaned mask · Dark blue = hull interior · **Yellow = hull boundary edge** · Red = pixels lost to erosion |
| **Bottom-left** | Grown mask (final result) | Green = cleaned mask · **Orange = new growth pixels (accepted)** · Blue = normal-ok but depth-fail · Dark red = eroded pixels |
| **Bottom-right** | Angle histogram | Blue bars = growth-region angle distribution · Red dashed = adaptive threshold |

> Yellow = hull boundary, **not** growth pixels. Growth pixels = **orange** in bottom-left.

![chair_cushion_mask_growth](../output/sam3d_dining_v6/vis/chair_cushion_mask_growth.png)

**chair_cushion:** +6px, identical to v5. The curved cushion's global reference normal fails — only 6 hull pixels pass the 45.8° angle gate. 874 additional pixels pass the depth gate alone but not the normal gate, showing the normal gate is still the bottleneck here.

![chair_legs_mask_growth](../output/sam3d_dining_v6/vis/chair_legs_mask_growth.png)

**chair_legs:** +2,077px (v5: +33,530px). The depth gate removes 29,897 pixels that pass the tight 19.0° normal threshold but are 5+ cm behind the nearest chair surface pixel — these are floor/tablecloth pixels between the legs. Depth gate is highly effective here.

![newspaper_mask_growth](../output/sam3d_dining_v6/vis/newspaper_mask_growth.png)

**newspaper:** +321px (v5: +4,041px). Zero pixels pass both gates, 321 come from just inside the angle threshold with similar depth. The hull region mostly overlaps the table — depth gate aggressively rejects table-depth pixels.

![placemat_mask_growth](../output/sam3d_dining_v6/vis/placemat_mask_growth.png)

**placemat:** +127px (v5: +168px), nearly unchanged. Very flat object with small hull gap — most hull pixels are genuinely on the placemat surface.

![round_table_with_tablecloth_mask_growth](../output/sam3d_dining_v6/vis/round_table_with_tablecloth_mask_growth.png)

**round_table_with_tablecloth:** +30,315px (v5: +48,088px). Draped tablecloth has depth discontinuities at folds — 17,676px pass the normal gate but are depth-rejected. Still fills 63% of the hull gap.

![sofa_with_patterned_cover_mask_growth](../output/sam3d_dining_v6/vis/sofa_with_patterned_cover_mask_growth.png)

**sofa_with_patterned_cover:** +5,681px (v5: +30,072px). Large curved sofa — 20,591 pixels pass the 58.5° normal threshold but are at different depths (background behind sofa). Depth gate strongly filters the sofa hull interior.

![strainer_mask_growth](../output/sam3d_dining_v6/vis/strainer_mask_growth.png)

**strainer:** +4,094px (v5: +4,823px), only 15% reduction. Small compact object with tight hull — most hull pixels are genuinely on the strainer surface at consistent depth.

![travel_pillow_mask_growth](../output/sam3d_dining_v6/vis/travel_pillow_mask_growth.png)

**travel_pillow:** +545px (v5: +1,745px). U-shaped pillow — 1,194 pixels pass the 60° normal cap but are rejected by depth. Pillow sides are at different depths than the nearest mask pixels.

![wooden_chair_mask_growth](../output/sam3d_dining_v6/vis/wooden_chair_mask_growth.png)

**wooden_chair:** +2,938px (v5: +22,795px). Sparse frame mask with large hull interior. 15,310 normal-passing pixels are depth-rejected (background between slats). 17,613 depth-ok pixels fail the normal gate (back of chair vs front reference). Combined gate is very selective on sparse structures.

---

## Observations

- **Depth gate most effective for sparse/complex structures** (`chair_legs`, `wooden_chair`, `sofa`): these objects have large hull interiors containing genuine background at different depths. Normal-only was adding too many background pixels; depth gate correctly removes them.
- **Depth gate minimal impact on compact objects** (`strainer`, `placemat`): hull interior is genuinely on-surface, so depth difference to nearest mask pixel is small.
- **`chair_cushion` remains unsolved**: curved cushion's global reference normal method gives +6px regardless of depth gate — the normal gate itself is the bottleneck. Plane-distance (v4) gave +6,436px by using local nearest-neighbor reference.
- **`newspaper` regression**: v5 added +4,041px but many were off-surface (table background). v6 reduces to +321px — this is likely more correct, matching the plane-distance result (+389px).
- **`Normal-fail+Depth+ok`** column (`wooden_chair`: 17,613px) shows pixels that are depth-consistent but normal-inconsistent. These could be legitimate object pixels (e.g., different surface facets) that both methods miss.

---

## Algorithm Comparison: Five Methods on Dining Scene

| Object | v5 Angle | Plane-Dist | RANSAC v1 | RANSAC v2 | v6 Normal+Depth |
|---|---|---|---|---|---|
| chair_cushion | +6px | +6,436px | +1,625px | +4,413px | **+6px** |
| chair_legs | +33,530px | +35,300px | +4,819px | +8,960px | **+2,077px** |
| newspaper | +4,041px | +389px | +365px | +389px | **+321px** |
| placemat | +168px | +1,187px | +220px | +375px | **+127px** |
| round_table | +48,088px | +45,143px | +12,044px | +22,111px | **+30,315px** |
| sofa | +30,072px | +36,710px | +4,101px | +11,287px | **+5,681px** |
| strainer | +4,823px | +4,922px | +2,735px | +3,544px | **+4,094px** |
| travel_pillow | +1,745px | +2,509px | +747px | +1,887px | **+545px** |
| wooden_chair | +22,795px | +23,945px | +4,325px | +14,955px | **+2,938px** |

---

## v7: Relaxed Depth Threshold (10 cm)

**Run:** `output/sam3d_dining_v7/`
**Change:** `depth_thresh` relaxed from **5 cm → 10 cm**. All other parameters identical to v6.

### v6 (5cm) vs v7 (10cm) comparison

| Object | v6 (5cm) | v7 (10cm) | Δ | Notes |
| --- | --- | --- | --- | --- |
| chair_cushion | +6px | **+6px** | 0 | Normal gate bottleneck — depth irrelevant |
| chair_legs | +2,077px | **+2,542px** | +465px | Slight relaxation |
| newspaper | +321px | **+321px** | 0 | Normal gate bottleneck |
| placemat | +127px | **+127px** | 0 | Normal gate bottleneck |
| round_table_with_tablecloth | +30,315px | **+40,734px** | +10,419px | Largest gain — cloth folds within 10cm |
| sofa_with_patterned_cover | +5,681px | **+9,153px** | +3,472px | Sofa depth variation fits within 10cm |
| strainer | +4,094px | **+4,756px** | +662px | Small gain |
| travel_pillow | +545px | **+1,048px** | +503px | Pillow sides now within threshold |
| wooden_chair | +2,938px | **+4,331px** | +1,393px | Some chair-frame gaps recovered |

Relaxing to 10cm helps `round_table` (+10K px) and `sofa` (+3.5K px) most. Objects where the normal gate is the bottleneck (`chair_cushion`, `newspaper`, `placemat`) are unchanged.

### v7 Per-Object Visualizations

Images from `output/sam3d_dining_v7/vis/` — panel layout and color legend same as v6 above (in-panel legend visible in each image).

![chair_cushion_mask_growth](../output/sam3d_dining_v7/vis/chair_cushion_mask_growth.png)

![chair_legs_mask_growth](../output/sam3d_dining_v7/vis/chair_legs_mask_growth.png)

![newspaper_mask_growth](../output/sam3d_dining_v7/vis/newspaper_mask_growth.png)

![placemat_mask_growth](../output/sam3d_dining_v7/vis/placemat_mask_growth.png)

![round_table_with_tablecloth_mask_growth](../output/sam3d_dining_v7/vis/round_table_with_tablecloth_mask_growth.png)

![sofa_with_patterned_cover_mask_growth](../output/sam3d_dining_v7/vis/sofa_with_patterned_cover_mask_growth.png)

![strainer_mask_growth](../output/sam3d_dining_v7/vis/strainer_mask_growth.png)

![travel_pillow_mask_growth](../output/sam3d_dining_v7/vis/travel_pillow_mask_growth.png)

![wooden_chair_mask_growth](../output/sam3d_dining_v7/vis/wooden_chair_mask_growth.png)

---

## What Was Not Isolated

- TRELLIS reconstruction quality impact not measured — visualization only.
- `depth_thresh` between 10cm and plane-distance (3cm fixed) not swept further.
- The depth gate could be combined with the plane-distance (local reference) approach instead of the global-reference normal approach — this might fix the `chair_cushion` failure.
- `Normal-fail+Depth+ok` pixels are currently discarded; accepting them (depth-only mode as fallback) could recover growth for objects where the global normal reference fails.

---

## Files

| File | Description |
| --- | --- |
| `visualize_convex_hull_growth.py` | v7 script (current config: `depth_thresh=10cm`, normal+depth, in-panel legend) |
| `output/sam3d_dining_v7/vis/` | v7 mask growth images (depth_thresh=10cm) |
| `output/sam3d_dining_v6/vis/` | v6 mask growth images (depth_thresh=5cm) |
| `output/sam3d_dining_v5/vis/` | v5 normal-only baseline |
| `output/sam3d_dining_plane_dist/vis/` | plane-distance baseline |

### Key Code Changes

- `visualize_convex_hull_growth.py` — `_grow_mask_normal_depth()` replaces previous growth functions; adds `distance_transform_edt` for nearest-neighbor depth lookup; `depth_ok = |depth_P - depth_neighbor| < depth_thresh_m`; visualization panel 3 shows blue = normal-ok but depth-fail pixels
- v7: `DEPTH_THRESH_M` changed from `0.05` → `0.10`; `matplotlib.patches.Patch` legends added inside each image panel
