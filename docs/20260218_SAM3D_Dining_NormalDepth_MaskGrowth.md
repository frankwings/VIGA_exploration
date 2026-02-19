# SAM3D Dining Scene — Convex Hull Mask Growth: Depth Gate Variants (v6–v8)

**Date:** 2026-02-18
**Scene:** Dining (9 objects)
**Baseline:** `output/sam3d_dining_v5/` (normal-consistency only, max 60°)

---

## Algorithm Summary: Depth Gate Variants

All versions share the same **normal-consistency** gate (global reference normal, adaptive threshold `clip(median+2σ, 10°, 60°)`, morphological opening 2×erosion+2×dilation). They differ only in the **depth gate**:

| Version | Depth Gate Method | Depth Reference | Threshold |
|---|---|---|---|
| **v5** | None | — | — |
| **v6** | EDT nearest mask pixel depth diff | Single nearest Q (EDT) | 5 cm fixed |
| **v7** | EDT nearest mask pixel depth diff | Single nearest Q (EDT) | 10 cm fixed |
| **v8** | KDTree K=60 nearest mask pixels, depth range [dmin, dmax] | All K neighbors | Adaptive (object depth span) |

```python
# v5 — normal only:
grown = cleaned | (hull_region & angle < threshold & valid_normal)

# v6/v7 — normal + EDT fixed threshold:
depth_neighbor  = depth[EDT_nearest_cleaned_pixel(P)]
depth_ok        = |depth[P] - depth_neighbor| < thresh   # 5cm (v6) / 10cm (v7)
grown = cleaned | (hull_region & angle < threshold & valid_normal & depth_ok)

# v8 — normal + all-K local depth range:
Q_depths        = depth[KDTree_K_nearest_mask_pixels(P)]  # K=60
depth_ok        = min(Q_depths) <= depth[P] <= max(Q_depths)
grown = cleaned | (hull_region & angle < threshold & valid_normal & depth_ok)
```

**Key behavioral difference:**

- v6/v7: depth of P vs its single spatially-nearest mask pixel. Fails for large gaps (nearest pixel is far from where P "should" be on the object).
- v8: depth of P vs the range spanned by its 60 nearest mask pixels. More conservative than v7 for objects where the K nearest pixels are all concentrated near one edge (narrow depth range); looser for compact objects.

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

![chair_cushion_mask_growth](test_results_images/sam3d_dining_v6/chair_cushion_mask_growth.png)

**chair_cushion:** +6px, identical to v5. The curved cushion's global reference normal fails — only 6 hull pixels pass the 45.8° angle gate. 874 additional pixels pass the depth gate alone but not the normal gate, showing the normal gate is still the bottleneck here.

![chair_legs_mask_growth](test_results_images/sam3d_dining_v6/chair_legs_mask_growth.png)

**chair_legs:** +2,077px (v5: +33,530px). The depth gate removes 29,897 pixels that pass the tight 19.0° normal threshold but are 5+ cm behind the nearest chair surface pixel — these are floor/tablecloth pixels between the legs. Depth gate is highly effective here.

![newspaper_mask_growth](test_results_images/sam3d_dining_v6/newspaper_mask_growth.png)

**newspaper:** +321px (v5: +4,041px). Zero pixels pass both gates, 321 come from just inside the angle threshold with similar depth. The hull region mostly overlaps the table — depth gate aggressively rejects table-depth pixels.

![placemat_mask_growth](test_results_images/sam3d_dining_v6/placemat_mask_growth.png)

**placemat:** +127px (v5: +168px), nearly unchanged. Very flat object with small hull gap — most hull pixels are genuinely on the placemat surface.

![round_table_with_tablecloth_mask_growth](test_results_images/sam3d_dining_v6/round_table_with_tablecloth_mask_growth.png)

**round_table_with_tablecloth:** +30,315px (v5: +48,088px). Draped tablecloth has depth discontinuities at folds — 17,676px pass the normal gate but are depth-rejected. Still fills 63% of the hull gap.

![sofa_with_patterned_cover_mask_growth](test_results_images/sam3d_dining_v6/sofa_with_patterned_cover_mask_growth.png)

**sofa_with_patterned_cover:** +5,681px (v5: +30,072px). Large curved sofa — 20,591 pixels pass the 58.5° normal threshold but are at different depths (background behind sofa). Depth gate strongly filters the sofa hull interior.

![strainer_mask_growth](test_results_images/sam3d_dining_v6/strainer_mask_growth.png)

**strainer:** +4,094px (v5: +4,823px), only 15% reduction. Small compact object with tight hull — most hull pixels are genuinely on the strainer surface at consistent depth.

![travel_pillow_mask_growth](test_results_images/sam3d_dining_v6/travel_pillow_mask_growth.png)

**travel_pillow:** +545px (v5: +1,745px). U-shaped pillow — 1,194 pixels pass the 60° normal cap but are rejected by depth. Pillow sides are at different depths than the nearest mask pixels.

![wooden_chair_mask_growth](test_results_images/sam3d_dining_v6/wooden_chair_mask_growth.png)

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

![chair_cushion_mask_growth](test_results_images/sam3d_dining_v7/chair_cushion_mask_growth.png)

![chair_legs_mask_growth](test_results_images/sam3d_dining_v7/chair_legs_mask_growth.png)

![newspaper_mask_growth](test_results_images/sam3d_dining_v7/newspaper_mask_growth.png)

![placemat_mask_growth](test_results_images/sam3d_dining_v7/placemat_mask_growth.png)

![round_table_with_tablecloth_mask_growth](test_results_images/sam3d_dining_v7/round_table_with_tablecloth_mask_growth.png)

![sofa_with_patterned_cover_mask_growth](test_results_images/sam3d_dining_v7/sofa_with_patterned_cover_mask_growth.png)

![strainer_mask_growth](test_results_images/sam3d_dining_v7/strainer_mask_growth.png)

![travel_pillow_mask_growth](test_results_images/sam3d_dining_v7/travel_pillow_mask_growth.png)

![wooden_chair_mask_growth](test_results_images/sam3d_dining_v7/wooden_chair_mask_growth.png)

---

## v8: All-K Local Depth Range Gate

**Run:** `output/sam3d_dining_v8/`
**Change:** Replace EDT single-neighbor fixed threshold with KDTree K=60 all-neighbor depth range `[dmin, dmax]`. Accept hull pixel P if `depth[P] ∈ [min(K depths), max(K depths)]`. No fixed threshold — depth gate adapts to local object depth span.

### v7 (10cm) vs v8 (all-K range) comparison

| Object | v7 (10cm) | v8 all-K | Δ | Notes |
| --- | --- | --- | --- | --- |
| chair_cushion | +6px | **+6px** | 0 | Normal gate bottleneck — unchanged |
| chair_legs | +2,542px | **+1,417px** | −1,125px | K=60 nearest all cluster near one edge → narrow range |
| newspaper | +321px | **+58px** | −263px | Very strict — K neighbors all at table depth |
| placemat | +127px | **+35px** | −92px | Flat object, K neighbors near one edge |
| round_table_with_tablecloth | +40,734px | **+16,641px** | −24,093px | Large reduction — cloth fold depth range narrower than 10cm |
| sofa_with_patterned_cover | +9,153px | **+2,617px** | −6,536px | Most conservative result yet |
| strainer | +4,756px | **+2,127px** | −2,629px | Compact but K range still narrower than 10cm |
| travel_pillow | +1,048px | **+396px** | −652px | U-shape — K nearest from one side only |
| wooden_chair | +4,331px | **+1,730px** | −2,601px | Sparse frame — K nearest from closest slat |

**Key finding:** v8 is **more conservative than even v6 (5cm)** for most objects. The reason: KDTree returns the K=60 spatially-nearest mask pixels, which for sparse/large objects are all clustered near the closest mask edge to P. This edge spans a narrow depth range (e.g., one chair slat), so the [dmin, dmax] gate is tight. The EDT approach (v6/v7) compares against the single nearest pixel but uses a generous fixed threshold; v8 uses many neighbors but they're all from one local area.

For the sector approach (tested between v7 and v8): picking nearest Q per 8 direction sectors forced directional coverage across the object, giving wider [dmin, dmax]. All-K without sectors collapses back to near-edge sampling.

### v8 Per-Object Visualizations

Images from `output/sam3d_dining_v8/vis/`.

![chair_cushion_mask_growth](test_results_images/sam3d_dining_v8/chair_cushion_mask_growth.png)

![chair_legs_mask_growth](test_results_images/sam3d_dining_v8/chair_legs_mask_growth.png)

![newspaper_mask_growth](test_results_images/sam3d_dining_v8/newspaper_mask_growth.png)

![placemat_mask_growth](test_results_images/sam3d_dining_v8/placemat_mask_growth.png)

![round_table_with_tablecloth_mask_growth](test_results_images/sam3d_dining_v8/round_table_with_tablecloth_mask_growth.png)

![sofa_with_patterned_cover_mask_growth](test_results_images/sam3d_dining_v8/sofa_with_patterned_cover_mask_growth.png)

![strainer_mask_growth](test_results_images/sam3d_dining_v8/strainer_mask_growth.png)

![travel_pillow_mask_growth](test_results_images/sam3d_dining_v8/travel_pillow_mask_growth.png)

![wooden_chair_mask_growth](test_results_images/sam3d_dining_v8/wooden_chair_mask_growth.png)

---

## Full Algorithm Comparison (all methods, dining scene)

| Object | v5 Normal-only | v6 (5cm EDT) | v7 (10cm EDT) | v8 (all-K range) | Plane-Dist | RANSAC v2 |
| --- | --- | --- | --- | --- | --- | --- |
| chair_cushion | +6px | +6px | +6px | **+6px** | +6,436px | +4,413px |
| chair_legs | +33,530px | +2,077px | +2,542px | **+1,417px** | +35,300px | +8,960px |
| newspaper | +4,041px | +321px | +321px | **+58px** | +389px | +389px |
| placemat | +168px | +127px | +127px | **+35px** | +1,187px | +375px |
| round_table | +48,088px | +30,315px | +40,734px | **+16,641px** | +45,143px | +22,111px |
| sofa | +30,072px | +5,681px | +9,153px | **+2,617px** | +36,710px | +11,287px |
| strainer | +4,823px | +4,094px | +4,756px | **+2,127px** | +4,922px | +3,544px |
| travel_pillow | +1,745px | +545px | +1,048px | **+396px** | +2,509px | +1,887px |
| wooden_chair | +22,795px | +2,938px | +4,331px | **+1,730px** | +23,945px | +14,955px |

---

## What Was Not Isolated

- TRELLIS reconstruction quality impact not measured — visualization only.
- v8 is more conservative than expected. A **sector-constrained all-K** approach (pick nearest Q per 8 sectors, then union all their depths) could give a wider, more accurate object-spanning depth range.
- The depth gate could be combined with the plane-distance (local reference) approach — this might fix `chair_cushion`.
- `Normal-fail+Depth+ok` pixels are currently discarded; depth-only fallback could recover growth where global normal reference fails.

---

## Files

| File | Description |
| --- | --- |
| `visualize_convex_hull_growth.py` | Current script (v8: all-K local depth range, K=60) |
| `output/sam3d_dining_v8/vis/` | v8 mask growth images |
| `output/sam3d_dining_v7/vis/` | v7 mask growth images (10cm EDT) |
| `output/sam3d_dining_v6/vis/` | v6 mask growth images (5cm EDT) |
| `output/sam3d_dining_v5/vis/` | v5 normal-only baseline |
| `output/sam3d_dining_plane_dist/vis/` | plane-distance baseline |

### Key Code Changes

- **v6**: `_grow_mask_normal_depth()` — `distance_transform_edt` nearest-neighbor depth lookup; `depth_ok = |depth_P - depth_neighbor| < 0.05`
- **v7**: `DEPTH_THRESH_M` relaxed `0.05 → 0.10`; `matplotlib.patches.Patch` in-panel legends added
- **v8**: EDT replaced with `cKDTree.query(gr_xy, k=60)`; depth gate = `q_dep_all.min() <= depth_P <= q_dep_all.max()` over all 60 neighbors; sector loop removed
