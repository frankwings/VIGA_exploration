# SAM3D Normal-Consistency Mask Growth — Results

**Date:** 2026-02-18
**Author:** kingy + Claude (Sonnet 4.6)
**Hardware:** RTX 5080 16GB | 32GB DDR5-6000 | Ryzen 9 9900X
**Run:** `output/sam3d_convex_hull_v3/`
**Scene:** Greentea (5 objects)
**Comparison baseline:** `output/sam3d_convex_hull_v2/` (Sobel-based growth)

---

## What Changed from v2

v2 used **Sobel depth gradient** as the growth stop criterion — allowed growth where `sobel_mag < median + k*std` inside the mask. This is purely 2D and does not reason about 3D surface continuity.

v3 replaces Sobel with **local surface normal consistency**:

1. Compute per-pixel surface normals from the MoGe 3D pointmap via central differences (`dx × dy`, Gaussian-smoothed at σ=1.5)
2. Reference normal = mean of valid normals inside the original mask
3. Angle deviation map = arccos(|normals · ref|) per pixel
4. Adaptive threshold = median + 2σ of angles inside mask, clamped to [10°, 60°]
5. All hull pixels with angle < threshold AND valid normal are added directly (no connectivity constraint)

**Key fix over first iteration**: Iterative connected dilation was blocked by high-deviation normals at depth-discontinuity edges (mask border ring). Fix: direct assignment — all qualifying pixels join at once.

**max_threshold_deg = 60°** handles curved surfaces (bottles: shoulder/neck normals deviate 40–60° from front-face mean) while still stopping at true surface boundaries (table/background typically ≥ 70° from object normals).

---

## Mask Growth Results: v2 (Sobel) vs v3 (Normal-Consistency)

| Object | v2 Sobel | v3 Normal (60°) | Notes |
|---|---|---|---|
| ito_en_bottle | +7,585px (0.392) | **+4,772px (60.0°)** | Curved bottle — threshold hits cap |
| green_tea_bottle | +5,903px (0.012) | **+209px (18.3°)** | Better: rejects spurious shadow pixels |
| alienware_keyboard | +1,154px (0.063) | **+2,129px (48.0°)** | Flat plate — more aggressive |
| headphones | +74px (0.064) | **+71px (60.0°)** | Nearly convex, same result |
| envelope | +1,059px (0.498) | **+727px (29.0°)** | Flat rect — tight adaptive threshold |

*Threshold column: Sobel = gradient magnitude; Normal = angle in degrees.*

### Key observations

- **green_tea_bottle**: Normal method adds only +209px vs Sobel's +5,903px. This is **correct** behavior — Sobel was adding more of the flat shadow/table region (uniform depth = low Sobel = "safe"), which further contaminated the TRELLIS input. Normal method correctly rejects these pixels (hull region has different surface normals from the bottle).
- **alienware_keyboard**: Normal method is more aggressive (+2,129 vs +1,154) — direct assignment bridges over the depth-edge ring that blocked connected dilation.
- **ito_en_bottle**: Normal method (+4,772) is less than Sobel (+7,585). Curved bottle surface spans 0–90° normal deviation from the mean; 60° cap blocks extreme side pixels. Sobel at depth-edge boundaries doesn't have this limitation.

---

## Visualization

### Per-Object — 4-Panel Format

Each panel: normal deviation angle map | original mask + convex hull | grown mask | angle histogram

![ito_en_bottle_mask_growth](../output/sam3d_convex_hull_v3/vis/ito_en_bottle_mask_growth.png)

**ito_en_bottle:** +4,772px at shoulder/neck arc (threshold=60.0°). Normal angle map shows bottle surface (green/yellow) vs background (red). Growth stops at real surface boundary.

![envelope_mask_growth](../output/sam3d_convex_hull_v3/vis/envelope_mask_growth.png)

**envelope:** +727px filling corner gaps (threshold=29.0°). Flat surface → tight adaptive threshold — only true envelope surface pixels accepted.

![headphones_mask_growth](../output/sam3d_convex_hull_v3/vis/headphones_mask_growth.png)

**headphones:** Only +71px — hull gap is just 61px (mask already nearly convex). Threshold hits 60° cap.

![alienware_keyboard_mask_growth](../output/sam3d_convex_hull_v3/vis/alienware_keyboard_mask_growth.png)

**alienware_keyboard:** +2,129px (threshold=48.0°). Direct assignment captures concavities that iterative dilation couldn't reach.

![green_tea_bottle_mask_growth](../output/sam3d_convex_hull_v3/vis/green_tea_bottle_mask_growth.png)

**green_tea_bottle:** Only +209px (threshold=18.3°). Hull region normals deviate significantly from the flat mask interior — correctly rejected. The degenerate mask (shadow/table contamination) is exposed here.

---

## Algorithm Comparison

| Property | v2 Sobel | v3 Normal-Consistency |
|---|---|---|
| Stop criterion | 2D depth gradient magnitude | 3D surface normal angle deviation |
| Threshold computation | median + 2σ of Sobel in mask | median + 2σ of angle-dev in mask |
| Threshold cap | None (can be very low) | [10°, 60°] |
| Growth method | Iterative connected dilation | Direct assignment (no connectivity) |
| Handles curved surfaces | Yes (no angular bias) | Partially (60° cap limits sides) |
| Handles shadow contamination | No (flat depth = permissive) | Yes (flat normal = tight threshold) |
| 3D surface awareness | No | Yes |

---

## Limitation: Curved Surfaces

The global mean reference normal breaks down for curved objects (bottles, cylinders). A cylinder's normals span 0°–90° from the front-face mean; extreme side pixels are rejected even though they're part of the same object.

**Potential fix (not yet implemented):** Per-pixel local reference — compare each candidate pixel's normal to its nearest original-mask pixel's normal, using `scipy.ndimage.distance_transform_edt` to find nearest indices. This would handle curvature without a global mean.

---

## Files

| File | Description |
|---|---|
| `visualize_convex_hull_growth.py` | Normal-consistency visualization (v3) |
| `visualize_convex_hull_growth_sobel.py` | Sobel-based visualization (v2, archived) |
| `output/sam3d_convex_hull_v3/vis/` | Normal-consistency mask growth images |
| `output/sam3d_convex_hull_v2/vis/` | Sobel-based mask growth images (original) |
| `utils/third_party/sam3d/.../layout_post_optimization_utils.py` | Normal-consistency in SAM3D pipeline |
