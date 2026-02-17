# ICP Depth-Scale Correction Design

**Date:** 2026-02-16
**File:** `reoptimize_depth.py`

## Problem

Convex hull mask growth + ICP improves depth alignment for 7/9 objects but makes 2 worse:
- **wooden_chair**: Sparse mask leaks into table/floor via convex hull growth. ICP pulls vertices to wrong-object depth. Scale ratio goes from 1.05 to 1.14.
- **round_table**: Mask holes filled by convex hull contain depth from objects ON the table, not the table surface.

Root causes:
1. No exclusion of other objects' masks from growth region
2. ICP is rigid (no scale) -- cannot fix depth/scale mismatch
3. No rejection when ICP increases depth error

## Key Insight

**Uniform scaling from the camera origin preserves the 2D projection exactly.**

For vertex (X, Y, Z) projecting to pixel (fx*X/Z + cx, fy*Y/Z + cy), scaling by s gives (sX, sY, sZ) which projects to the same pixel because s cancels in X/Z and Y/Z. Only depth changes: Z -> sZ.

This means depth can be corrected without affecting the silhouette at all.

## Design

Three changes to `reoptimize_depth.py`, applied in order:

### 1. Mask Exclusion

Load all object masks upfront. When growing object A's mask toward its convex hull, subtract the union of all other objects' masks from the allowed growth region. Prevents leaking into neighboring objects' depth.

### 2. Post-ICP Depth-Scale Correction

After ICP (rigid alignment):
1. Project ICP-aligned vertices to pixels
2. For vertices inside the **original** mask, collect (z_vertex, z_moge)
3. Compute `s = median(z_moge / z_vertex)` -- robust to outliers
4. Apply `v' = s * v` to all vertices (scale from camera origin)

Silhouette is mathematically unchanged. Depth is corrected.

### 3. Rejection Gate

After ICP + scale correction, compare final depth error to the original (pre-ICP) error. If error increased, reject and keep the original vertices. Safety net.

## Pipeline Order

```
Load vertices
  -> Grow mask (with other-object exclusion)
  -> ICP rigid alignment (rotation + translation)
  -> Depth-scale correction (uniform scale from origin)
  -> Rejection gate (compare to original, keep best)
  -> Save
```
