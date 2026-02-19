# SAM3D Pipeline — Step 5 Corrections

**Date:** 2026-02-19
**Corrects:** `docs/20260216_SAM3D_Pipeline.md` — Step 5 (Layout Post-Optimization)
**Source:** Empirical investigation of wooden_chair run logs + code trace through
`inference_utils.py` and `layout_post_optimization_utils.py`

---

## Correction 1: Stage 5a uses the grown mask, not the raw SAM mask

The 2026-02-16 doc wrote:

```python
# WRONG — what the doc said:
target_object_points = Point_Map[mask[0, 0].bool()]
```

The actual code (`layout_post_optimization_utils.py:422-431`):

```python
# CORRECT:
grown_np = grow_mask_v9(mask_np, pointmap_np, depth_np)   # v9 normal + 8-dir ray gate
grown_mask = torch.from_numpy(grown_np).to(device)
target_object_points = Point_Map[grown_mask]              # grown mask, not raw SAM mask
```

The mask growth (v9) runs **before** target point extraction. `target_object_points` comes
from the convex-hull-grown region, not the original SAM mask.

---

## Correction 2: flag_notgt triggers from the depth filter, not empty grown mask

The 2026-02-16 doc implied `flag_notgt` fires when the grown mask is empty. The actual
trigger is the **depth quantile filter** that runs after extraction
(`layout_post_optimization_utils.py:441-450`):

```python
if target_object_points.shape[0] > 0:
    depth_quantile = torch.quantile(target_object_points[:, 2], 0.9)
    target_object_points = target_object_points[
        target_object_points[:, 2] <= depth_quantile   # NaN <= NaN → False → empty!
    ]
flag_notgt = (target_object_points.shape[0] == 0)
```

If **all Z values are NaN**, `NaN <= NaN` is `False` for every point → the filter empties
the tensor → `flag_notgt = True` → entire post-opt aborted, IoU = -1.

This is why the diagnostic line `Z range=[nan, nan]` is meaningful: it signals that
`.min()` found NaN somewhere in the pointmap. But it does **not** mean all points are NaN
— it means at least one pixel is NaN (PyTorch `.min()` propagates NaN). If the grown mask
region has valid Z values, `flag_notgt` stays False even when the global `Z range=[nan, nan]`.

---

## Correction 3: Adam always runs, regardless of ICP outcome

The 2026-02-16 doc's decision tree was correct but did not state this explicitly. From
today's code trace (`inference_utils.py:186-191`):

```python
# Step 3: Render-and-Compare — NO condition on Flag_ICP
if not Enable_rendering_optimization:
    Flag_optim = False
    tfm = tfm_ori.compose(tfm1).compose(tfm2)
else:
    quat, translation, scale, R = run_render_compare(   # ALWAYS runs
        mesh, center, renderer, mask, device
    )
```

Adam runs on **whatever mesh came out of Step 1 or 2**:
- ICP accepted → Adam starts from the ICP-refined mesh
- ICP rejected → Adam starts from the Step 1 manually-aligned mesh

---

## Corrected Step 5 Sub-Pipeline

```
TRELLIS pose (Q, T, S)
    |
    v
[Occlusion check]  ──fail──>  return raw TRELLIS pose (no IoU computed)
    |
    v
[Stage 5a: Manual Alignment]
    grow_mask_v9(SAM mask, pointmap)  →  grown_mask
    target_points = pointmap[grown_mask]
    filter target_points: z <= 90th-percentile z
    │
    ├── target_points empty (all NaN z)?  ──yes──>  flag_notgt=True
    │                                               return raw TRELLIS pose, IoU = -1
    │
    └── scale mesh height to match target height
        translate mesh centroid to match target centroid
        render → ori_iou = 0.0938
    |
    v
[Stage 5b: ICP]
    Open3D point-to-point ICP (threshold=0.05m)
    render ICP result → ori_iou_shapeICP
    │
    ├── ori_iou_shapeICP > ori_iou?  ──yes──>  mesh = ICP result
    │
    └── no  ──>  mesh stays at Stage 5a result (tfm2 = identity)
    |
    v
[Stage 5c: Adam — ALWAYS RUNS]
    30-step gradient descent (5 × translation+scale, 25 × all)
    loss = 200 × silhouette_MSE + regularization
    render → optimized_iou = 0.1684
    │
    ├── optimized_iou > 0.5 AND > ori_iou?  ──yes──>  accept (5a + 5b + 5c)
    │
    └── no  ──>  FULL REVERT: tfm = tfm_ori  (raw TRELLIS pose, same as occlusion fail)
```

---

## What Z range=[nan, nan] Actually Means

Two distinct situations produce identical log output:

| Situation | Z range log | What it means | flag_notgt? |
|---|---|---|---|
| MoGe on masked image, all pixels NaN | `[nan, nan]` | All target_points are NaN → filter empties them | **True → IoU = -1** |
| MoGe on full scene, border pixels NaN | `[nan, nan]` | Some border pixels are NaN, chair pixels are valid | **False → pipeline continues** |

The `--scene-image` fix converts the first situation into the second. The log line is
**not diagnostic enough** to distinguish them. A better diagnostic would be:

```python
z = Point_Map[..., 2]
valid_frac = (~z.isnan()).float().mean()
print(f"Z range=[{z.nanmin():.3f}, {z.nanmax():.3f}], valid={valid_frac:.1%}")
```

---

## wooden_chair Alignment Result (2026-02-19 experiment)

For reference, the wooden_chair run today confirmed the corrected pipeline in practice:

| Stage | IoU | Notes |
|---|---|---|
| After Stage 5a (manual alignment) | 0.0938 | flag_notgt=False, target_points non-empty |
| After Stage 5b (ICP) | < 0.0938 (inferred) | ICP rejected — sparse frame, tight threshold |
| After Stage 5c (Adam) | 0.1684 | Improvement from Adam alone |

Both runs (with and without `--scene-image`) produced identical IoU ~0.17, confirming
that for wooden_chair the limiting factor is TRELLIS's initial wrong orientation — a local
minimum Adam cannot escape — not the depth source.
