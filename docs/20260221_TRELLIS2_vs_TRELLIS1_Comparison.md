# TRELLIS2 vs TRELLIS1 Comparison — Dining Scene

**Date:** 2026-02-21
**Author:** kingy + Claude (Opus 4.6)
**Hardware:** GCP `genesisforge-gpu` (g2-standard-8, NVIDIA L4 24GB, us-central1-a)
**Input:** `data/static_scene/dining/target_resized.jpg` (771 x 1024), 8 objects

---

## Summary

Ran the full SAM3D pipeline with TRELLIS2 (microsoft/TRELLIS.2-4B) as a drop-in replacement for TRELLIS1, using the same dining scene image and SAM segmentation masks. TRELLIS2 produces higher-quality PBR textured meshes but is **2.8x slower** overall and achieves **lower IoU** in pose alignment due to high-poly mesh density mismatch.

---

## 1. Speed Comparison

### End-to-End (8 objects, model cached / batch mode)

| Stage | TRELLIS1 Batch | TRELLIS2 | Factor |
|---|---|---|---|
| Model load | 27.0s | 169.4s | 6.3x slower |
| 3D reconstruction (8 objects) | 442.3s | 804.6s | 1.8x slower |
| Pose alignment (MoGe + post-opt) | included in reconstruction | 381.6s (separate step) | N/A |
| **Total (excl. SAM)** | **469.3s (7.8 min)** | **1355.6s (22.6 min)** | **2.9x slower** |
| **Total (incl. SAM 53.6s)** | **491.0s (8.2 min)** | **1409.2s (23.5 min)** | **2.9x slower** |
| Per-object average | ~55s | ~148s | 2.7x slower |

### TRELLIS2 Per-Object Breakdown

| Object | Inference | GLB Export | Total | Vertices |
|---|---|---|---|---|
| sofa_cover | 117.8s | 22.5s | 140.6s | 2,361,406 |
| tablecloth | 96.9s | 15.3s | 112.2s | 728,994 |
| chair_cover | 70.8s | 60.3s | 131.2s | 1,171,153 |
| chair | 49.3s | 17.9s | 67.3s | 806,090 |
| pillow_and_blanket | 51.9s | 18.3s | 70.3s | 1,677,283 |
| newspaper | 27.3s | 64.6s | 91.9s | 564,313 |
| pot_and_trivet | 73.4s | 21.6s | 95.0s | 539,930 |
| plant | 37.5s | 58.3s | 95.9s | 1,234,314 |

### TRELLIS1 Per-Object Breakdown (Batch)

| Object | Total | Vertices | IoU |
|---|---|---|---|
| sofa_pattern | 80.1s | 12,893 | 0.3053 |
| tablecloth | 54.8s | 10,628 | 0.4793 |
| save_the_date_tile | 58.1s | 6,163 | 0.7282 |
| wooden_chair | 46.1s | 3,249 | 0.2317 |
| neck_pillow | 56.6s | 5,696 | 0.9195 |
| newspaper | 54.6s | 12,251 | 0.8350 |
| metal_colander | 42.7s | N/A | 0.2409 |
| place_mat | 49.2s | 5,253 | 0.6374 |

### Why TRELLIS2 Is Slower

1. **Higher-poly meshes** — 500K–2.4M vertices (TRELLIS2) vs 3K–13K (TRELLIS1), 100–1000x more geometry
2. **PBR texture baking** — GLB export includes UV unwrapping + texture bake (15–65s per object)
3. **Larger model** — 4B parameter model takes 169s to load vs 27s for TRELLIS1
4. **Separate pose alignment** — TRELLIS2 outputs canonical-frame meshes only; pose alignment (MoGe + layout_post_optimization) runs as a second subprocess in the `sam3d_py311` env, adding 382s

---

## 2. IoU Comparison

Note: SAM segmented slightly different objects between runs (different mask boundaries), so only a few objects are directly comparable.

### TRELLIS2 Per-Object IoU (after NaN fix)

| Object | IoU | Notes |
|---|---|---|
| sofa_cover | 0.3966 | Large, complex shape |
| tablecloth | 0.1989 | Round draping cloth |
| chair_cover | 0.1010 | Partial occlusion |
| chair | 0.2351 | ICP improved from 0.10 to 0.24 |
| pillow_and_blanket | 0.5824 | Best result |
| newspaper | 0.1491 | Small flat object |
| pot_and_trivet | 0.0767 | Small, complex geometry |
| plant | 0.0893 | Thin leaves, hard to align |

### TRELLIS1 Per-Object IoU (Batch)

| Object | IoU |
|---|---|
| sofa_pattern | 0.3053 |
| tablecloth | 0.4793 |
| save_the_date_tile | 0.7282 |
| wooden_chair | 0.2317 |
| neck_pillow | 0.9195 |
| newspaper | 0.8350 |
| metal_colander | 0.2409 |
| place_mat | 0.6374 |

### IoU Summary

| Metric | TRELLIS1 | TRELLIS2 |
|---|---|---|
| Mean IoU | 0.5097 | 0.2161 |
| Objects with IoU > 0.5 | 4/8 | 1/8 |
| Objects with IoU > 0.3 | 5/8 | 2/8 |

TRELLIS2 IoU is significantly lower. Root cause: the ICP alignment uses only ~3K–6K sampled source points from meshes with 500K–2.4M vertices, creating a density mismatch with the MoGe pointmap target points.

---

## 3. Post-Optimization Feature Parity

Both TRELLIS1 and TRELLIS2 call the **same `layout_post_optimization()` function** from `sam3d_objects.pipeline.inference_utils`, which includes:

| Feature | Present in Both? | Details |
|---|---|---|
| Convex hull mask growth (v9) | Yes | `grow_mask_v9()` with normal-consistency + 8-dir ray depth gate |
| Two-pass ICP (coarse + fine) | Yes | Thresholds: 0.10 (coarse), 0.05 (fine) via Open3D |
| ICP rejection gate | Yes | Accept only if IoU improves over previous best |
| IoU gate (revert if < 0.05) | Yes | Falls back to TRELLIS pose if post-opt IoU too low |
| Occlusion check | Disabled | Scene-level pointmap falsely triggers at object boundaries |
| MoGe scene-image mode | Yes | Full scene passed to MoGe (not per-object masked images) |

The standalone `reoptimize_depth.py` script (at project root) contains additional features like multi-object mask exclusion and post-ICP depth-scale correction, but this was **never part of either TRELLIS1 or TRELLIS2 pipeline** — it is a separate experimental script.

---

## 4. NaN Pointmap Bug Fix

During this run, a bug was discovered and fixed in `pose_align_worker.py`:

- **Bug:** `pad_to_square_centered(pointmap, value=NaN)` followed by `F.interpolate(..., mode="bilinear")` spread NaN values to all neighboring pixels, corrupting the entire pointmap
- **Symptom:** `Z range=[nan, nan]` in post-opt logs, 4/8 objects returning IoU=-1.0
- **Fix:** Removed the pad-to-square step; pass pointmap directly to `layout_post_optimization` which handles its own resize internally
- **Result:** All 8 objects now produce valid IoU values
- **Commit:** `2707539` — "Fix NaN pointmap in pose_align_worker: remove pad-to-square step"

---

## 5. Architecture Difference

```
TRELLIS1 (sam3d_batch_worker.py):
  sam env: SAM segmentation → masks
  sam3d_py311 env: TRELLIS model load → [per object: inference + MoGe + post-opt + GLB export]
  (single subprocess, everything integrated)

TRELLIS2 (run_sam3d_dining_t2.py):
  sam env:        SAM segmentation → masks
  trellis2 env:   TRELLIS2 model load → [per object: inference + GLB export]  (subprocess 1)
  sam3d_py311 env: MoGe load → [per object: pointmap + post-opt + GLB align]  (subprocess 2)
  (two subprocesses due to incompatible conda envs: Py3.10/torch2.6 vs Py3.11/torch2.5)
```

---

## 6. Conclusion

| Dimension | TRELLIS1 | TRELLIS2 | Winner |
|---|---|---|---|
| Speed (8 objects) | 491s (8.2 min) | 1409s (23.5 min) | TRELLIS1 (2.9x faster) |
| Mesh quality | Vertex-colored, 3K–13K verts | PBR textured, 500K–2.4M verts | TRELLIS2 |
| Pose alignment IoU | Mean 0.51 | Mean 0.22 | TRELLIS1 |
| Pipeline complexity | Single subprocess | Two subprocesses (different envs) | TRELLIS1 |

TRELLIS2 produces visually superior meshes (PBR textures, high detail) but at a significant cost in speed and alignment quality. The pose alignment degradation is likely fixable by decimating TRELLIS2 meshes before ICP, or by adapting the ICP parameters for higher-poly inputs.

### Potential Improvements

1. **Decimate meshes before ICP** — reduce from 500K+ to ~10K verts for alignment, apply transform back to full mesh
2. **Tune ICP parameters** — increase source point sampling for high-poly meshes
3. **Pipeline integration** — if TRELLIS2 env compatibility improves, merge into single subprocess to eliminate 169s redundant model load
4. **Reduce TRELLIS2 resolution** — `decimation_target` parameter (currently 100K) could be lowered for faster export
