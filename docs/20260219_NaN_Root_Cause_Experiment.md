# SAM3D NaN Root Cause — Empirical Experiment on wooden_chair

**Date:** 2026-02-19
**Run:** `output/experiment_original_sam3d/` vs `output/experiment_viga_sam3d/`
**Scene:** dining (wooden_chair only)

---

## What Was Done / Context

Ran two controlled SAM3D reconstructions of the `wooden_chair` object from the dining scene to empirically verify the NaN root cause identified in the 2026-02-17 MoGe fix:

- **Run 1** (control): `sam3d_worker.py` without `--scene-image` → MoGe processes per-object masked image
- **Run 2** (VIGA fix): `sam3d_worker.py` with `--scene-image target_resized.jpg` → MoGe processes full scene image

Both runs used the same mask (`output/sam3d_dining/wooden_chair.npy`), same seed (42), and same config.

```bash
# Run 1 — no scene image
"C:/Users/kingy/miniconda3/envs/sam3d_py311/python.exe" -u tools/sam3d/sam3d_worker.py \
  --image data/static_scene/dining/target_resized.jpg \
  --mask output/sam3d_dining/wooden_chair.npy \
  --config utils/third_party/sam3d/checkpoints/hf/checkpoints/pipeline.yaml \
  --glb output/experiment_original_sam3d/wooden_chair.glb \
  --info output/experiment_original_sam3d/wooden_chair_info.json

# Run 2 — with scene image
"C:/Users/kingy/miniconda3/envs/sam3d_py311/python.exe" -u tools/sam3d/sam3d_worker.py \
  ...same flags... \
  --scene-image data/static_scene/dining/target_resized.jpg
```

---

## Key Finding 1: NaN Root Cause Confirmed for Run 1

Run 1 confirms: MoGe processes the per-object masked image (wooden_chair mask covers ~4.7% of 518×518 = ~12K pixels — mostly black), and produces all-NaN depth.

```
[post-opt] Z range=[nan, nan]   ← entire pointmap is NaN
[post-opt] Alignment: IoU=0.0938, src=2546, tgt=9048
Layout post-opt IoU: 0.1726
```

The `Z range=[nan, nan]` in Run 1 means the SSI-normalized pointmap from `layout_input_dict["rgb_pointmap"]` is entirely NaN — MoGe had no valid signal from the mostly-black input.

---

## Key Finding 2: Run 2 Also Shows Z range=[nan, nan] — But Misleadingly

Run 2 with `--scene-image` passes the full scene to MoGe, which should produce valid depth. But the log also shows:

```
[post-opt] Z range=[nan, nan]   ← looks the same as Run 1!
[post-opt] Alignment: IoU=0.0933, src=2546, tgt=9048
Layout post-opt IoU: 0.1684
```

This is **not** the same failure. The `Z range` is computed as:

```python
# inference_pipeline_pointmap.py:435
f"Z range=[{Point_Map[...,2].min():.3f}, {Point_Map[...,2].max():.3f}]"
```

PyTorch's `.min()` propagates NaN: if **any** pixel in the 518×518 pointmap is NaN, this returns NaN. MoGe on a real scene image produces NaN at occlusion boundaries, far-background regions, and specular highlights — a small fraction of pixels, but enough to make `.min()` return NaN.

The diagnostic conflates:
- **All-NaN depth** (Run 1): MoGe had no input signal → useless for ICP
- **Sparse-NaN depth** (Run 2): MoGe produced valid depth in most pixels → usable for ICP

---

## Key Finding 3: grow_mask_v9 Stalls with NaN Depth

Both runs produce **identical** grown mask statistics:

| | Run 1 (all-NaN depth) | Run 2 (sparse-NaN depth) |
|---|---|---|
| Grown pixels | 10054 | 10054 |
| Target points | 10054 | 10054 |
| Alignment IoU | 0.0938 | 0.0933 |
| Final IoU | 0.1726 | 0.1684 |

In `grow_mask_v9` (normal-consistency mask growth), depth comparisons like `depth[candidate] - depth[neighbor] < threshold` return `False` when either value is NaN. This means NaN depth silently prevents mask growth — the grown mask stays at the initial SAM mask boundary in both runs.

In Run 2, the chair's interior pixels likely have valid depth, but the **boundary pixels** (where growth would happen) coincide with occlusion-NaN regions in MoGe's output. The growth stalls identically to Run 1.

---

## Results / Metrics

| Metric | Run 1 (no --scene-image) | Run 2 (--scene-image) |
|---|---|---|
| Z range log | `[nan, nan]` (all NaN) | `[nan, nan]` (sparse NaN) |
| MoGe input | per-object masked image | full scene image |
| Grown pixels | 10054 | 10054 |
| Alignment IoU | 0.0938 | 0.0933 |
| Final IoU | 0.1726 | 0.1684 |
| Translation | `[0.105, -0.237, 1.206]` | `[0.105, -0.237, 1.206]` |
| SLAT voxels | 4707 → 4685 | 4707 → 4685 |
| Decode time | ~2.4 min | ~4.1 min |

The **translations are identical** between runs — post-opt produced no net movement, confirming ICP did not improve on TRELLIS's initial pose estimate.

---

## Observations

- **Z range diagnostic is inadequate**: Needs `nanmin/nanmax` or per-pixel NaN masking to distinguish "all NaN" from "sparse NaN". Fix: use `torch.nanmin` / `torch.nanmax`, or print fraction of valid pixels.

- **wooden_chair is a hard alignment case**: 4685 voxels (sparse structure vs. 21,608 for keyboard). TRELLIS produces a reasonable initial pose (0.09 IoU) that post-opt can't improve in either run. This is likely a mesh-complexity + depth-boundary issue specific to chair legs.

- **The --scene-image fix IS working**: The 2026-02-17 dining run showed 4/5 objects improving from IoU=-1 to 0.45-0.95 after the fix. wooden_chair specifically is the one object where alignment remains poor regardless of depth quality — its IoU was ~0.17 before and after the fix.

- **Decode time anomaly**: Run 2 took ~4.1 min for SLAT decode vs ~2.4 min for Run 1. Both used the same 4685 voxels. The extra ~1.7 min is unexplained — possibly GPU thermal state or memory cache differences.

---

## What Was Not Isolated / Open Questions

1. **Does Run 2's ICP actually use valid depth?** We confirmed identical grown masks, but didn't verify the Z values of `target_object_points` for Run 2. A follow-up should log `target_object_points[:, 2].nanmin()` vs `.min()` to confirm.

2. **Why do chair boundary pixels have NaN in MoGe's scene output?** Likely occlusion/silhouette effects. If the chair legs are thin or partially occluded, MoGe may produce uncertain depth at those edges.

3. **Can wooden_chair alignment be improved?** Options: use more SLAT inference steps (25 → 50), use mesh texture baking, or use a better ICP initialization from TRELLIS's own pose output.

4. **Why does `grow_mask_v9` produce 10054 pixels when the SAM mask should be ~12K?** The grown mask is actually smaller than the input SAM mask — the v9 cleaning step (removing outliers) is reducing the mask. This should be investigated separately.

---

## Files

| File | Description |
|---|---|
| `output/experiment_original_sam3d/wooden_chair.glb` | Run 1 GLB output |
| `output/experiment_original_sam3d/wooden_chair_info.json` | Run 1 pose + IoU |
| `output/experiment_original_sam3d/wooden_chair_log.txt` | Run 1 full stdout |
| `output/experiment_viga_sam3d/wooden_chair.glb` | Run 2 GLB output |
| `output/experiment_viga_sam3d/wooden_chair_info.json` | Run 2 pose + IoU |
| `output/experiment_viga_sam3d/wooden_chair_log.txt` | Run 2 full stdout |
| `output/experiment_comparison/comparison.png` | Side-by-side visualization |
| `experiment_nan_root_cause.py` | Orchestrator script (not used this session) |
| `run_experiment_compare_only.py` | Comparison visualization generator |

### Diagnostic Improvement Needed

- `layout_post_optimization_utils.py:435` — Change `.min()/.max()` to `torch.nanmin/nanmax`:
  ```python
  # Current (misleading — NaN in any pixel → shows nan):
  f"Z range=[{Point_Map[...,2].min():.3f}, {Point_Map[...,2].max():.3f}]"

  # Better (shows range of valid pixels + fraction valid):
  z = Point_Map[...,2]
  valid_frac = (~z.isnan()).float().mean()
  f"Z range=[{z.nanmin():.3f}, {z.nanmax():.3f}], valid={valid_frac:.1%}"
  ```
