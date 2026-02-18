# SAM3D Convex Hull v2 — Results & Findings

**Date:** 2026-02-18
**Run:** `output/sam3d_convex_hull_v2/`
**Scene:** Greentea (5 objects)

---

## What Was Run

Batch SAM3D reconstruction using two combined fixes:

1. **Convex hull mask growth** — segmentation mask grows toward the convex hull of the object, stopping at depth edges. Gives TRELLIS more silhouette context.
2. **Scene-image MoGe fix** — full scene image passed to MoGe instead of the per-object masked image (which is mostly black and causes NaN). See [key finding](#key-finding-scene-image-moge-is-our-modification-not-original-sam3d) below.

Run script: `run_sam3d_convex_hull.sh`
Input masks: `output/test/greentea/sam_init/`
Scene image: `data/static_scene/greentea/target.png`

---

## IoU Results

| Object | IoU | Notes |
|---|---|---|
| ito_en_bottle | **0.9486** | Excellent — clean bottle shape |
| envelope | **0.8576** | Good — flat rectangular shape |
| headphones | **0.8256** | Good — small capsule shape |
| alienware_keyboard | **0.6466** | Moderate — flat plate, slightly offset |
| green_tea_bottle | **0.4545** | Poor — degenerate flat disk (see below) |

Before this fix, all 5 objects had IoU = -1 (NaN pointmap → pipeline fails).

---

## 2D Projection Visualization

Generated `visualize_sam3d_convex_hull.py`, matching the `sam3d_dining_v4` format:

- **`scene_2d_comparison.png`** — side-by-side: 2D masks (left) vs 3D projections (right) on grayscale depth map background
- **`{name}_compare.png`** — 3-panel per object: photo + mask outline | depth-colored 3D projection | object-colored projection on depth map

Output: `output/sam3d_convex_hull_v2/vis/`

### Observations from Projections

- **ito_en_bottle**: 3D projection aligns perfectly with the bottle region in the depth map. Bottle silhouette is clearly reconstructed.
- **alienware_keyboard**: Correct rectangular plate shape. Positioned to the right of scene — approximately correct.
- **headphones**: Small capsule/disc shape at top-right of depth map — correct position.
- **envelope**: Flat rectangular shape at top-left — correct position.
- **green_tea_bottle**: Reconstruction degenerated to a flat disk. The projection shows a circular ring at the wrong depth. The TRELLIS model likely reconstructed the dark shadow/table area around the bottle cap rather than the bottle body itself. This is the object with the largest shadow artifact in the SAM segmentation mask.

---

## Key Finding: Scene-Image MoGe is Our Modification, Not Original SAM3D

The original SAM3D pipeline (`InferencePipelinePointMap.run()`) runs MoGe on each individual object's masked RGBA image:

```python
# Original behavior
pointmap_dict = self.compute_pointmap(image, pointmap)
# where `image` = per-object RGBA (background transparent/black)
```

Our `--scene-image` fix adds a branch:

```python
# Our modification (inference_pipeline_pointmap.py, run())
if scene_image is not None:
    scene_rgba = np.concatenate([np.array(scene_image)[..., :3], alpha], axis=-1)
    pointmap_dict = self.compute_pointmap(scene_rgba, pointmap)  # full scene
else:
    pointmap_dict = self.compute_pointmap(image, pointmap)        # original
```

**Why the original design fails here:** The original SAM3D is designed for clean single-object close-up crops (like the Meta demo). In our use case, per-object PNGs are full scene size (771×1024) with most pixels black — MoGe needs texture/structure to estimate depth:

| Object | Visible pixels | Original behavior |
|---|---|---|
| green_tea_bottle | ~35% | borderline (NaN in some cases) |
| ito_en_bottle | ~22% | NaN → IoU = -1 |
| alienware_keyboard | ~10% | NaN → IoU = -1 |
| envelope | ~2.5% | NaN → IoU = -1 |
| headphones | ~1.5% | NaN → IoU = -1 |

**Our fix:** pass the full scene image (same pixel coordinates, plenty of texture) to MoGe, then use the resulting scene-level pointmap for the object's post-optimization. This works because the per-object PNGs are already full-scene-sized — pixel coordinates align without any warping.

---

## What Was Not Isolated

The convex hull v2 run combined both changes simultaneously. The **individual contribution** of convex hull mask growth vs. scene-image MoGe fix is not yet measured. A comparison baseline (scene-image MoGe only, no convex hull growth) would be needed to separate them.

---

## Files

| File | Description |
|---|---|
| `run_sam3d_convex_hull.sh` | Batch runner for all 5 objects |
| `visualize_sam3d_convex_hull.py` | 2D projection visualization script |
| `output/sam3d_convex_hull_v2/*.glb` | Reconstructed meshes (PyTorch3D camera space) |
| `output/sam3d_convex_hull_v2/*_info.json` | IoU + transform data per object |
| `output/sam3d_convex_hull_v2/vis/` | Visualization images |
| `output/sam3d_convex_hull_v2/gifs/` | Rotation GIFs (Blender Cycles, 24-frame, 384×384) |

### Key Code Changes (sam3d submodule)

- `utils/third_party/sam3d/sam3d_objects/pipeline/inference_pipeline_pointmap.py` — `scene_image` param in `run()`, pad-to-square alignment in `run_post_optimization()`
- `utils/third_party/sam3d/sam3d_objects/pipeline/inference_utils.py` — occlusion pre-check disabled (falsely triggered on scene-level depth boundaries)
- `utils/third_party/sam3d/notebook/inference.py` — `scene_image` param forwarded to pipeline
- `tools/sam3d/sam3d_worker.py` — `--scene-image` CLI arg
