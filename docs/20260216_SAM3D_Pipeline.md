# SAM3D Pipeline: Image to 3D GLB Objects

**Date:** 2026-02-16
**Code:** `utils/third_party/sam3d/` (Meta's original), `tools/sam3d/` (VIGA wrappers)

## Overview

SAM3D takes a single scene image and produces per-object 3D meshes (GLB files) positioned in camera space. It combines three models -- SAM (segmentation), MoGe (monocular geometry), and TRELLIS (3D reconstruction) -- with a multi-stage pose alignment pipeline.

## Pipeline Steps

```
Input Image (e.g., dining scene photo)
    |
    v
[Step 1] SAM Segmentation ----------- N binary masks
    |
    v
[Step 2] MoGe Depth Estimation ------ (H,W,3) pointmap + intrinsics
    |
    v
[Step 3] TRELLIS 3D Reconstruction -- per-object GLB mesh (model space)
    |
    v
[Step 4] Pose Decoding -------------- initial S, R, T per object
    |
    v
[Step 5] Layout Post-Optimization --- refined S, R, T (3 sub-stages)
    |
    v
[Step 6] Export ---------------------- GLBs with baked transforms
```

---

## Step 1: SAM Segmentation

**Conda env:** `sam` (Python 3.10)
**Model:** SAM ViT-H (Meta's Segment Anything Model)
**Code:** `tools/sam3d/sam_worker.py`

Takes the full scene image and produces per-object binary masks. Each mask is a (H, W) uint8 array with values 0 (background) or 255 (object).

**Input:** RGB image
**Output:** N binary masks (one per detected object)

---

## Step 2: MoGe Depth Estimation

**Conda env:** `sam3d_py311` (Python 3.11)
**Model:** MoGe (Monocular Geometry estimator)
**Code:** `utils/third_party/sam3d/sam3d_objects/pipeline/depth_models/moge.py`

Runs on the **full scene image** (not per-object). Produces a dense **pointmap** and camera intrinsics.

### What is a pointmap?

A pointmap is **NOT an RGB color image**. It is a per-pixel 3D coordinate map -- an (H, W, 3) array where each pixel maps to a 3D point in camera space:

```
pointmap[y, x] = (X, Y, Z)   # the 3D camera-space position of pixel (x, y)
```

- The **Z channel** is depth (distance from camera along the optical axis)
- The **X, Y channels** are the lateral 3D positions (horizontal and vertical offsets from the optical axis)
- This is richer than a depth map alone because it includes the full (X, Y, Z) -- you don't need camera intrinsics to go from pixel to 3D

**Visual comparison:**

| Representation | Shape | What each pixel stores | Example value at pixel (400, 300) |
|---|---|---|---|
| RGB image | (H, W, 3) | Red, Green, Blue color (0-255) | (180, 120, 85) |
| Depth map | (H, W) | Distance Z in meters | 1.45 |
| **Pointmap** | **(H, W, 3)** | **3D position (X, Y, Z) in camera space** | **(-0.12, 0.08, 1.45)** |

MoGe also estimates camera intrinsics (focal length, principal point) from the image.

**Input:** RGB image
**Output:**
- `points` -- (H, W, 3) dense pointmap in camera space
- `intrinsics_px` -- (3, 3) camera intrinsic matrix in pixel coordinates
- `depth` -- (H, W) depth map (the Z channel of the pointmap)

### How MoGe is called

**MoGe wrapper** (`depth_models/moge.py:5-10`):

```python
output = self.model.infer(image.to(self.device), force_projection=False)
pointmaps = output["points"]
```

**Pipeline call** (`inference_pipeline_pointmap.py:248-259`):

```python
output = self.depth_model(loaded_image)       # depth_model = MoGe
pointmaps = output["pointmaps"]               # (H, W, 3) in MoGe camera space
# Transform from MoGe camera space to PyTorch3D camera space
camera_convention_transform = Transform3d().rotate(
    camera_to_pytorch3d_camera(device=self.device).rotation
)
points_tensor = camera_convention_transform.transform_points(pointmaps)
intrinsics = output.get("intrinsics", None)
```

**Camera intrinsics inference** -- if MoGe doesn't provide intrinsics directly, they are recovered from the pointmap geometry (`pipeline/utils/pointmap.py:21-108`):

```python
# Recovers focal length and shift from the pointmap structure
shift, focal = recover_focal_shift(points, mask_binary)   # line 78
fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio
fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
```

There is **no separate camera pose estimation**. The camera is implicitly at the origin. MoGe outputs a pointmap directly in camera space.

MoGe is called once for the full scene image. The pointmap is then used in Steps 4 and 5 for pose estimation and alignment.

**Why `--scene-image` was needed despite MoGe already using the full image:** In the original pipeline, the Step 5 `Point_Map` was not taken directly from MoGe. It came from the sparse structure preprocessor's `rgb_pointmap`, which SSI-normalizes the pointmap using the per-object mask region. For small objects (< 5% of image pixels), SSI produced `scale ≈ 0`, causing NaN on inverse normalization. The `--scene-image` flag bypasses SSI entirely by forwarding the raw MoGe pointmap (`pipeline_pointmap`) straight to `run_post_optimization`.

### Where intrinsics are used

The intrinsics from Step 2 are the **single source of camera calibration for the entire pipeline**. They flow directly to the layout post-optimization:

```python
# inference_pipeline_pointmap.py:460
run_post_optimization(
    pipeline_pointmap=pointmap,
    intrinsics=pointmap_dict["intrinsics"],   # <-- from Step 2
    ...
)
```

Inside Stage 5c, they are used to construct `PerspectiveCameras` for differentiable silhouette rendering (`layout_post_optimization_utils.py:141-151`):

```python
cameras = PerspectiveCameras(
    focal_length=((fx, fy),),
    principal_point=((cx, cy),),
    image_size=((H, W),),
    device=device,
)
```

There is no ground-truth camera calibration input. MoGe's estimate is authoritative for all downstream stages. If MoGe's intrinsics are wrong, the silhouette renderer in Stage 5c will project the mesh incorrectly, affecting alignment quality.

---

## Step 3: TRELLIS 3D Reconstruction

**Conda env:** `sam3d_py311` (Python 3.11)
**Model:** TRELLIS (feed-forward flow matching)
**Code:** `utils/third_party/sam3d/sam3d_objects/pipeline/inference_pipeline_pointmap.py`

For **each** object mask:
1. Crop the image to the mask bounding box, apply alpha from the mask
2. Feed the RGBA image to TRELLIS

### What TRELLIS receives

**TRELLIS takes only an RGBA image** (RGB + alpha mask from segmentation). It does **NOT** receive the MoGe depth/pointmap for shape reconstruction.

The nuance: the MoGe pointmap is passed to the **sparse structure generator** (Stage 1) as conditioning to help with **pose prediction**, but the actual **shape reconstruction** (SLAT / Stage 2) uses only the image (`inference_pipeline_pointmap.py:344-348`):

```python
ss_input_dict = self.preprocess_image(
    image, self.ss_preprocessor, pointmap=pointmap   # Stage 1: gets pointmap
)
slat_input_dict = self.preprocess_image(
    image, self.slat_preprocessor                     # Stage 2: NO pointmap
)
```

This means the 3D mesh shape comes purely from the image appearance. The pointmap only influences where the mesh gets placed in camera space.

### TRELLIS internal stages

| Stage | Name | Code | Solver | Steps | Input | Output |
| ----- | ---- | ---- | ------ | ----- | ----- | ------ |
| 1 | Sparse Structure | `inference_pipeline.py:644-721` | ShortCut | 2 | Image + pointmap | Shape latent + pose heads (rotation, translation, scale) |
| 2 | Sparse Latent (SLAT) | `inference_pipeline.py:723-770` | Euler | 12 | Image only (NO pointmap) | Detailed shape latent |
| 3 | Dual Decode | `inference_pipeline.py:591-617` | -- | -- | Shape latent | 3D Gaussians (32/voxel, 64^3) -> mesh (~300K vertices) |

**Input:** RGBA image (per object)
**Output:** Mesh in TRELLIS model space (Z-up, ~[-0.5, 0.5] range)

**Note:** TRELLIS does NOT output any pixel-to-vertex mapping or 2D-to-3D correspondence. The output is a standalone 3D mesh with no link back to image pixels. All spatial anchoring to the image comes from MoGe's pointmap in Steps 4 and 5.

---

## Step 4: Pose Decoding

**Conda env:** `sam3d_py311`
**Component:** Deterministic math function (NOT a separate neural network)
**Code:** `inference_utils.py:224-327` (function `pose_decoder`)

### What is the pose decoder?

The "pose decoder" is **not a neural network** -- it is a pure Python function (a closure returned by `pose_decoder()` at `inference_utils.py:224`) that performs deterministic mathematical conversion. The actual pose **prediction** happens inside the sparse structure generator from Step 3 (Stage 1), which is an MM-DiT (Multi-Modal Diffusion Transformer) with **multiple output heads** that jointly predict:

- Shape latent
- 6D rotation (continuous rotation representation)
- Translation (in log-space)
- Scale (in log-space)

The pose decoder merely converts these raw network outputs into usable parameters.

### How it is called

In the main pipeline (`inference_pipeline_pointmap.py:358-366`):

```python
pointmap_scale = ss_input_dict.get("pointmap_scale", None)
pointmap_shift = ss_input_dict.get("pointmap_shift", None)
ss_return_dict.update(
    self.pose_decoder(
        ss_return_dict,
        scene_scale=pointmap_scale,
        scene_shift=pointmap_shift,
    )
)
```

### Conversion steps

1. **Remap keys** (`inference_utils.py:231-244`): Raw output keys (`"6drotation_normalized"`, `"translation"`, `"scale"`) are renamed to internal convention
2. **Denormalize rotation** (`inference_utils.py:253-260`): 6D rotation is un-standardized using pre-computed `ROTATION_6D_MEAN` and `ROTATION_6D_STD` (lines 49-67)
3. **6D -> quaternion** (`inference_utils.py:261-281`): Gram-Schmidt orthogonalization converts 6D rotation to a 3x3 rotation matrix, then to quaternion (wxyz)
4. **Exponentiate** (`inference_utils.py:283-291`): Scale and translation are predicted in log-space; `torch.exp()` converts them to real values
5. **Undo MoGe normalization** via `PoseTargetConverter.dicts_pose_target_to_instance_pose()` (`inference_utils.py:310-325`): The MoGe pointmap is normalized by `scene_scale` and `scene_shift` during preprocessing (`pose_target.py:361-372`):

   ```python
   # ScaleShiftInvariant.get_scale_and_shift (pose_target.py:361):
   shift_z = pointmap[..., -1].nanmedian()
   scale = (pointmap - shift).abs().nanmean()
   ```

   The `ScaleShiftInvariant.to_instance_pose()` (`pose_target.py:325-348`) reverses this normalization via `ssi_to_metric()` (`pose_target.py:374-380`) which composes `Transform3d().scale(scale).translate(shift)` to recover camera-space values.

6. **Pose target convention** (`pose_target.py:283-381`): Applies the `ScaleShiftInvariant` training convention to produce the final instance pose

**Input:** Raw network outputs from sparse structure generator + MoGe scale/shift
**Output:** Per-object quaternion (wxyz), translation (3D), scale (3D)

---

## Step 5: Layout Post-Optimization

**Conda env:** `sam3d_py311`
**Component:** Part of **Meta's original SAM3D code** (NOT something we added)
**Orchestrator:** `inference_utils.py:70-221` (function `layout_post_optimization`)
**Implementation:** `layout_post_optimization_utils.py` (alignment, ICP, differentiable rendering)

This is a **3-stage refinement** that adjusts each object's pose to better match the scene. The code has `Copyright (c) Meta Platforms, Inc. and affiliates` headers and was present in the initial commit of the SAM3D third-party code.

**Pipeline call** (`inference_pipeline_pointmap.py:396-413`):

```python
if with_layout_postprocess and self.layout_post_optimization_method is not None:
    postprocessed_pose = self.run_post_optimization(
        deepcopy(glb), pointmap_dict["intrinsics"], ss_return_dict, ss_input_dict,
    )
    ss_return_dict.update(postprocessed_pose)
```

**Occlusion check** (`inference_utils.py:98-106`): Before any optimization, `check_occlusion()` (`layout_post_optimization_utils.py:93-111`) tests if the mask touches the image border, is occluded by other objects (via depth comparison), or has internal holes. If any check triggers, the entire post-optimization is **skipped** and the raw Step 4 pose is returned.

### Stage 5a: Manual Alignment

**Code:** `layout_post_optimization_utils.py:167-222` (function `run_alignment`)
**Called at:** `inference_utils.py:109-113`

Aligns the mesh to the MoGe pointmap within the object's mask:

1. **Extract target points** (lines 180-189): Get 3D points from the MoGe pointmap **within the object mask only** (not the full image). Apply outlier filtering at the 90th depth percentile:

   ```python
   target_object_points = Point_Map[mask[0, 0].bool()]
   depth_quantile = torch.quantile(target_object_points[:, 2], 0.9)
   target_object_points = target_object_points[target_object_points[:, 2] <= depth_quantile]
   ```

2. **Scale matching** (lines 196-202): Scale the mesh to match the height (Y-extent) of the target point cloud
3. **Centroid alignment** (lines 204-209): Translate the mesh centroid to match the target point cloud centroid
4. **Quality check** (line 219): Compute silhouette IoU via `compute_iou()` (line 329)
5. **flag_notgt check** (`inference_utils.py:123-132`): If the grown mask contains **zero** valid MoGe points (`target_object_points.shape[0] == 0`), the entire layout post-optimization is aborted immediately -- returns the raw TRELLIS pose with IoU = -1:

   ```python
   flag_notgt = target_object_points.shape[0] == 0
   if flag_notgt:
       return original_trellis_pose, iou=-1
   ```

   This was the failure mode for all 5 objects before the `--scene-image` fix. MoGe always ran on the full scene image — the NaN was NOT from MoGe. It came from SSI (ScaleShiftInvariant) normalization in the sparse structure preprocessor: the full-scene pointmap was cropped to the object bounding box, resized to 518×518, then SSI-normalized using the per-object mask region. For small objects (mask < 5% of image), the mask region had near-zero variance → `scale ≈ 0` → inverse normalization (`_apply_metric_to_ssi(..., apply_inverse=True)`) produced NaN/inf. The `--scene-image` fix bypasses the SSI chain entirely by passing the raw MoGe pointmap directly, making `flag_notgt` almost always False.

**Space:** 3D camera space (PyTorch3D convention: X-left, Y-up, Z-forward)
**Target:** Masked MoGe pointmap (only pixels within the object mask, with depth outlier filtering)

### Stage 5b: ICP (Iterative Closest Point)

**Code:** `layout_post_optimization_utils.py:383-403` (function `run_ICP`)
**Called at:** `inference_utils.py:127-165`

**What is ICP?** An iterative algorithm that finds the rigid transformation (rotation + translation) that best aligns two 3D point clouds. At each iteration, it finds the closest point in the target for each source point, then computes the transformation that minimizes the sum of squared distances.

**Implementation** (`layout_post_optimization_utils.py:390-397`): Open3D `registration_icp` with `TransformationEstimationPointToPoint`:

```python
reg_p2p = o3d.pipelines.registration.registration_icp(
    src_pcd, tgt_pcd, threshold=0.05, trans_init=np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
)
```

**Key details:**
- Operates in **3D space** (not 2D image projections) -- it aligns 3D point clouds directly
- **Source:** Mesh vertices (from TRELLIS, after Stage 5a alignment)
- **Target:** MoGe pointmap points **within the object mask only** (masked region, with depth outlier filtering from Stage 5a)
- **Threshold:** 0.05 camera-space units -- point pairs farther apart than this are ignored
- **Acceptance** (`inference_utils.py:140`): Accepted **only if** the resulting silhouette IoU improves (`ori_iou_shapeICP > ori_iou`); otherwise the ICP result is discarded (lines 156-165)

### Stage 5c: Differentiable Rendering Optimization

**Code:** `layout_post_optimization_utils.py:406-445` (function `run_render_compare`)
**Called at:** `inference_utils.py:175-205`

**What is differentiable rendering?** A rendering technique where the entire render pipeline is implemented with differentiable operations, so gradients can flow backward from a 2D image loss through the renderer to 3D parameters. This allows optimizing 3D pose parameters (rotation, translation, scale) by:
1. Rendering the mesh from the current pose
2. Comparing the render to a target (here: ground-truth segmentation mask)
3. Computing gradients of the loss w.r.t. the pose parameters
4. Updating the parameters with gradient descent

This is **part of Meta's original SAM3D code** -- it was NOT added by us. It is the final refinement stage in the original pipeline.

**Renderer** (`layout_post_optimization_utils.py:152-164`, function `get_mask_renderer`): PyTorch3D `MeshRenderer` with `SoftSilhouetteShader`:

```python
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4)),
)
```

**What it renders:** A **soft silhouette ONLY** (alpha channel). It does **NOT** render depth or color. The `SoftSilhouetteShader` produces a smooth, differentiable approximation of the object silhouette -- a grayscale image where pixels are 1 inside the object and 0 outside, with soft edges that allow gradient flow.

**Camera setup** (`layout_post_optimization_utils.py:141-151`): Uses the MoGe-estimated intrinsics (focal length, principal point) with PyTorch3D's `PerspectiveCameras`.

**Parameters optimized** (`layout_post_optimization_utils.py:408-414`):
- Quaternion (rotation) -- 4 values, wxyz
- Translation (3D) -- 3 values
- Scale (uniform) -- 1 value

**Two-stage optimization** (`layout_post_optimization_utils.py:416-440`):

| Stage | Iterations | Learning Rate | Parameters | Why |
| ----- | ---------- | ------------- | ---------- | --- |
| 1 | 5 | 1e-2 | Translation + Scale only | Coarse positioning first |
| 2 | 25 | 5e-3 | Quaternion + Translation + Scale | Fine-tune with rotation |

**Loss function** (`layout_post_optimization_utils.py:243-265`, function `compute_loss`):

```
L = 200   * MSE(rendered_silhouette, gt_mask)     # silhouette alignment
  + 0.1   * MSE(quaternion, identity_quaternion)   # rotation regularization
  + 0.05  * ||translation||^2                      # translation regularization
  + 0.05  * (scale - 1)^2                          # scale regularization
```

The silhouette term has weight 200, dominating the loss. The regularization terms prevent the optimizer from drifting too far from the initial pose.

**Acceptance criteria** (`inference_utils.py:190-193`): The optimized result is accepted only if:
- Final IoU > 0.5 (minimum quality threshold)
- Final IoU > pre-optimization IoU (must actually improve)

If either condition fails, **the entire post-optimization is rejected** -- not just Stage 5c, but also Stages 5a and 5b. The code reverts to the raw Step 4 pose (`tfm = tfm_ori`, line 192). There is a commented-out alternative (line 193) that would keep Stages 5a+5b while rejecting only 5c, but it is not active.

### Acceptance / Rejection Decision Tree

Three rejection checkpoints in `layout_post_optimization` (`inference_utils.py:70-221`):

| Checkpoint | Code Location | Condition | Result |
| --- | --- | --- | --- |
| Occlusion check | `inference_utils.py:98-106` | Border touch / occluded / holed mask | Skip all optimization → return raw Step 4 pose |
| `flag_notgt` | `inference_utils.py:123-132` | No MoGe points inside grown mask | Abort → return raw TRELLIS pose, IoU = -1 |
| ICP rejection | `inference_utils.py:140-165` | `ori_iou_shapeICP <= ori_iou` | Keep Stage 5a, discard ICP (`tfm2 = identity`) |
| Global rejection | `inference_utils.py:190-193` | `optimized_iou < 0.5` OR `<= ori_iou` | Full revert: `tfm = tfm_ori` → raw Step 4 pose |

```
layout_post_optimization()
    |
    +--> [Occlusion check] failed?  →  return raw Step 4 pose
    |
    +--> [Stage 5a] run_alignment()
    |        |
    |        +--> flag_notgt (no valid points)?  →  return raw TRELLIS pose, IoU = -1
    |
    +--> [Stage 5b] run_ICP()
    |        |
    |        +--> IoU improved?  -- yes -->  keep ICP result (tfm2 = ICP)
    |                            -- no  -->  discard ICP   (tfm2 = identity, keeps 5a only)
    |
    +--> [Stage 5c] run_render_compare()  (30-step Adam)
             |
             +--> IoU > 0.5 AND improved?  -- yes -->  accept (5a + 5b + 5c)
                                           -- no  -->  tfm = tfm_ori → full revert to Step 4
```

**Key asymmetry:** ICP rejection is *local* (Stage 5a is preserved). Stage 5c rejection is *global* (reverts 5a + 5b + 5c together). A Stage 5c failure produces the same output as if no optimization ran at all.

### Critical Limitation of Step 5

**The entire layout post-optimization (all 3 stages) only optimizes for 2D silhouette alignment. There is NO depth loss term anywhere in the pipeline.**

- Stage 5a aligns centroids in 3D (which includes depth), but doesn't minimize depth error
- Stage 5b ICP aligns in 3D, but acceptance is judged by 2D silhouette IoU (`inference_utils.py:140`)
- Stage 5c explicitly optimizes silhouette MSE only (`layout_post_optimization_utils.py:247`)

This means objects can achieve good 2D alignment (high mask IoU) while having incorrect depth placement. This is the **root cause** of the depth alignment issues we diagnosed in `20260216_Depth_Alignment_Analysis.md`:
- Sofa: 96% mask coverage but 51% depth error
- Chair legs: 94% mask coverage but 55% depth error

A depth loss term in Step 5c would directly fix this:

```python
# Proposed fix: add depth loss to the differentiable rendering loop
depth_loss = L1(projected_vertex_z, moge_depth_at_projected_pixels)
total_loss = 200 * silhouette_loss + lambda_depth * depth_loss + regularization
```

---

## Step 6: Export

**Code:** `tools/sam3d/sam3d_worker.py`
**Transform function:** `transform_mesh_vertices` (lines 113-152)
**Main export logic:** `main()` (lines 155-217)

1. **Coordinate conversion** (lines 104-110, 142): Apply Z-up to Y-up rotation to TRELLIS mesh vertices. This matches `layout_post_optimization_utils.get_mesh()` line 116:

   ```python
   # R_zup_to_yup (sam3d_worker.py:108-110):
   R_zup_to_yup = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
   vertices = vertices @ R_zup_to_yup   # line 142
   ```

2. **Bake transforms** (lines 143-152): Apply the final S, R, T into vertex positions using a custom `Transform3d` class (lines 46-91) that follows PyTorch3D row-vector convention (`points_h @ M`). After this, the GLB file contains vertices already in camera space -- no additional transforms needed at render time:

   ```python
   tfm = Transform3d().scale(scale).rotate(R_mat).translate(tx, ty, tz)
   vertices_world = tfm.transform_points(vertices)
   ```

3. **Export** (lines 182-183): Save as `.glb` per object via `trimesh.export()`
4. **Save metadata** (lines 198-213):
   - `{name}_info.json` -- per-object transform details (translation, rotation wxyz, scale, intrinsics)
   - Additional files saved by the orchestrator (`tools/sam3d/init.py`):
     - `object_transforms.json` -- all objects' translation, rotation (wxyz quaternion), scale
     - `target_moge.npz` -- MoGe pointmap, intrinsics, depth
     - `{name}.npy` -- per-object binary masks (uint8, 0/255)

**Note:** The `Transform3d` class in `sam3d_worker.py` (lines 46-91) is a **custom pure-PyTorch reimplementation**, NOT PyTorch3D's `Transform3d`. It was written to avoid a PyTorch3D dependency in the worker. It stores translation in row 3 (matching PyTorch3D's row-vector convention), which was the subject of a previous bug fix (see MEMORY.md: "SAM3D Transform3d Bugs Fixed").

---

## Conda Environment Summary

| Conda Env | Python | Used By | Key Packages |
|-----------|--------|---------|--------------|
| `sam3d` | 3.11 | Orchestrator (`tools/sam3d/init.py`) | Subprocess management only |
| `sam` | 3.10 | SAM segmentation (`sam_worker.py`) | SAM ViT-H, PyTorch |
| `sam3d_py311` | 3.11 | TRELLIS + MoGe + layout opt (`sam3d_worker.py`) | TRELLIS, MoGe, PyTorch3D, Open3D |

**Why three separate environments?** SAM and TRELLIS have incompatible PyTorch/CUDA dependencies and cannot coexist in one environment. The orchestrator (`sam3d` env) spawns each worker as a subprocess in its own conda env:

```
sam3d env (orchestrator)
  |-- subprocess --> sam env       --> SAM segmentation (produces masks)
  |-- subprocess --> sam3d_py311   --> MoGe + TRELLIS + layout optimization
```

The env mapping is defined in `utils/_path.py`. Each MCP tool script automatically launches its subprocess in the correct env.

---

## Coordinate Systems

```
TRELLIS model space:   Z-up, range ~[-0.5, 0.5]
PyTorch3D camera:      X-left,  Y-up,   Z-forward
OpenCV camera:         X-right, Y-down,  Z-forward
MoGe pointmap:         Same convention as PyTorch3D camera space

Conversion (TRELLIS -> PT3D):
    1. Z-up to Y-up rotation: verts @ [[1,0,0],[0,0,-1],[0,1,0]].T
    2. Apply S, R, T from pose decoder + layout optimization

Conversion (PT3D -> OpenCV):
    X_cv = -X_pt3d
    Y_cv = -Y_pt3d
    Z_cv =  Z_pt3d

Projection (OpenCV -> pixel):
    u = fx * X_cv / Z_cv + cx
    v = fy * Y_cv / Z_cv + cy

Quaternion convention: wxyz (w = real part first)
Transform order: v' = v @ R_pre @ S @ R @ T  (right-multiply, homogeneous)
```

---

## Key Files

| File | Role |
|------|------|
| `tools/sam3d/init.py` | MCP tool server (orchestrator) |
| `tools/sam3d/sam_worker.py` | SAM segmentation subprocess |
| `tools/sam3d/sam3d_worker.py` | TRELLIS + MoGe + layout subprocess |
| `tools/sam3d/sam3d_worker_v2.py` | TRELLIS.2 worker (simpler path) |
| `tools/sam3d/adapters/trellis2_adapter.py` | TRELLIS.2 adapter (no layout opt, identity transforms) |
| `utils/third_party/sam3d/notebook/inference.py` | High-level inference wrapper |
| `utils/third_party/sam3d/sam3d_objects/pipeline/inference_pipeline_pointmap.py` | Main inference pipeline (orchestrates all stages) |
| `utils/third_party/sam3d/sam3d_objects/pipeline/inference_utils.py` | Pose decoder + layout optimization orchestration |
| `utils/third_party/sam3d/sam3d_objects/pipeline/layout_post_optimization_utils.py` | Manual alignment + ICP + differentiable rendering |
| `utils/third_party/sam3d/sam3d_objects/pipeline/depth_models/moge.py` | MoGe model wrapper |
| `utils/third_party/sam3d/sam3d_objects/data/dataset/tdfy/pose_target.py` | Pose target conventions (ScaleShiftInvariant, etc.) |

---

## Output Files

For a scene with N objects, the pipeline produces:

| File | Description |
|------|-------------|
| `target_moge.npz` | MoGe output: `points` (H,W,3), `depth` (H,W), `intrinsics_px` (3,3), `image_width`, `image_height` |
| `object_transforms.json` | Array of N objects with `translation`, `rotation` (wxyz), `scale`, `object_name` |
| `{object_name}.glb` | Per-object mesh with transforms baked into vertices (PyTorch3D camera space) |
| `{object_name}.npy` | Per-object binary mask (H, W) uint8, 0 or 255 |
| `{object_name}_info.json` | Per-object transform details |

---

## Known Limitations

1. **No depth loss in layout optimization:** The differentiable rendering step (5c) only uses silhouette loss. Objects can be well-aligned in 2D but at the wrong depth. See `20260216_Depth_Alignment_Analysis.md` for quantitative analysis.

2. **No pixel-to-vertex mapping from TRELLIS:** TRELLIS outputs a standalone mesh with no correspondence back to image pixels. All spatial anchoring comes from MoGe's pointmap, not from TRELLIS itself.

3. **Sequential object processing:** Objects are reconstructed one at a time. No inter-object depth consistency is enforced -- each object is aligned to MoGe independently.

4. **MoGe depth ambiguity:** MoGe estimates depth from a single image, which has inherent scale ambiguity. The pointmap normalization (scale/shift) partially addresses this, but errors propagate to all downstream steps.

5. **Acceptance threshold at IoU > 0.5:** If the layout optimization fails to reach 0.5 IoU, the entire optimization is rejected and the raw pose from Step 4 is used -- which may be significantly worse.
