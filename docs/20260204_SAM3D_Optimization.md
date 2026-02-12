# SAM3D with Optimization

**Date:** February 4, 2026 | **Object:** Ito En Green Tea Bottle

> **Pipeline Complete!**
> Single Image → SAM3D → GLB Mesh → Blender Quaternion Rotation → GIF Animation

## Input Image

![Input Image](../output/test_sam/ito_en_green_tea_bottle.png)

*Original Input (2D Image + Mask)*

## Output: 3D Rotation GIFs

| | |
|:---:|:---:|
| ![X-axis rotation](test_results_images/bottle_obj_rotate/ito_en_green_tea_bottle_x_rotation.gif) | ![Y-axis rotation](test_results_images/bottle_obj_rotate/ito_en_green_tea_bottle_y_rotation.gif) |
| X-Axis Rotation (Pitch) | Y-Axis Rotation (Yaw) |

## Mesh Statistics

| Metric | Value |
|--------|-------|
| Vertices | 379K |
| Faces | 758K |
| GLB Size | 15 MB |
| Frames/Axis | 30 |
| SAM3D Time | ~9 min |

## Optimization: Quaternion vs Euler Rotation

> **Key Optimization:** Using quaternion rotation instead of Euler angles avoids gimbal lock. The object rotates smoothly around its own center while the camera stays fixed.

| Method | Result |
|--------|--------|
| Camera Orbit | Camera moves, object stays - less intuitive for product display |
| Object Euler Rotation | Gimbal lock at 90 degrees poles causes jerky motion |
| **Object Quaternion Rotation** | Smooth 360 degree rotation, no gimbal lock |

## Frame Samples

### X-Axis Rotation (0 degrees → 90 degrees → 180 degrees)

| | | |
|:---:|:---:|:---:|
| ![0 degrees](test_results_images/bottle_obj_rotate/ito_en_green_tea_bottle_x_00.png) | ![90 degrees](test_results_images/bottle_obj_rotate/ito_en_green_tea_bottle_x_07.png) | ![180 degrees](test_results_images/bottle_obj_rotate/ito_en_green_tea_bottle_x_15.png) |
| 0 degrees (Front) | ~90 degrees (Top) | 180 degrees (Back) |

### Y-Axis Rotation (0 degrees → 90 degrees → 180 degrees)

| | | |
|:---:|:---:|:---:|
| ![0 degrees](test_results_images/bottle_obj_rotate/ito_en_green_tea_bottle_y_00.png) | ![90 degrees](test_results_images/bottle_obj_rotate/ito_en_green_tea_bottle_y_07.png) | ![180 degrees](test_results_images/bottle_obj_rotate/ito_en_green_tea_bottle_y_15.png) |
| 0 degrees (Front) | ~90 degrees (Side) | 180 degrees (Back) |

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA RTX 5080 (16GB) |
| OS | Windows 11 |
| CUDA | 12.8 |
| PyTorch | 2.8.0+cu128 |
| Blender | 4.5.5 LTS |
| Conda Env | sam3d_py311 |

## Files

| Type | Path |
|------|------|
| Input Image | `output/test_sam/ito_en_green_tea_bottle.png` |
| GLB Mesh | `output/viga_test/ito_en_green_tea_bottle.glb` |
| X-Rotation GIF | `docs/test_results_images/bottle_obj_rotate/ito_en_green_tea_bottle_x_rotation.gif` |
| Y-Rotation GIF | `docs/test_results_images/bottle_obj_rotate/ito_en_green_tea_bottle_y_rotation.gif` |

---

*Generated: February 4, 2026*
