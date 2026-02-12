# SAM3D All Masks Pipeline

**Date:** February 4, 2026 | **Object:** Single Green Tea Bottle Image

> **Critical Observation!**
> Current mask generation appears to be detecting unexpected objects. All masks seem to identify objects different from the expected green tea bottle scene. **Requires detailed investigation of mask generation algorithm.**

> **Pipeline Technically Complete**
> Single Image → All Object Masks → SAM3D → GLB Meshes → Rotation GIFs

## Input Image

![Input Image](test_results_images/01_greentea_input.jpg)

*Original Input Image*

## Mask Rotation GIFs

| | |
|:---:|:---:|
| ![Green Tea Bottle X](test_results_images/all_masks_results/green_tea_bottle_x_rotation.gif) | ![Green Tea Bottle Y](test_results_images/all_masks_results/green_tea_bottle_y_rotation.gif) |
| Green Tea Bottle X-Axis | Green Tea Bottle Y-Axis |
| ![Ito En Bottle X](test_results_images/all_masks_results/ito_en_green_tea_bottle_x_rotation.gif) | ![Ito En Bottle Y](test_results_images/all_masks_results/ito_en_green_tea_bottle_y_rotation.gif) |
| Ito En Bottle X-Axis | Ito En Bottle Y-Axis |
| ![Bottle Cap X](test_results_images/all_masks_results/bottle_cap_x_rotation.gif) | ![Bottle Cap Y](test_results_images/all_masks_results/bottle_cap_y_rotation.gif) |
| Bottle Cap X-Axis | Bottle Cap Y-Axis |
| ![Label Wrap X](test_results_images/all_masks_results/label_wrap_x_rotation.gif) | ![Label Wrap Y](test_results_images/all_masks_results/label_wrap_y_rotation.gif) |
| Label Wrap X-Axis | Label Wrap Y-Axis |
| ![Bottle Neck X](test_results_images/all_masks_results/bottle_neck_x_rotation.gif) | ![Bottle Neck Y](test_results_images/all_masks_results/bottle_neck_y_rotation.gif) |
| Bottle Neck X-Axis | Bottle Neck Y-Axis |
| ![Headphones X](test_results_images/all_masks_results/headphones_x_rotation.gif) | ![Headphones Y](test_results_images/all_masks_results/headphones_y_rotation.gif) |
| Headphones X-Axis | Headphones Y-Axis |

## Mask Details

| Mask Name | Coverage (%) | GLB Size (MB) | Inference Time (min) |
|-----------|-------------|---------------|---------------------|
| Green Tea Bottle | 65.0% | 7.43 | 4.2 |
| Ito En Green Tea Bottle | 91.7% | 11.38 | 4.8 |
| Bottle Cap | 95.0% | 12.49 | 0.7 |
| Label Wrap | 95.5% | 14.22 | 1.2 |
| Bottle Neck | 96.2% | 10.67 | 0.9 |
| Headphones | 98.9% | 15.36 | 2.1 |

> **Observations and Concerns:**
> - Unexpected mask generation with unrelated objects detected
> - Mask coverage varies from 65% to 98.9%
> - GLB mesh sizes range from 7.43 to 15.36 MB
> - Inference times vary from 0.7 to 4.8 minutes
> - Strong indication of mask generation algorithm malfunction

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA RTX 5080 (16GB) |
| OS | Windows 11 |
| CUDA | 12.8 |
| PyTorch | 2.8.0+cu128 |
| Blender | 4.5.5 LTS |
| Conda Env | sam3d_py311 |

---

*Generated: February 4, 2026*
*Created by Yuna (Claude)*
