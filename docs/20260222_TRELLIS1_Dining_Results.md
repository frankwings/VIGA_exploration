# TRELLIS1 Dining Scene — SAM3D Reconstruction Results

**Date:** 2026-02-22
**Hardware:** GCP g2-standard-8, NVIDIA L4 24GB, us-central1-a
**Pipeline:** SAM ViT-H segmentation + TRELLIS1 batch reconstruction (model cached)
**Total time:** 567.9s (9.5 min) — SAM 52.7s + TRELLIS1 515.2s

---

## 1. Input Image

**Source:** `data/static_scene/dining/target_resized.jpg` (771 x 1024)

![Input Image](../data/static_scene/dining/target_resized.jpg)

---

## 2. Segmentation Masks

SAM ViT-H detected **8 objects**. Each mask is shown as a red overlay on the original image.

### All Masks Overview

![All Masks Grid](../output/sam3d_dining_t1/viz/all_masks_grid.png)

### Per-Object Masks

#### sofa_with_patterned_cover
| Mask Overlay | Binary Mask | Segmented Input |
|:---:|:---:|:---:|
| ![mask](../output/sam3d_dining_t1/viz/sofa_with_patterned_cover_mask.png) | ![binary](../output/sam3d_dining_t1/viz/sofa_with_patterned_cover_mask_binary.png) | ![seg](../output/sam3d_dining_t1/sofa_with_patterned_cover.png) |

#### table_with_flower_tablecloth
| Mask Overlay | Binary Mask | Segmented Input |
|:---:|:---:|:---:|
| ![mask](../output/sam3d_dining_t1/viz/table_with_flower_tablecloth_mask.png) | ![binary](../output/sam3d_dining_t1/viz/table_with_flower_tablecloth_mask_binary.png) | ![seg](../output/sam3d_dining_t1/table_with_flower_tablecloth.png) |

#### broken_tile
| Mask Overlay | Binary Mask | Segmented Input |
|:---:|:---:|:---:|
| ![mask](../output/sam3d_dining_t1/viz/broken_tile_mask.png) | ![binary](../output/sam3d_dining_t1/viz/broken_tile_mask_binary.png) | ![seg](../output/sam3d_dining_t1/broken_tile.png) |

#### wooden_chair
| Mask Overlay | Binary Mask | Segmented Input |
|:---:|:---:|:---:|
| ![mask](../output/sam3d_dining_t1/viz/wooden_chair_mask.png) | ![binary](../output/sam3d_dining_t1/viz/wooden_chair_mask_binary.png) | ![seg](../output/sam3d_dining_t1/wooden_chair.png) |

#### neck_pillow
| Mask Overlay | Binary Mask | Segmented Input |
|:---:|:---:|:---:|
| ![mask](../output/sam3d_dining_t1/viz/neck_pillow_mask.png) | ![binary](../output/sam3d_dining_t1/viz/neck_pillow_mask_binary.png) | ![seg](../output/sam3d_dining_t1/neck_pillow.png) |

#### newspaper
| Mask Overlay | Binary Mask | Segmented Input |
|:---:|:---:|:---:|
| ![mask](../output/sam3d_dining_t1/viz/newspaper_mask.png) | ![binary](../output/sam3d_dining_t1/viz/newspaper_mask_binary.png) | ![seg](../output/sam3d_dining_t1/newspaper.png) |

#### metal_colander
| Mask Overlay | Binary Mask | Segmented Input |
|:---:|:---:|:---:|
| ![mask](../output/sam3d_dining_t1/viz/metal_colander_mask.png) | ![binary](../output/sam3d_dining_t1/viz/metal_colander_mask_binary.png) | ![seg](../output/sam3d_dining_t1/metal_colander.png) |

#### placemat
| Mask Overlay | Binary Mask | Segmented Input |
|:---:|:---:|:---:|
| ![mask](../output/sam3d_dining_t1/viz/placemat_mask.png) | ![binary](../output/sam3d_dining_t1/viz/placemat_mask_binary.png) | ![seg](../output/sam3d_dining_t1/placemat.png) |

---

## 3. 3D Reconstructions (Rotation GIFs)

Each object was reconstructed by TRELLIS1 into a GLB mesh, then rendered as a 360-degree turntable using Blender Cycles (512x512, 24 frames, GPU).

| Object | IoU | Y-Rotation (turntable) | X-Rotation (tumble) |
|:---|:---:|:---:|:---:|
| neck_pillow | **0.90** | ![y](../output/sam3d_dining_t1/rotation_gifs/neck_pillow_y_rotation.gif) | ![x](../output/sam3d_dining_t1/rotation_gifs/neck_pillow_x_rotation.gif) |
| newspaper | **0.84** | ![y](../output/sam3d_dining_t1/rotation_gifs/newspaper_y_rotation.gif) | ![x](../output/sam3d_dining_t1/rotation_gifs/newspaper_x_rotation.gif) |
| broken_tile | **0.72** | ![y](../output/sam3d_dining_t1/rotation_gifs/broken_tile_y_rotation.gif) | ![x](../output/sam3d_dining_t1/rotation_gifs/broken_tile_x_rotation.gif) |
| placemat | **0.62** | ![y](../output/sam3d_dining_t1/rotation_gifs/placemat_y_rotation.gif) | ![x](../output/sam3d_dining_t1/rotation_gifs/placemat_x_rotation.gif) |
| table_with_flower_tablecloth | **0.48** | ![y](../output/sam3d_dining_t1/rotation_gifs/table_with_flower_tablecloth_y_rotation.gif) | ![x](../output/sam3d_dining_t1/rotation_gifs/table_with_flower_tablecloth_x_rotation.gif) |
| sofa_with_patterned_cover | **0.25** | ![y](../output/sam3d_dining_t1/rotation_gifs/sofa_with_patterned_cover_y_rotation.gif) | ![x](../output/sam3d_dining_t1/rotation_gifs/sofa_with_patterned_cover_x_rotation.gif) |
| metal_colander | **0.24** | ![y](../output/sam3d_dining_t1/rotation_gifs/metal_colander_y_rotation.gif) | ![x](../output/sam3d_dining_t1/rotation_gifs/metal_colander_x_rotation.gif) |
| wooden_chair | **0.23** | ![y](../output/sam3d_dining_t1/rotation_gifs/wooden_chair_y_rotation.gif) | ![x](../output/sam3d_dining_t1/rotation_gifs/wooden_chair_x_rotation.gif) |

---

## 4. Rendered 3D Scene on 2D Image

All 8 reconstructed objects placed in a Blender scene using MoGe camera intrinsics, rendered with EEVEE (64 TAA samples).

### Full Scene Comparison

![Full Scene Comparison](../output/sam3d_dining_t1/full_scene_comparison.png)

**Left:** Original target photograph | **Right:** TRELLIS1 3D render with MoGe camera alignment

### 3D Render Only

![Full Scene Render](../output/sam3d_dining_t1/full_scene_render.png)

---

## 5. Summary

| # | Object | IoU | GLB Size | Notes |
|:---:|:---|:---:|:---:|:---|
| 1 | neck_pillow | 0.90 | 1.2 MB | Best alignment — compact, well-defined shape |
| 2 | newspaper | 0.84 | 1.4 MB | Flat rectangular, good match |
| 3 | broken_tile | 0.72 | 1.4 MB | Small tile/card on chair seat |
| 4 | placemat | 0.62 | 1.4 MB | Flat on table surface |
| 5 | table_with_flower_tablecloth | 0.48 | 1.6 MB | Round table with cloth draping |
| 6 | sofa_with_patterned_cover | 0.25 | 1.7 MB | Large complex shape, low IoU expected |
| 7 | metal_colander | 0.24 | 1.2 MB | Small object with holes |
| 8 | wooden_chair | 0.23 | 1.2 MB | Complex 3D structure (back, legs, seat) |

**Average IoU:** 0.53
**Success rate:** 8/8 (100%)
**Total pipeline time:** 9.5 min (SAM 52.7s + TRELLIS1 batch 515.2s)

### Pipeline Configuration

| Setting | Value |
|:---|:---|
| Segmentation | SAM ViT-H (automatic mask generation) |
| 3D Reconstruction | TRELLIS1 (batch mode, model cached) |
| Depth Estimation | MoGe ViT-L (scene-level pointmap) |
| Pose Alignment | layout_post_optimization (ICP + gradient) |
| Scene Render | Blender EEVEE, 771x1024, 64 TAA samples |
| Rotation Render | Blender Cycles GPU, 512x512, 64 samples, 24 frames |
| GPU | NVIDIA L4 24GB |

### Output Files

```
output/sam3d_dining_t1/
├── full_scene_comparison.png       # Target vs 3D render side-by-side
├── full_scene_render.png           # All 8 objects in one Blender scene
├── object_transforms.json          # Per-object pose (quaternion, translation, scale)
├── moge_intrinsics.npz             # MoGe camera intrinsics (pixel units)
├── summary.json                    # Pipeline summary with IoU per object
├── *.glb                           # 8 reconstructed GLB meshes
├── *.png                           # 8 segmented input images
├── *.npy                           # 8 binary masks
├── *_info.json                     # 8 per-object pose JSONs
├── *_checkpoint.npz                # 8 TRELLIS checkpoints
├── viz/
│   ├── all_masks_grid.png          # 4x2 grid of all mask overlays
│   ├── *_mask.png                  # Per-object mask overlay on target
│   └── *_mask_binary.png           # Per-object binary mask (white on black)
└── rotation_gifs/
    ├── *_y_rotation.gif            # 8 Y-axis turntable GIFs
    ├── *_x_rotation.gif            # 8 X-axis tumble GIFs
    └── *_frames/                   # Raw PNG frames (24 per axis per object)
```
