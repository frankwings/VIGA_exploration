# VIGA SAM3D Workflow

**Complete call record from image to 3D mesh**

---

## Pipeline Overview

```
Input Image + Mask (.npy)  -->  SAM3D Inference  -->  Coordinate Transform (VIGA)  -->  GLB Output + Transform Info
```

---

## Environment Configuration

| Configuration Item | Value |
|---|---|
| Operating System | Windows 11 |
| Conda Environment | `sam3d_py311` |
| Python | 3.11.14 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5080 (16GB) |

---

## Command Used

```powershell
# Set PYTHONPATH
$env:PYTHONPATH = "D:\Projects\ProjectGenesis\GenesisVIGA"

# Run VIGA SAM3D Worker
conda run -n sam3d_py311 python "tools\sam3d\sam3d_worker.py" `
    --image "output\test_sam\green_tea_bottle.png" `
    --mask  "output\test_sam\green_tea_bottle.npy" `
    --config "utils\third_party\sam3d\checkpoints\hf\checkpoints\pipeline.yaml" `
    --glb   "output\viga_test\green_tea_bottle_viga.glb"
```

---

## Input and Output

### Input Image (with Mask)

![Input Image](test_results_images/01_greentea_input.jpg)

green_tea_bottle.png + green_tea_bottle.npy (1024x771)

### Blender Render Result

![Render Output](test_results_images/green_tea_bottle_viga_render.png)

The rendered mesh is a flat "board"

---

## Output JSON Data

```json
{
    "glb_path": "output/viga_test/green_tea_bottle_viga.glb",
    "translation": [0.066, -0.440, 1.458],
    "rotation": [0.00007, -0.019, -0.998, -0.062],  // quaternion
    "scale": [2.31, 2.31, 2.31]
}
```

VIGA outputs world coordinate system transform information, used to position objects in the Blender scene.

---

## Mesh Structure Analysis

| Property | VIGA Output | Original SAM3D |
|---|---|---|
| Vertex Count | 118,684 | 118,676 |
| Face Count | 237,360 | 237,344 |
| Dimensions (X, Y, Z) | 2.35, 1.10, 0.65 | 1.00, 0.23, 0.42 |

> **Warning: Mesh is flat**
>
> Because `with_mesh_postprocess=False`, SAM3D outputs a "billboard"-style flat mesh, not a true 3D bottle shape.
>
> The Y and Z dimensions are significantly smaller than X, proving this is a nearly planar structure.

---

## Why is the Mesh Flat?

In `inference.py` lines 144-145, mesh post-processing is hardcoded as disabled:

```python
return self._pipeline.run(
    image,
    None,
    seed,
    stage1_only=False,
    with_mesh_postprocess=False,   # <-- Mesh post-processing disabled
    with_texture_baking=False,      # <-- Texture baking disabled
    with_layout_postprocess=True,
    use_vertex_color=True,
    ...
)
```

**Reason:** The `utils3d` library API is incompatible with the current version, and certain functions from `nvdiffrast` are missing.

---

## Performance Statistics

| Stage | Duration |
|---|---|
| Model Loading | ~22 seconds |
| Sparse Structure Sampling | ~12 seconds |
| Sparse Latent Sampling | ~20 seconds |
| Decoding + Export | ~135 seconds |
| **Total** | **~3.5 minutes** |

---

## File Structure

```
D:\Projects\ProjectGenesis\GenesisVIGA\
├── tools\sam3d\
│   ├── sam3d_worker.py      # VIGA's SAM3D invocation script
│   ├── sam3_worker.py       # SAM3 segmentation script
│   └── bridge.py            # MCP service bridge
├── utils\third_party\sam3d\
│   ├── notebook\inference.py # SAM3D wrapper class
│   └── checkpoints\hf\...    # Model weights
└── output\viga_test\
    └── green_tea_bottle_viga.glb # Output GLB
```

---

## Next Steps

- Fix `utils3d` API compatibility issues
- Enable `with_mesh_postprocess=True`
- Test true 3D mesh output
- Or use a Gaussian Splatting dedicated renderer to view results

---

Test date: 2026-02-03 03:28 - 03:40 PST

*-- Created by Yuna*
