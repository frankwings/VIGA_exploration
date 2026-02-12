# SAM3D Windows Test Report

*Single-image to 3D Mesh Generation*

**Status: Test Passed**

## Environment Configuration

| Configuration Item | Value |
|-------------------|-------|
| Operating System | Windows 11 |
| Conda Environment | sam3d_py311 |
| Python | 3.11.14 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5080 (16GB) |
| Open3D | 0.19.0 |
| spconv | Backend: spconv, Attention: sdpa |

## Test Input

| Parameter | Value |
|-----------|-------|
| Image | green_tea_bottle.png |
| Mask | green_tea_bottle.npy (1024 x 771) |
| Seed | 42 |
| compile | False |

## Execution Timeline

- **03:15:09** -- GPU detected: NVIDIA GeForce RTX 5080
- **03:15:18** -- Started loading model weights
- **03:15:40** -- Model weights loaded (~22 seconds)
- **03:15:43** -- Started Sparse Structure sampling (25 steps, cfg=7)
- **03:15:57** -- Sparse Structure completed, downsampled 5623 -> 4220 points
- **03:15:57** -- Started Sparse Latent sampling (25 steps, cfg=1)
- **03:16:21** -- Started decoding Sparse Latent
- **03:18:13** -- Post-processing completed, GLB export successful

## Performance Statistics

### Stage Timing

| Stage | Duration |
|-------|----------|
| Model Loading | ~22 seconds |
| Sparse Structure | ~14 seconds |
| Sparse Latent | ~24 seconds |
| Decoding + Export | ~112 seconds |

### Summary

| Item | Value |
|------|-------|
| Total Inference Time | ~3 minutes |
| Output Format | GLB |
| Mesh Post-processing | False |
| Texture Baking | False |

## Input vs Output Comparison

### Input Image

![Input](test_results_images/01_greentea_input.jpg)

### SAM3D 3D Output

![Output](test_results_images/02_greentea_output.png)

## Output Details

```
Success!
Output keys: dict_keys([
    '6drotation_normalized', 'scale', 'shape', 'translation',
    'translation_scale', 'coords_original', 'coords',
    'downsample_factor', 'rotation', 'mesh', 'gaussian',
    'glb', 'gs', 'pointmap', 'pointmap_colors'
])

Saved GLB to: D:\Projects\ProjectGenesis\GenesisVIGA\output\sam3d_test\green_tea_bottle.glb
```

## Comparison with WSL

| Comparison Item | Windows (Pass) | WSL (Fail) |
|----------------|----------------|------------|
| PyTorch | 2.8.0+cu128 (stable) | 2.11.0.dev (nightly) |
| Mask Quality | Clean | Has noise/artifacts |
| GLB Output | Pass | Only .ply |
| OOM Issues | None | Frequent OOM |

## Conclusions

- Windows + `sam3d_py311` environment runs SAM3D completely normally
- RTX 5080 + PyTorch 2.8.0+cu128 compatibility is good
- WSL issues may be related to nightly PyTorch or memory limitations
- Recommendation: WSL should also use stable PyTorch 2.8.0 or 2.9.1

---

*Test date: 2026-02-03 03:11 - 03:18 PST*
*-- Created by Yuna*
