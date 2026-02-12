# VIGA Progress Report

*GenesisVIGA Pipeline Testing & SAM3D Environment Setup -- January 30, 2026 (Updated 18:24 PST)*

## Critical Finding: RTX 5080 + PyTorch Compatibility

### The Core Problem

RTX 5080 uses **CUDA compute capability sm_120** (Blackwell architecture). This is cutting-edge hardware with limited software support.

| PyTorch Version | Max CUDA Compute | RTX 5080 (sm_120) | Kaolin Support |
|----------------|------------------|-------------------|----------------|
| 2.5.1 | sm_90 (RTX 4090) | No | 0.17.0 |
| 2.8.0 | sm_120 (Blackwell) | Yes | 0.18.0 |
| 2.9.x | sm_120 | Yes | No wheels |
| 2.10.0 | sm_120 | Yes | No wheels |

### Solution Found

**Kaolin 0.18.0** (Released Aug 8, 2025) adds Blackwell support!

- Supports PyTorch up to 2.8.0
- Supports CUDA up to 12.9
- Adds sm_120 (Blackwell) compatibility
- Windows wheels available

## Current Status Overview

| Component | Status | Notes |
|-----------|--------|-------|
| sam3d_viga env | Partial | PyTorch 2.5.1 - GPU disabled for RTX 5080 |
| SAM3D-Objects Import | Working | Inference module loads successfully |
| Kaolin 0.17.0 | Installed | Works but no GPU (PyTorch 2.5.1) |
| PyTorch3D | Built | v0.7.9 compiled from source |
| spconv-cu121 | Working | Sparse convolutions operational |
| detectron2 | Built | v0.6 compiled from source |

## Environment Comparison

| Environment | PyTorch | CUDA | Kaolin | RTX 5080 GPU | Use Case |
|------------|---------|------|--------|--------------|----------|
| `sam3d` | 2.10.0 | 12.8 | None | Yes | General GPU work |
| `sam3d_viga` | 2.5.1 | 12.1 | 0.17.0 | CPU only | SAM3D (slow) |
| **`sam3d_kaolin` (NEW)** | 2.8.0 | 12.9 | 0.18.0 | **Yes** | **SAM3D + GPU** |

## Kaolin Version Research

### Kaolin 0.18.0 Release Notes (Aug 8, 2025)

**Key Changes:**

- Make kaolin compatible with numpy>=2.0
- Add conversion from polyscope camera to kaolin camera
- Implement camera trajectories interpolation
- **Add compatibility to Blackwell architecture**
- **Support pytorch up to 2.8.0 and cuda up to 12.9**

### Available Wheel Combinations

| Wheel Index | PyTorch | CUDA | Blackwell |
|-------------|---------|------|-----------|
| torch-2.5.1_cu121 | 2.5.1 | 12.1 | No |
| torch-2.7.0_cu128 | 2.7.0 | 12.8 | Yes |
| **torch-2.8.0_cu129** | **2.8.0** | **12.9** | **Yes** |

### PyTorch 2.9 Status

> **No Kaolin wheels for PyTorch 2.9 yet.** Latest Kaolin 0.18.0 only supports up to PyTorch 2.8.0. If you need PyTorch 2.9, you'd have to build Kaolin from source (risky).

## sam3d_viga Environment (Current)

### Successfully Installed Dependencies

- torch 2.5.1+cu121
- kaolin 0.17.0
- pytorch3d 0.7.9
- spconv-cu121
- detectron2 0.6
- lightning 2.6.1
- diffusers 0.36.0
- transformers 5.0.0
- open3d 0.18.0
- ultralytics 8.4.9
- moge 1.0.0
- gsplat 1.5.3
- timm 1.0.24
- xatlas 0.0.11

### Import Test Result

```
$ conda run -n sam3d_viga python -c "from inference import Inference, load_image; print('OK')"

Warp 1.11.0 initialized:
   CUDA Toolkit 12.9, Driver 13.1
   Devices:
     "cuda:0" : "NVIDIA GeForce RTX 5080" (16 GiB, sm_120, mempool enabled)
[SPARSE] Backend: spconv, Attention: sdpa
[ATTENTION] Using backend: sdpa
Inference module: OK
```

> **GPU Warning:**
> ```
> NVIDIA GeForce RTX 5080 with CUDA capability sm_120 is not compatible
> with the current PyTorch installation.
> Current PyTorch supports: sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90
> ```

## Solution: Create sam3d_kaolin Environment

### Recommended Setup

```bash
# Create new environment
conda create -n sam3d_kaolin python=3.11 -y
conda activate sam3d_kaolin

# Install PyTorch 2.8.0 with CUDA 12.9 (Blackwell support)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129

# Install Kaolin 0.18.0 with Blackwell support
pip install kaolin==0.18.0 \
    -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu129.html

# Install remaining SAM3D dependencies
pip install pytorch3d spconv-cu121 lightning omegaconf hydra-core \
    einops diffusers transformers accelerate open3d ultralytics \
    moge gsplat timm detectron2 ...
```

### Expected Result

| Component | Version | Status |
|-----------|---------|--------|
| PyTorch | 2.8.0+cu129 | GPU Works |
| Kaolin | 0.18.0 | Blackwell |
| RTX 5080 | sm_120 | Supported |

## Setup Timeline

- **17:54 PST** -- Created sam3d_viga Environment: Python 3.11 + PyTorch 2.5.1+cu121
- **17:56 PST** -- Kaolin 0.17.0 Installed: Pre-built wheel from NVIDIA S3
- **17:58 PST** -- PyTorch3D 0.7.9 Built: Compiled from source successfully
- **18:08 PST** -- detectron2 0.6 Built: Compiled from source with --no-build-isolation
- **18:10 PST** -- SAM3D-Objects Imports Successfully: All dependencies resolved, inference module loads
- **18:13 PST** -- GPU Not Compatible: RTX 5080 (sm_120) not supported by PyTorch 2.5.1
- **18:21 PST** -- Found Kaolin 0.18.0: Supports Blackwell + PyTorch 2.8.0 + CUDA 12.9
- **18:24 PST** -- Creating sam3d_kaolin Environment: PyTorch 2.8.0 + Kaolin 0.18.0 for full GPU support

## Next Steps

1. **Create sam3d_kaolin environment** with PyTorch 2.8.0+cu129
2. **Install Kaolin 0.18.0** from torch-2.8.0_cu129 wheel index
3. **Build/install remaining deps:** pytorch3d, detectron2, spconv
4. **Verify GPU acceleration** works with RTX 5080
5. **Test SAM3D inference** on green tea bottle image
6. **Run full VIGA pipeline** with 3D reconstruction

---

*Generated by Yuna | VIGA Progress Day 2*
*January 30, 2026 18:24 PST*
