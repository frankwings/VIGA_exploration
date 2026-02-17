"""SAM3D Worker for 3D Reconstruction from Masked Images.

This script uses SAM3D to reconstruct a 3D mesh from an image and its
segmentation mask, then transforms the mesh vertices to world coordinates
and exports as GLB format.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Pure PyTorch replacements for pytorch3d functions
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.
    
    Args:
        quaternions: quaternions with real part first, shape (..., 4).
    
    Returns:
        Rotation matrices, shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


class Transform3d:
    """Simple Transform3d replacement using pure PyTorch."""
    
    def __init__(self, dtype=torch.float32, device="cpu"):
        self.dtype = dtype
        self.device = device
        self._matrix = torch.eye(4, dtype=dtype, device=device)
    
    def scale(self, scale):
        if isinstance(scale, (int, float)):
            scale = torch.tensor([scale, scale, scale], dtype=self.dtype, device=self.device)
        elif isinstance(scale, torch.Tensor) and scale.numel() == 1:
            scale = scale.expand(3)
        S = torch.eye(4, dtype=self.dtype, device=self.device)
        S[0, 0] = scale[0] if len(scale.shape) == 1 else scale
        S[1, 1] = scale[1] if len(scale.shape) == 1 else scale
        S[2, 2] = scale[2] if len(scale.shape) == 1 else scale
        self._matrix = self._matrix @ S
        return self
    
    def rotate(self, R):
        if R.shape == (3, 3):
            R4 = torch.eye(4, dtype=self.dtype, device=self.device)
            R4[:3, :3] = R
        else:
            R4 = R
        self._matrix = self._matrix @ R4
        return self
    
    def translate(self, x, y, z):
        T = torch.eye(4, dtype=self.dtype, device=self.device)
        T[3, 0] = x
        T[3, 1] = y
        T[3, 2] = z
        self._matrix = self._matrix @ T
        return self
    
    def transform_points(self, points):
        # points: (B, N, 3) or (N, 3)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        B, N, _ = points.shape
        ones = torch.ones(B, N, 1, dtype=self.dtype, device=self.device)
        points_h = torch.cat([points, ones], dim=-1)  # (B, N, 4)
        transformed = points_h @ self._matrix  # (B, N, 4)
        return transformed[..., :3]  # (B, N, 3)

ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "utils", "third_party", "sam3d", "notebook"))
sys.path.append(os.path.join(ROOT, "utils", "third_party", "sam3d"))

from inference import Inference, load_image

if "CONDA_PREFIX" not in os.environ:
    python_bin = sys.executable
    conda_env = os.path.dirname(os.path.dirname(python_bin))
    os.environ["CONDA_PREFIX"] = conda_env

# Pre-transform: convert TRELLIS model space (Z-up) to Y-up before applying S@R+T.
# This matches layout_post_optimization_utils.get_mesh() line 116:
#   mesh_vertices = mesh_vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
# Which is [[1,0,0],[0,0,1],[0,-1,0]], mapping [x,y,z] -> [x, z, -y].
R_zup_to_yup: torch.Tensor = torch.tensor(
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32
)


def transform_mesh_vertices(
    vertices: np.ndarray,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """Transform mesh vertices from TRELLIS model space to PyTorch3D camera space.

    Pipeline:
    1. Pre-transform: convert TRELLIS Z-up to Y-up (matches get_mesh in
       layout_post_optimization_utils.py)
    2. Apply scale, rotation, and translation predicted by the SAM3D pipeline.
       These S, R, T map from Y-up model space to PyTorch3D camera space.

    The result is in PyTorch3D camera convention (X-left, Y-up, Z-forward).

    Args:
        vertices: Mesh vertices as numpy array (N, 3).
        rotation: Quaternion rotation tensor.
        translation: Translation vector tensor.
        scale: Scale factor tensor.

    Returns:
        Transformed vertices as torch tensor (N, 3).
    """
    if isinstance(vertices, np.ndarray):
        vertices = torch.tensor(vertices, dtype=torch.float32)

    vertices = vertices.unsqueeze(0)  # Add batch dimension [1, N, 3]
    vertices = vertices @ R_zup_to_yup.to(vertices.device)
    R_mat = quaternion_to_matrix(rotation.to(vertices.device))
    tfm = Transform3d(dtype=vertices.dtype, device=vertices.device)
    tfm = (
        tfm.scale(scale)
           .rotate(R_mat)
           .translate(translation[0], translation[1], translation[2])
    )
    vertices_world = tfm.transform_points(vertices)

    return vertices_world[0]  # Remove batch dimension


def main() -> None:
    """Run SAM3D reconstruction on a masked image and export as GLB."""
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--mask", required=True, help="Path to mask npy file")
    p.add_argument("--config", required=True, help="Path to SAM3D config file")
    p.add_argument("--glb", required=True, help="Path for output GLB file")
    p.add_argument("--info", required=False, help="Path to save JSON output (instead of stdout)")
    p.add_argument("--scene-image", required=False, help="Path to full scene image for MoGe depth (avoids NaN on dark per-object images)")
    args = p.parse_args()

    inference = Inference(args.config, compile=False)

    # Enable layout post-optimization with LAZY import to avoid VRAM conflict.
    # pytorch3d import (required by post-opt) allocates CUDA memory; deferring
    # the import until post-opt runs keeps VRAM free for TRELLIS model inference.
    def _lazy_layout_post_opt(*a, **kw):
        from sam3d_objects.pipeline.inference_utils import layout_post_optimization
        return layout_post_optimization(*a, **kw)

    inference._pipeline.layout_post_optimization_method = _lazy_layout_post_opt

    image = load_image(args.image)
    # Load mask from npy file
    mask = np.load(args.mask)
    mask = mask > 0

    # Load full scene image for MoGe depth if provided (avoids NaN on
    # mostly-black per-object images where MoGe can't estimate depth).
    scene_image = None
    if args.scene_image:
        scene_image = load_image(args.scene_image)

    output = inference(image, mask, seed=42, scene_image=scene_image)

    mesh = output["glb"]
    vertices = mesh.vertices

    S = output["scale"][0].cpu().float()
    T = output["translation"][0].cpu().float()
    R = output["rotation"].squeeze().cpu().float()

    vertices_transformed = transform_mesh_vertices(vertices, R, T, S)
    mesh.vertices = vertices_transformed.cpu().numpy().astype(np.float32)

    os.makedirs(os.path.dirname(args.glb), exist_ok=True)
    mesh.export(args.glb)

    # Extract camera intrinsics if available (from MoGe pointmap estimation)
    intrinsics_data = {}
    if "intrinsics" in output:
        intrinsics = output["intrinsics"]
        if hasattr(intrinsics, 'cpu'):
            intrinsics = intrinsics.cpu().float()
        intrinsics_data["intrinsics"] = intrinsics.tolist()
    if "pointmap" in output:
        # Pointmap shape gives image dimensions used by MoGe
        pm = output["pointmap"]
        if hasattr(pm, 'shape'):
            intrinsics_data["pointmap_shape"] = list(pm.shape)

    # Extract layout post-optimization IoU if available
    iou_data = {}
    if "iou" in output:
        iou_val = output["iou"]
        iou_data["iou"] = float(iou_val) if not isinstance(iou_val, float) else iou_val
        print(f"Layout post-opt IoU: {iou_data['iou']:.4f}", flush=True)

    # Prepare output data
    translation_data = {
        "glb_path": args.glb,
        "translation": T.tolist(),
        "rotation": R.tolist(),
        "scale": S.tolist(),
        **intrinsics_data,
        **iou_data,
    }

    # Write to file if --info provided, otherwise print to stdout for backward compatibility
    if args.info:
        os.makedirs(os.path.dirname(args.info), exist_ok=True)
        with open(args.info, 'w') as f:
            json.dump(translation_data, f, indent=2)
    else:
        print(json.dumps(translation_data))


if __name__ == "__main__":
    main()
