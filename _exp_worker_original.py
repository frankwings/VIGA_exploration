"""Minimal SAM3D worker for original Meta pipeline (no scene_image).

Calls Inference.__call__ without scene_image so the original pipeline runs:
- MoGe on per-object masked image (not full scene)
- Original SSI-normalized pointmap in run_post_optimization
- Original flat mask for ICP (no v9 mask growth)
- Occlusion check enabled

Usage:
    python _exp_worker_original.py --image IMG --mask MASK --config CFG --glb GLB --info INFO
"""
import argparse
import json
import os
import sys

# Set CONDA_PREFIX before importing inference.py — it reads CONDA_PREFIX at module-load time
if "CONDA_PREFIX" not in os.environ:
    conda_env = os.path.dirname(os.path.dirname(sys.executable))
    os.environ["CONDA_PREFIX"] = conda_env

import numpy as np
import torch


# Pure-PyTorch Transform3d (same as current worker — avoids pytorch3d install issues)
def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack([
        1 - two_s*(j*j+k*k), two_s*(i*j-k*r), two_s*(i*k+j*r),
        two_s*(i*j+k*r), 1 - two_s*(i*i+k*k), two_s*(j*k-i*r),
        two_s*(i*k-j*r), two_s*(j*k+i*r), 1 - two_s*(i*i+j*j),
    ], -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


class Transform3d:
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
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
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
        if points.dim() == 2:
            points = points.unsqueeze(0)
        N, P, _ = points.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
        points_h = torch.cat([points, ones], dim=-1)
        transformed = points_h @ self._matrix
        return transformed[..., :3]


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(ROOT, "utils", "third_party", "sam3d", "notebook"))
sys.path.append(os.path.join(ROOT, "utils", "third_party", "sam3d"))

from inference import Inference, load_image

# Pre-transform: TRELLIS Z-up → Y-up (matches get_mesh() in layout_post_optimization_utils.py)
R_zup_to_yup = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32)


def transform_mesh_vertices(vertices, rotation, translation, scale):
    if isinstance(vertices, np.ndarray):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    vertices = vertices.unsqueeze(0)
    vertices = vertices @ R_zup_to_yup.to(vertices.device)
    R_mat = quaternion_to_matrix(rotation.to(vertices.device))
    tfm = Transform3d(dtype=vertices.dtype, device=vertices.device)
    tfm = (tfm.scale(scale).rotate(R_mat)
              .translate(translation[0], translation[1], translation[2]))
    return tfm.transform_points(vertices)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--glb", required=True)
    p.add_argument("--info", required=False)
    args = p.parse_args()

    inference = Inference(args.config, compile=False)

    # Enable layout post-optimization (lazy import to save VRAM during TRELLIS)
    def _lazy_layout_post_opt(*a, **kw):
        from sam3d_objects.pipeline.inference_utils import layout_post_optimization
        return layout_post_optimization(*a, **kw)

    inference._pipeline.layout_post_optimization_method = _lazy_layout_post_opt

    image = load_image(args.image)
    mask = np.load(args.mask)
    mask = mask > 0

    # ORIGINAL: call without scene_image — original Inference.__call__ at af582ce
    # doesn't have scene_image parameter. This uses per-object masked MoGe + SSI path.
    output = inference(image, mask, seed=42)

    mesh = output["glb"]
    vertices = mesh.vertices

    S = output["scale"][0].cpu().float()
    T = output["translation"][0].cpu().float()
    R = output["rotation"].squeeze().cpu().float()

    vertices_transformed = transform_mesh_vertices(vertices, R, T, S)
    mesh.vertices = vertices_transformed.cpu().numpy().astype(np.float32)

    os.makedirs(os.path.dirname(args.glb), exist_ok=True)
    mesh.export(args.glb)

    # Build info JSON
    info = {
        "glb_path": args.glb,
        "translation": T.tolist(),
        "rotation": R.tolist(),
        "scale": S.item() if S.numel() == 1 else S.tolist(),
    }
    if "pointmap" in output:
        pm = output["pointmap"]
        if hasattr(pm, "shape"):
            info["pointmap_shape"] = list(pm.shape)
    if "final_iou" in output:
        info["final_iou"] = float(output["final_iou"])

    result = json.dumps(info, indent=2)
    if args.info:
        with open(args.info, "w", encoding="utf-8") as f:
            f.write(result)
    print(result)


if __name__ == "__main__":
    main()
