"""SAM3D Worker v2 (TRELLIS.2 Adapter)
"""

print("Worker V2 Starting...", flush=True)
import argparse
import json
import os
import sys

import numpy as np
import torch
import traceback

# Import the Adapter
# Assuming this script is running from GenesisVIGA/tools/sam3d/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add TRELLIS.2 source path
sys.path.append(r"D:\Projects\ProjectGenesis\TRELLIS.2")

try:
    from adapters.trellis2_adapter import Trellis2Inference as Inference, load_image
except ImportError as e:
    print(f"Failed to import adapter: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"Error during import of adapter: {e}")
    traceback.print_exc()
    sys.exit(1)

# Pure PyTorch replacements for pytorch3d functions
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
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
        if points.dim() == 2:
            points = points.unsqueeze(0)
        B, N, _ = points.shape
        ones = torch.ones(B, N, 1, dtype=self.dtype, device=self.device)
        points_h = torch.cat([points, ones], dim=-1)
        transformed = points_h @ self._matrix
        return transformed[..., :3]

R_zup_to_yup: torch.Tensor = torch.tensor(
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32
)

def transform_mesh_vertices(
    vertices: np.ndarray,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    if isinstance(vertices, np.ndarray):
        vertices = torch.tensor(vertices, dtype=torch.float32)

    vertices = vertices.unsqueeze(0)
    vertices = vertices @ R_zup_to_yup.to(vertices.device)
    R_mat = quaternion_to_matrix(rotation.to(vertices.device))
    tfm = Transform3d(dtype=vertices.dtype, device=vertices.device)
    tfm = (
        tfm.scale(scale)
           .rotate(R_mat)
           .translate(translation[0], translation[1], translation[2])
    )
    vertices_world = tfm.transform_points(vertices)
    return vertices_world[0]

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--mask", required=True, help="Path to mask npy file")
    p.add_argument("--config", required=False, default="microsoft/TRELLIS.2-4B", help="Model Name or Config")
    p.add_argument("--glb", required=True, help="Path for output GLB file")
    p.add_argument("--info", required=False, help="Path to save JSON output (instead of stdout)")
    args = p.parse_args()

    # Use args.config as model name
    inference = Inference(model_name=args.config)
    image = load_image(args.image)
    mask = np.load(args.mask)
    mask = mask > 0
    
    output = inference(image, mask, seed=42)

    mesh = output["glb"]
    vertices = mesh.vertices

    S = output["scale"][0].cpu().float()
    T = output["translation"][0].cpu().float()
    R = output["rotation"].squeeze().cpu().float()

    # Apply transform (even if identity, we do Z-up -> Y-up conversion)
    vertices_transformed = transform_mesh_vertices(vertices, R, T, S)
    mesh.vertices = vertices_transformed.cpu().numpy().astype(np.float32)

    os.makedirs(os.path.dirname(args.glb), exist_ok=True)
    mesh.export(args.glb)

    # Simplified translation data (no intrinsics/pointmap from TRELLIS 2)
    translation_data = {
        "glb_path": args.glb,
        "translation": T.tolist(),
        "rotation": R.tolist(),
        "scale": S.tolist(),
    }

    if args.info:
        os.makedirs(os.path.dirname(args.info), exist_ok=True)
        with open(args.info, 'w') as f:
            json.dump(translation_data, f, indent=2)
    else:
        print(json.dumps(translation_data))

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("ERROR IN WORKER:")
        traceback.print_exc()
        sys.exit(1)
