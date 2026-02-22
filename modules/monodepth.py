"""Module 3: Monocular Depth Estimation (MoGe).

Runs MoGe ViT-L on the full scene image to produce a dense pointmap and
camera intrinsics in PyTorch3D camera space.

Usage:
    python modules/monodepth.py --image <target.jpg> --output-dir <output/monodepth/>

Output:
    monodepth_manifest.json       — manifest with intrinsics + paths
    pointmap.npz                  — pointmap (3, H, W) float32
    viz/depth_map.png             — colorized depth visualization
    viz/normal_map.png            — surface normals from pointmap gradients

Conda env: sam3d_py311 (Python 3.11, MoGe, PyTorch3D)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAM3D_ROOT = os.path.join(ROOT, "utils", "third_party", "sam3d")
sys.path.insert(0, SAM3D_ROOT)

# Skip sam3d_objects.init module (not needed, causes ImportError)
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

if "CONDA_PREFIX" not in os.environ:
    python_bin = sys.executable
    conda_env = os.path.dirname(os.path.dirname(python_bin))
    os.environ["CONDA_PREFIX"] = conda_env


# ---------------------------------------------------------------------------
# MoGe depth estimation (extracted from pose_align_worker.py)
# ---------------------------------------------------------------------------

def load_moge_model(device="cuda"):
    """Load MoGe depth estimation model."""
    from moge.model.v1 import MoGeModel
    from sam3d_objects.pipeline.depth_models.moge import MoGe

    print("[MONODEPTH] Loading MoGe model: Ruicheng/moge-vitl...", flush=True)
    t0 = time.time()
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl")
    depth_model = MoGe(model=moge_model, device=device)
    print(f"[MONODEPTH] MoGe loaded in {time.time() - t0:.1f}s", flush=True)
    return depth_model


def compute_pointmap(depth_model, image_np, device="cuda", dtype=torch.float16):
    """Run MoGe depth estimation and convert to PyTorch3D camera space.

    Returns dict with:
        pointmap: (3, H, W) float32 tensor in PyTorch3D camera space
        intrinsics: (3, 3) tensor — normalized camera intrinsics
        pts_color: (3, H, W) float32 tensor — image colors
    """
    from pytorch3d.transforms import Transform3d
    from pytorch3d.renderer import look_at_view_transform
    from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap

    # Convert image to float [0, 1] and to tensor (3, H, W)
    if image_np.dtype == np.uint8:
        image_float = image_np.astype(np.float32) / 255.0
    else:
        image_float = image_np.astype(np.float32)
    if image_float.ndim == 2:
        image_float = np.stack([image_float] * 3, axis=-1)
    if image_float.shape[-1] == 4:
        image_float = image_float[..., :3]
    image_tensor = torch.from_numpy(image_float).permute(2, 0, 1).contiguous()  # (3, H, W)

    # Run MoGe
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = depth_model(image_tensor)
    pointmaps = output["pointmaps"]  # (H, W, 3) in R3 camera space

    # Convert R3 camera space -> PyTorch3D camera space
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )
    cam_transform = Transform3d().rotate(r3_to_p3d_R).to(device)
    points_tensor = cam_transform.transform_points(pointmaps)

    # Convert to (3, H, W)
    points_tensor = points_tensor.permute(2, 0, 1)

    # Infer intrinsics
    intrinsics = output.get("intrinsics", None)
    if intrinsics is None:
        intrinsics_result = infer_intrinsics_from_pointmap(
            points_tensor.permute(1, 2, 0), device=device
        )
        intrinsics = intrinsics_result["intrinsics"]

    return {
        "pointmap": points_tensor,
        "intrinsics": intrinsics,
        "pts_color": image_tensor,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_depth(pointmap_3hw: np.ndarray, output_path: str) -> None:
    """Create a colorized depth map from the pointmap Z channel."""
    from PIL import Image

    # Z channel = depth (index 2 of (3, H, W))
    depth = pointmap_3hw[2]  # (H, W)

    # Handle NaN
    valid = ~np.isnan(depth)
    if not np.any(valid):
        print("[MONODEPTH] WARNING: all depth values are NaN")
        return

    d_min = np.nanmin(depth)
    d_max = np.nanmax(depth)
    if d_max - d_min < 1e-6:
        d_max = d_min + 1.0

    # Normalize to 0-1
    normalized = (depth - d_min) / (d_max - d_min)
    normalized = np.clip(normalized, 0, 1)
    normalized[~valid] = 0

    # Apply viridis-like colormap manually (avoid matplotlib dependency)
    # Simple blue-green-yellow gradient
    r = np.clip(normalized * 3 - 1, 0, 1)
    g = np.clip(1 - np.abs(normalized * 3 - 1.5) * 2, 0, 1)
    b = np.clip(1 - normalized * 2, 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = 0
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    Image.fromarray(rgb_uint8).save(output_path)
    print(f"[MONODEPTH] Depth map saved: {output_path}")


def visualize_normals(pointmap_3hw: np.ndarray, output_path: str) -> None:
    """Compute surface normals from pointmap gradients and save as RGB image."""
    from PIL import Image

    # (3, H, W) -> (H, W, 3)
    pm = pointmap_3hw.transpose(1, 2, 0)

    # Compute gradients
    dy = np.gradient(pm, axis=0)
    dx = np.gradient(pm, axis=1)

    # Cross product -> normal
    normal = np.cross(dx, dy)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    normal = normal / norm

    # Handle NaN
    valid = ~np.isnan(normal).any(axis=-1)

    # Map [-1, 1] to [0, 255]
    normal_vis = (normal * 0.5 + 0.5)
    normal_vis = np.clip(normal_vis, 0, 1)
    normal_vis[~valid] = 0
    normal_uint8 = (normal_vis * 255).astype(np.uint8)

    Image.fromarray(normal_uint8).save(output_path)
    print(f"[MONODEPTH] Normal map saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Module 3: MoGe Monocular Depth")
    parser.add_argument("--image", required=True, help="Path to input scene image")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    image_path = os.path.abspath(args.image)
    print(f"[MONODEPTH] Image: {image_path}")
    print(f"[MONODEPTH] Output: {output_dir}")

    # Load image
    from PIL import Image
    scene_img = np.array(Image.open(image_path).convert("RGB"))
    h, w = scene_img.shape[:2]
    print(f"[MONODEPTH] Image size: {w}x{h}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load MoGe model
    depth_model = load_moge_model(device=device)

    # Compute pointmap
    print("[MONODEPTH] Computing pointmap...", flush=True)
    t0 = time.time()
    result = compute_pointmap(depth_model, scene_img, device=device)
    elapsed = time.time() - t0
    print(f"[MONODEPTH] Pointmap computed in {elapsed:.1f}s")

    pointmap = result["pointmap"]  # (3, H, W) on GPU
    intrinsics = result["intrinsics"]  # (3, 3)

    # Save pointmap
    pm_np = pointmap.cpu().float().numpy()
    pm_h, pm_w = pm_np.shape[1], pm_np.shape[2]
    pointmap_path = os.path.join(output_dir, "pointmap.npz")
    np.savez(pointmap_path, pointmap=pm_np)
    print(f"[MONODEPTH] Pointmap saved: {pointmap_path} shape=({pm_np.shape})")

    # Convert intrinsics to list
    if hasattr(intrinsics, "cpu"):
        intrinsics_list = intrinsics.cpu().float().tolist()
    else:
        intrinsics_list = intrinsics.tolist() if hasattr(intrinsics, "tolist") else intrinsics

    # Visualizations
    visualize_depth(pm_np, os.path.join(viz_dir, "depth_map.png"))
    visualize_normals(pm_np, os.path.join(viz_dir, "normal_map.png"))

    # Free GPU
    depth_model.model.cpu()
    torch.cuda.empty_cache()

    # Write manifest
    manifest = {
        "image": image_path,
        "pointmap_path": pointmap_path,
        "pointmap_shape": [pm_h, pm_w],
        "intrinsics": intrinsics_list,
        "intrinsics_type": "normalized",
    }
    manifest_path = os.path.join(output_dir, "monodepth_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[MONODEPTH] Done.")
    print(f"[MONODEPTH] Manifest: {manifest_path}")
    print(f"[MONODEPTH] Intrinsics: {intrinsics_list}")
    print(f"[MONODEPTH] Pointmap shape: ({3}, {pm_h}, {pm_w})")


if __name__ == "__main__":
    main()
