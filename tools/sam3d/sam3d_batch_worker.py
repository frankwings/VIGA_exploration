"""SAM3D Batch Worker — processes multiple objects with a single model load.

Accepts a JSON manifest listing objects to reconstruct. Loads the TRELLIS
pipeline once and reuses it for all objects, saving ~21s per object that
would otherwise be spent on model loading.

Manifest format:
{
  "config": "path/to/pipeline.yaml",
  "scene_image": "path/to/scene.jpg",   // optional
  "objects": [
    {
      "name": "object_name",
      "image": "path/to/image.jpg",
      "mask": "path/to/mask.npy",
      "glb": "path/to/output.glb",
      "info": "path/to/info.json",       // optional
      "checkpoint": "path/to/ckpt.npz"   // optional
    },
    ...
  ]
}
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

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


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "utils", "third_party", "sam3d", "notebook"))
sys.path.append(os.path.join(ROOT, "utils", "third_party", "sam3d"))

from inference import Inference, load_image

if "CONDA_PREFIX" not in os.environ:
    python_bin = sys.executable
    conda_env = os.path.dirname(os.path.dirname(python_bin))
    os.environ["CONDA_PREFIX"] = conda_env

R_zup_to_yup = torch.tensor(
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32
)


def transform_mesh_vertices(vertices, rotation, translation, scale):
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
    return tfm.transform_points(vertices)[0]


def reload_pipeline_to_gpu(pipeline):
    """Move all pipeline components back to GPU after post-opt offloaded them to CPU.

    The pipeline's run() method offloads self.models, condition_embedders, and
    depth_model to CPU to free VRAM for layout post-optimization (which uses
    pytorch3d). For batch processing, we need to reload everything to GPU
    before processing the next object.
    """
    device = pipeline.device
    t0 = time.time()

    # Reload main models (ModuleDict: ss_generator, slat_generator, decoders, etc.)
    pipeline.models.to(device)

    # Reload condition embedders
    for emb_dict in pipeline.condition_embedders.values():
        if hasattr(emb_dict, 'embedder_list'):
            for emb, _ in emb_dict.embedder_list:
                if hasattr(emb, 'to'):
                    emb.to(device)

    # Reload depth model (MoGe)
    if hasattr(pipeline, 'depth_model') and pipeline.depth_model is not None:
        if hasattr(pipeline.depth_model, 'model'):
            pipeline.depth_model.model.to(device)

    torch.cuda.empty_cache()
    elapsed = time.time() - t0
    print(f"[BATCH] Models reloaded to GPU in {elapsed:.1f}s", flush=True)


def process_single_object(inference, obj, scene_image, save_checkpoint):
    """Process one object using the already-loaded inference pipeline."""
    name = obj["name"]
    t0 = time.time()

    image = load_image(obj["image"])
    mask = np.load(obj["mask"])
    mask = mask > 0

    # Capture checkpoint data via wrapper
    _ckpt = {}

    original_post_opt = inference._pipeline.layout_post_optimization_method

    def _lazy_layout_post_opt(*a, **kw):
        if save_checkpoint:
            def _np(x):
                if hasattr(x, "cpu"):
                    return x.cpu().detach().float().numpy()
                return np.array(x, dtype=np.float32)
            _ckpt["rotation"]    = _np(a[1])
            _ckpt["translation"] = _np(a[2])
            _ckpt["scale"]       = _np(a[3])
            _ckpt["mask"]        = _np(a[4])
            _ckpt["point_map"]   = _np(a[5])
            _ckpt["intrinsics"]  = _np(a[6])
        from sam3d_objects.pipeline.inference_utils import layout_post_optimization
        return layout_post_optimization(*a, **kw)

    inference._pipeline.layout_post_optimization_method = _lazy_layout_post_opt

    try:
        output = inference(image, mask, seed=42, scene_image=scene_image)
    finally:
        inference._pipeline.layout_post_optimization_method = original_post_opt
        # Reload models to GPU for the next object (post-opt offloads to CPU)
        reload_pipeline_to_gpu(inference._pipeline)

    mesh = output["glb"]
    vertices = mesh.vertices

    # Save checkpoint
    ckpt_path = obj.get("checkpoint")
    if ckpt_path and _ckpt:
        ckpt_dir = os.path.dirname(ckpt_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        save_dict = {
            "vertices": np.array(vertices, dtype=np.float32),
            "faces": np.array(mesh.faces, dtype=np.int32),
            **_ckpt,
        }
        try:
            vc = mesh.visual.vertex_colors
            if vc is not None:
                save_dict["vertex_colors"] = np.array(vc, dtype=np.uint8)
        except Exception:
            pass
        np.savez(ckpt_path, **save_dict)
        print(f"[checkpoint] Saved: {ckpt_path}", flush=True)

    # Export canonical GLB (raw Z-up frame, with textures) before transforming
    canonical_glb_path = obj.get("canonical_glb")
    if canonical_glb_path:
        os.makedirs(os.path.dirname(canonical_glb_path), exist_ok=True)
        mesh.export(canonical_glb_path)
        print(f"[canonical] Saved: {canonical_glb_path}", flush=True)

    # Transform vertices
    S = output["scale"][0].cpu().float()
    T = output["translation"][0].cpu().float()
    R = output["rotation"].squeeze().cpu().float()
    vertices_transformed = transform_mesh_vertices(vertices, R, T, S)
    mesh.vertices = vertices_transformed.cpu().numpy().astype(np.float32)

    # Export GLB
    glb_path = obj["glb"]
    os.makedirs(os.path.dirname(glb_path), exist_ok=True)
    mesh.export(glb_path)

    # Build info dict
    intrinsics_data = {}
    if "intrinsics" in output:
        intrinsics = output["intrinsics"]
        if hasattr(intrinsics, 'cpu'):
            intrinsics = intrinsics.cpu().float()
        intrinsics_data["intrinsics"] = intrinsics.tolist()
    if "pointmap" in output:
        pm = output["pointmap"]
        if hasattr(pm, 'shape'):
            intrinsics_data["pointmap_shape"] = list(pm.shape)

    iou_data = {}
    if "iou" in output:
        iou_val = output["iou"]
        iou_data["iou"] = float(iou_val) if not isinstance(iou_val, float) else iou_val

    info = {
        "glb_path": glb_path,
        "translation": T.tolist(),
        "rotation": R.tolist(),
        "scale": S.tolist(),
        **intrinsics_data,
        **iou_data,
    }

    # Write info JSON
    info_path = obj.get("info")
    if info_path:
        os.makedirs(os.path.dirname(info_path), exist_ok=True)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)

    elapsed = time.time() - t0
    iou_str = f"IoU={iou_data.get('iou', 'N/A')}"
    glb_mb = os.path.getsize(glb_path) / (1024 * 1024)
    print(f"[BATCH] {name}: OK ({elapsed:.1f}s, {glb_mb:.1f}MB, {iou_str})", flush=True)

    return info


def main():
    p = argparse.ArgumentParser(description="SAM3D batch worker — single model load, multiple objects")
    p.add_argument("--manifest", required=True, help="Path to JSON manifest file")
    args = p.parse_args()

    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    config_path = manifest["config"]
    objects = manifest["objects"]
    scene_image_path = manifest.get("scene_image")

    # Load scene image once
    scene_image = None
    if scene_image_path:
        scene_image = load_image(scene_image_path)

    # Load model ONCE
    print(f"[BATCH] Loading TRELLIS pipeline from {config_path}...", flush=True)
    t_load = time.time()
    inference = Inference(config_path, compile=False)
    load_time = time.time() - t_load
    print(f"[BATCH] Pipeline loaded in {load_time:.1f}s", flush=True)

    # Process each object
    results = {}
    total_start = time.time()
    for i, obj in enumerate(objects):
        name = obj["name"]
        glb_path = obj["glb"]
        info_path = obj.get("info")

        # Skip if already completed
        if os.path.exists(glb_path) and info_path and os.path.exists(info_path):
            print(f"[BATCH] {name}: already exists, skipping", flush=True)
            with open(info_path, 'r') as f:
                results[name] = json.load(f)
            continue

        print(f"[BATCH] Processing {name} ({i+1}/{len(objects)})...", flush=True)
        has_checkpoint = "checkpoint" in obj and obj["checkpoint"]
        try:
            info = process_single_object(inference, obj, scene_image, has_checkpoint)
            results[name] = info
        except Exception as e:
            print(f"[BATCH] {name}: FAILED - {e}", flush=True)
            results[name] = None

    total_time = time.time() - total_start
    success = sum(1 for v in results.values() if v is not None)
    print(f"\n[BATCH] Completed: {success}/{len(objects)} in {total_time:.1f}s "
          f"(model load: {load_time:.1f}s, inference: {total_time:.1f}s)", flush=True)

    # Write batch results to manifest dir
    manifest_dir = os.path.dirname(args.manifest)
    if manifest_dir:
        summary_path = os.path.join(manifest_dir, "batch_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                "model_load_time": load_time,
                "total_inference_time": total_time,
                "objects": {name: {"success": v is not None, "iou": v.get("iou") if v else None}
                           for name, v in results.items()},
            }, f, indent=2)


if __name__ == "__main__":
    main()
