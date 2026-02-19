"""Replay layout_post_optimization from a TRELLIS checkpoint.

Re-runs only the ICP + Adam alignment stage without re-running the expensive
TRELLIS reconstruction (~10 min → ~30 s).

Usage:
    # 1. First generate a checkpoint (add --checkpoint to your sam3d_worker.py call):
    $PYTHON_SAM3D tools/sam3d/sam3d_worker.py \\
        --image <img> --mask <mask.npy> --config <config> \\
        --glb output/original/object.glb \\
        --scene-image data/static_scene/dining/target.jpg \\
        --checkpoint output/checkpoints/object_ckpt.npz

    # 2. Then replay with different post-opt settings (no GPU wait for TRELLIS):
    $PYTHON_SAM3D replay_post_opt.py \\
        --checkpoint output/checkpoints/object_ckpt.npz \\
        --output-dir output/my_experiment \\
        [--name object_name]  \\
        [--no-icp] [--no-adam]

Environment: sam3d_py311 (same as sam3d_worker.py)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, "utils", "third_party", "sam3d", "notebook"))
sys.path.append(os.path.join(ROOT, "utils", "third_party", "sam3d"))

if "CONDA_PREFIX" not in os.environ:
    python_bin = sys.executable
    conda_env = os.path.dirname(os.path.dirname(python_bin))
    os.environ["CONDA_PREFIX"] = conda_env


# R_zup_to_yup: matches sam3d_worker.py and layout_post_optimization_utils.get_mesh().
# Maps TRELLIS Z-up model space → PyTorch3D Y-up camera space (pre-transform step).
_R_ZUP_TO_YUP = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32)


def _transform_vertices(vertices_np, R_quat, T, S, device):
    """Apply TRELLIS → PyTorch3D-camera-space transform (mirrors sam3d_worker.py)."""
    from pytorch3d.transforms import quaternion_to_matrix

    R_zup = _R_ZUP_TO_YUP.to(device)
    verts = torch.tensor(vertices_np, dtype=torch.float32, device=device)

    verts = verts @ R_zup          # Z-up → Y-up (row-vector convention)
    verts = verts * S              # non-isotropic scale
    R_mat = quaternion_to_matrix(R_quat)  # (3, 3) — row-vector convention
    verts = verts @ R_mat          # rotate
    verts = verts + T              # translate

    return verts  # (N, 3)


def _build_mesh(ckpt_data) -> trimesh.Trimesh:
    """Reconstruct trimesh from checkpoint arrays."""
    vertices = ckpt_data["vertices"]  # (N, 3) float32
    faces    = ckpt_data["faces"]     # (F, 3) int32

    if "vertex_colors" in ckpt_data:
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces,
            vertex_colors=ckpt_data["vertex_colors"],
            process=False,
        )
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    return mesh


def main():
    p = argparse.ArgumentParser(description="Replay layout_post_optimization from checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .npz checkpoint saved by sam3d_worker.py --checkpoint")
    p.add_argument("--output-dir", required=True,
                   help="Directory to write <name>.glb and <name>_info.json")
    p.add_argument("--name", required=False,
                   help="Output filename stem (default: derived from checkpoint name)")
    p.add_argument("--no-icp",  action="store_true", help="Disable ICP pass")
    p.add_argument("--no-adam", action="store_true", help="Disable Adam optimizer")
    args = p.parse_args()

    # Derive output name
    stem = args.name
    if stem is None:
        stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        if stem.endswith("_ckpt"):
            stem = stem[:-5]

    os.makedirs(args.output_dir, exist_ok=True)
    out_glb  = os.path.join(args.output_dir, f"{stem}.glb")
    out_info = os.path.join(args.output_dir, f"{stem}_info.json")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    data = np.load(args.checkpoint)
    for k in ("vertices", "faces", "rotation", "translation", "scale",
              "mask", "point_map", "intrinsics"):
        if k not in data:
            raise KeyError(f"Checkpoint is missing key '{k}'. "
                           f"Re-generate with sam3d_worker.py --checkpoint.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Tensors — restore original shapes recorded during capture.
    Quaternion  = torch.tensor(data["rotation"],    dtype=torch.float32).to(device)
    Translation = torch.tensor(data["translation"], dtype=torch.float32).to(device)
    Scale       = torch.tensor(data["scale"],       dtype=torch.float32).to(device)
    Mask        = torch.tensor(data["mask"],        dtype=torch.float32).to(device)
    Point_Map   = torch.tensor(data["point_map"],   dtype=torch.float32).to(device)
    Intrinsics  = torch.tensor(data["intrinsics"],  dtype=torch.float32).to(device)

    mesh = _build_mesh(data)

    print(f"Mesh  : {len(mesh.vertices)} verts, {len(mesh.faces)} faces", flush=True)
    print(f"Q shape: {Quaternion.shape}  T shape: {Translation.shape}  S shape: {Scale.shape}", flush=True)
    print(f"Mask shape: {Mask.shape}  PM shape: {Point_Map.shape}  Intr shape: {Intrinsics.shape}", flush=True)

    # ── Run layout_post_optimization ─────────────────────────────────────────
    from sam3d_objects.pipeline.inference_utils import layout_post_optimization

    revised_quat, revised_t, revised_scale, final_iou, flag_icp, flag_optim = (
        layout_post_optimization(
            mesh,
            Quaternion,
            Translation,
            Scale,
            Mask,
            Point_Map,
            Intrinsics,
            Enable_shape_ICP=not args.no_icp,
            Enable_rendering_optimization=not args.no_adam,
            min_size=518,
            device=device,
        )
    )
    print(f"\nResult: IoU={final_iou:.4f}  ICP={flag_icp}  Adam={flag_optim}", flush=True)

    # ── Apply transform and export GLB ────────────────────────────────────────
    R_out = revised_quat.squeeze().cpu().float()   # (4,)
    T_out = revised_t[0].cpu().float()             # (3,)
    S_out = revised_scale[0].cpu().float()         # (3,)

    verts_tf = _transform_vertices(data["vertices"], R_out, T_out, S_out, R_out.device)
    mesh.vertices = verts_tf.cpu().numpy().astype(np.float32)
    mesh.export(out_glb)
    print(f"GLB  : {out_glb}", flush=True)

    # ── Save info.json ────────────────────────────────────────────────────────
    info = {
        "glb_path":  out_glb,
        "translation": T_out.tolist(),
        "rotation":    R_out.tolist(),
        "scale":       S_out.tolist(),
        "iou":         final_iou,
        "flag_icp":    bool(flag_icp),
        "flag_optim":  bool(flag_optim),
        "checkpoint":  args.checkpoint,
    }
    with open(out_info, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Info : {out_info}", flush=True)


if __name__ == "__main__":
    main()
