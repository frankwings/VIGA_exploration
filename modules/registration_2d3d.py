"""Module 5: 2D3DRegistration (ICP Pose Alignment).

Aligns each TRELLIS mesh to the 2D image using MoGe pointmap and
layout_post_optimization (ICP + gradient refinement). Produces aligned GLBs
and per-object pose transforms with IoU scores.

This module is an orchestrator that calls pose_align_worker.py in the
sam3d_py311 conda environment.

Usage:
    python modules/registration_2d3d.py \\
        --reconstruct-manifest <output/3d_reconstruction/reconstruction_3d_manifest.json> \\
        --recognize-manifest <output/recognize/recognize_manifest.json> \\
        --monodepth-manifest <output/monodepth/monodepth_manifest.json> \\
        --output-dir <output/2d3d_registration/>

Output:
    registration_2d3d_manifest.json — manifest with per-object aligned poses + IoU
    object_transforms.json          — combined transforms for downstream use
    {name}.glb                      — aligned GLB meshes
    {name}_info.json                — per-object pose info (translation, rotation, scale, IoU)

Conda env: agent (orchestrator), spawns sam3d_py311 for alignment
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.resolve()

TAG = "[2D3D_REGISTRATION]"


def get_python_path(env_name: str) -> str:
    """Get conda env python path."""
    if sys.platform == "win32":
        candidates = [
            Path.home() / "miniconda3" / "envs" / env_name / "python.exe",
            Path.home() / "miniconda3" / "envs" / env_name / "Scripts" / "python.exe",
        ]
    else:
        candidates = [
            Path.home() / "miniconda3" / "envs" / env_name / "bin" / "python",
        ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "python"


def run_pose_alignment(scene_image: str, objects: list, output_dir: str) -> dict:
    """Run pose_align_worker.py to align all objects.

    The worker loads MoGe internally, computes the scene pointmap once,
    then aligns each object using layout_post_optimization (ICP + gradient).
    """
    # Build pose alignment manifest
    pose_manifest = {
        "scene_image": scene_image,
        "objects": [],
    }

    for obj in objects:
        name = obj["name"]
        pose_manifest["objects"].append({
            "name": name,
            "mesh": obj["mesh_path"],
            "mask": obj["npy_path"],
            "glb": os.path.join(output_dir, f"{name}.glb"),
            "info": os.path.join(output_dir, f"{name}_info.json"),
        })

    manifest_path = os.path.join(output_dir, "pose_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(pose_manifest, f, indent=2)

    log_path = os.path.join(output_dir, "pose_align.log")
    sam3d_python = get_python_path("sam3d_py311")
    cmd = [
        sam3d_python, "-u",
        str(ROOT / "tools" / "sam3d" / "pose_align_worker.py"),
        "--manifest", manifest_path,
    ]

    print(f"{TAG} Running pose alignment ({len(objects)} objects)...")
    print(f"{TAG} Log: {log_path}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["LIDRA_SKIP_INIT"] = "1"
    conda_prefix = os.path.dirname(os.path.dirname(sam3d_python))
    env["CONDA_PREFIX"] = conda_prefix

    # SSL cert fix for GCP VM
    certifi_path = os.path.join(
        conda_prefix, "lib", "python3.11", "site-packages", "certifi", "cacert.pem"
    )
    if os.path.exists(certifi_path):
        env["SSL_CERT_FILE"] = certifi_path

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        result = subprocess.run(
            cmd, cwd=str(ROOT), text=True,
            stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=env,
        )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"{TAG} Pose alignment FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        print(f"{TAG} Check log: {log_path}")

    # Collect results from info JSONs
    results = {}
    for obj in objects:
        name = obj["name"]
        info_path = os.path.join(output_dir, f"{name}_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            results[name] = info
            iou = info.get("iou", "N/A")
            glb_path = os.path.join(output_dir, f"{name}.glb")
            glb_mb = os.path.getsize(glb_path) / (1024*1024) if os.path.exists(glb_path) else 0
            print(f"{TAG} {name}: IoU={iou}, GLB {glb_mb:.1f}MB")
        else:
            results[name] = None
            print(f"{TAG} {name}: FAILED (no info JSON)")

    print(f"{TAG} Pose alignment completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return results


def build_object_transforms(results: dict, output_dir: str) -> str:
    """Build combined object_transforms.json from per-object info JSONs."""
    transforms = {}
    for name, info in results.items():
        if info is None:
            continue
        transforms[name] = {
            "glb_path": os.path.join(output_dir, f"{name}.glb"),
            "translation": info.get("translation", [0, 0, 0]),
            "rotation": info.get("rotation", [1, 0, 0, 0]),
            "scale": info.get("scale", [1, 1, 1]),
            "iou": info.get("iou", -1),
        }

    # Add intrinsics from the first object that has them
    for info in results.values():
        if info and "intrinsics" in info:
            transforms["_intrinsics"] = info["intrinsics"]
            transforms["_pointmap_shape"] = info.get("pointmap_shape", [0, 0])
            break

    path = os.path.join(output_dir, "object_transforms.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)
    print(f"{TAG} Object transforms: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Module 5: 2D-3D Registration (ICP)")
    parser.add_argument("--reconstruct-manifest", required=True,
                        help="Path to reconstruction_3d_manifest.json from Module 4")
    parser.add_argument("--recognize-manifest", required=True,
                        help="Path to recognize_manifest.json from Module 2")
    parser.add_argument("--monodepth-manifest", required=True,
                        help="Path to monodepth_manifest.json from Module 3")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Read manifests
    with open(args.reconstruct_manifest, "r", encoding="utf-8") as f:
        recon_manifest = json.load(f)
    with open(args.recognize_manifest, "r", encoding="utf-8") as f:
        rec_manifest = json.load(f)
    with open(args.monodepth_manifest, "r", encoding="utf-8") as f:
        depth_manifest = json.load(f)

    scene_image = rec_manifest["image"]

    # Build object list: merge reconstruct mesh paths with recognize mask paths
    recon_by_name = {o["name"]: o for o in recon_manifest["objects"]}
    rec_by_name = {o["name"]: o for o in rec_manifest["objects"]}

    objects = []
    for name, recon_obj in recon_by_name.items():
        rec_obj = rec_by_name.get(name)
        if rec_obj is None:
            print(f"{TAG} WARNING: {name} not in recognize manifest, skipping")
            continue
        objects.append({
            "name": name,
            "mesh_path": recon_obj["mesh_path"],
            "npy_path": rec_obj["npy_path"],
        })

    print(f"{TAG} Scene image: {scene_image}")
    print(f"{TAG} {len(objects)} objects to align")

    # Step 1: Run pose alignment (ICP + gradient refinement)
    results = run_pose_alignment(scene_image, objects, output_dir)

    # Step 2: Build combined transforms
    build_object_transforms(results, output_dir)

    # Build final manifest
    obj_entries = []
    for name, info in results.items():
        if info is None:
            continue
        obj_entries.append({
            "name": name,
            "aligned_glb": os.path.join(output_dir, f"{name}.glb"),
            "iou": info.get("iou", -1),
            "translation": info.get("translation", [0, 0, 0]),
            "rotation": info.get("rotation", [1, 0, 0, 0]),
            "scale": info.get("scale", [1, 1, 1]),
        })

    # Include intrinsics from monodepth manifest
    manifest = {
        "objects": obj_entries,
        "intrinsics": depth_manifest.get("intrinsics"),
        "pointmap_shape": depth_manifest.get("pointmap_shape"),
    }
    manifest_path = os.path.join(output_dir, "registration_2d3d_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    success = len(obj_entries)
    total = len(objects)
    avg_iou = np.mean([o["iou"] for o in obj_entries if o["iou"] > 0]) if obj_entries else 0

    print(f"\n{TAG} Done: {success}/{total} objects aligned")
    print(f"{TAG} Average IoU: {avg_iou:.2f}")
    print(f"{TAG} Manifest: {manifest_path}")
    for o in obj_entries:
        print(f"  {o['name']}: IoU={o['iou']:.4f}")


if __name__ == "__main__":
    main()
