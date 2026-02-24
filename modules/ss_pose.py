"""Module 4b: SS Pose Estimation.

Uses Meta's Sparse Structure model (from SAM3D Objects) to predict initial
rotation/translation/scale for objects that lack this information. This is
primarily for TRELLIS2 objects, which don't have their own pose prediction.

The SS model is a 1.2B-parameter Mixture-of-Transformers that jointly predicts
64^3 voxel occupancy and 6D rotation + translation + scale from a masked image
and MoGe pointmap. We only run Stage 1 (SS) — the expensive SLAT and decoder
stages are skipped via stage1_only=True.

Inputs:
    - reconstruction_3d_manifest.json (from Module 4)
    - recognize_manifest.json (from Module 2, for mask paths)
    - monodepth_manifest.json (from Module 3, for pointmap)
    - scene image

Output:
    ss_pose_manifest.json — maps object name → checkpoint path
    {name}_ss_checkpoint.npz — rotation/translation/scale per object

Usage:
    python modules/ss_pose.py \\
        --reconstruct-manifest <output/3d_reconstruction/reconstruction_3d_manifest.json> \\
        --recognize-manifest <output/recognize/recognize_manifest.json> \\
        --monodepth-manifest <output/monodepth/monodepth_manifest.json> \\
        --scene-image <target.jpg> \\
        --output-dir <output/ss_pose/>

Conda env: agent (orchestrator — launches sam3d_py311 subprocess)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

TAG = "[SS_POSE]"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 4b: SS Pose Estimation")
    parser.add_argument("--reconstruct-manifest", required=True,
                        help="Path to reconstruction_3d_manifest.json from Module 4")
    parser.add_argument("--recognize-manifest", required=True,
                        help="Path to recognize_manifest.json from Module 2")
    parser.add_argument("--monodepth-manifest", required=True,
                        help="Path to monodepth_manifest.json from Module 3")
    parser.add_argument("--scene-image", required=True,
                        help="Path to input scene image")
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

    scene_image = os.path.abspath(args.scene_image)
    pointmap_path = depth_manifest["pointmap_path"]

    # SAM3D pipeline config
    config_path = str(ROOT / "utils" / "third_party" / "sam3d" / "checkpoints" /
                      "hf" / "checkpoints" / "pipeline.yaml")

    # Build mask lookup from recognize manifest
    rec_by_name = {o["name"]: o for o in rec_manifest["objects"]}

    # Filter to objects that DON'T already have a checkpoint with pose data.
    # TRELLIS1 objects already have SS pose from sam3d_batch_worker; skip them.
    objects_needing_pose = []
    for obj in recon_manifest["objects"]:
        name = obj["name"]
        # Skip if reconstruction already has a checkpoint with pose
        ckpt = obj.get("checkpoint_path")
        if ckpt and os.path.exists(ckpt):
            import numpy as np
            try:
                data = np.load(ckpt)
                if all(k in data for k in ("rotation", "translation", "scale")):
                    print(f"{TAG} {name}: already has SS pose from reconstruction, skipping")
                    continue
            except Exception:
                pass

        rec_obj = rec_by_name.get(name)
        if rec_obj is None:
            print(f"{TAG} WARNING: {name} not in recognize manifest, skipping")
            continue

        objects_needing_pose.append({
            "name": name,
            "mask": rec_obj["npy_path"],
        })

    if not objects_needing_pose:
        print(f"{TAG} All objects already have SS pose — nothing to do")
        # Write empty manifest
        manifest = {"objects": {}}
        manifest_path = os.path.join(output_dir, "ss_pose_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"{TAG} Manifest: {manifest_path}")
        return

    print(f"{TAG} {len(objects_needing_pose)} objects need SS pose estimation")

    # Build worker manifest
    worker_manifest = {
        "config": config_path,
        "scene_image": scene_image,
        "pointmap": pointmap_path,
        "objects": [],
    }
    for obj in objects_needing_pose:
        worker_manifest["objects"].append({
            "name": obj["name"],
            "image": scene_image,
            "mask": obj["mask"],
            "checkpoint": os.path.join(output_dir, f"{obj['name']}_ss_checkpoint.npz"),
        })

    worker_manifest_path = os.path.join(output_dir, "ss_pose_worker_manifest.json")
    with open(worker_manifest_path, "w", encoding="utf-8") as f:
        json.dump(worker_manifest, f, indent=2)

    # Launch worker subprocess in sam3d_py311 env
    log_path = os.path.join(output_dir, "ss_pose_worker.log")
    sam3d_python = get_python_path("sam3d_py311")
    cmd = [
        sam3d_python, "-u",
        str(ROOT / "tools" / "sam3d" / "ss_pose_worker.py"),
        "--manifest", worker_manifest_path,
    ]

    print(f"{TAG} Running SS pose worker ({len(objects_needing_pose)} objects)...")
    print(f"{TAG} Command: {' '.join(cmd)}")
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
        print(f"{TAG} SS pose worker FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        print(f"{TAG} Check log: {log_path}")
    else:
        print(f"{TAG} SS pose worker completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Collect results: map object name → checkpoint path
    checkpoint_map = {}
    for obj in objects_needing_pose:
        name = obj["name"]
        ckpt_path = os.path.join(output_dir, f"{name}_ss_checkpoint.npz")
        if os.path.exists(ckpt_path):
            checkpoint_map[name] = ckpt_path
            print(f"{TAG} {name}: checkpoint OK")
        else:
            print(f"{TAG} {name}: FAILED (no checkpoint)")

    # Write module manifest
    manifest = {"objects": checkpoint_map}
    manifest_path = os.path.join(output_dir, "ss_pose_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    success = len(checkpoint_map)
    total = len(objects_needing_pose)
    print(f"\n{TAG} Done: {success}/{total} objects")
    print(f"{TAG} Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
