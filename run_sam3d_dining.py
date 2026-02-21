#!/usr/bin/env python3
"""Run full SAM3D pipeline (segmentation + 3D reconstruction) for the dining scene.

Usage on GCP VM:
    export DISPLAY=:99
    export LIDRA_SKIP_INIT=1
    ~/miniconda3/envs/agent/bin/python run_sam3d_dining.py \
        --output-dir output/sam3d_dining

This script:
  1. Runs SAM segmentation to detect all objects in data/static_scene/dining/target_resized.jpg
  2. Runs SAM3D (TRELLIS) reconstruction for each detected object
  3. Produces per-object GLBs + info JSONs with transform data
  4. Optionally imports all into a Blender scene

Uses --batch mode by default: loads the TRELLIS model once and processes all
objects in a single subprocess, saving ~21s per object on model reloading.
Use --no-batch to fall back to per-object subprocesses.

Note: numpy .npy files are used for mask interchange between SAM and SAM3D workers
(this is the existing VIGA convention for binary mask data).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.resolve()


def get_python_path(env_name: str) -> str:
    """Get conda env python path (Linux)."""
    home = Path.home()
    p = home / "miniconda3" / "envs" / env_name / "bin" / "python"
    if p.exists():
        return str(p)
    return "python"


def run_sam_segmentation(image_path: str, output_dir: str) -> str:
    """Step 1: Run SAM to detect and segment all objects."""
    masks_path = os.path.join(output_dir, "all_masks.npy")
    log_path = os.path.join(output_dir, "sam_worker.log")

    sam_python = get_python_path("sam")
    cmd = [
        sam_python, "-u",
        str(ROOT / "tools" / "sam3d" / "sam_worker.py"),
        "--image", image_path,
        "--out", masks_path,
    ]

    print(f"[SAM] Running segmentation...")
    print(f"[SAM] Command: {' '.join(cmd)}")
    print(f"[SAM] Log: {log_path}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    conda_prefix = os.path.dirname(os.path.dirname(sam_python))
    env["CONDA_PREFIX"] = conda_prefix

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        subprocess.run(cmd, cwd=str(ROOT), check=True, text=True,
                       stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - start
    print(f"[SAM] Segmentation completed in {elapsed:.1f}s")
    return masks_path


def run_sam3d_batch(image_path: str, object_names: list, output_dir: str,
                    config_path: str) -> dict:
    """Run SAM3D reconstruction for all objects in a single process (model cached)."""
    manifest = {
        "config": config_path,
        "scene_image": image_path,
        "objects": [],
    }
    for name in object_names:
        manifest["objects"].append({
            "name": name,
            "image": image_path,
            "mask": os.path.join(output_dir, f"{name}.npy"),
            "glb": os.path.join(output_dir, f"{name}.glb"),
            "info": os.path.join(output_dir, f"{name}_info.json"),
            "checkpoint": os.path.join(output_dir, f"{name}_checkpoint.npz"),
        })

    manifest_path = os.path.join(output_dir, "batch_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log_path = os.path.join(output_dir, "sam3d_batch.log")
    sam3d_python = get_python_path("sam3d_py311")
    cmd = [
        sam3d_python, "-u",
        str(ROOT / "tools" / "sam3d" / "sam3d_batch_worker.py"),
        "--manifest", manifest_path,
    ]

    print(f"[SAM3D-BATCH] Processing {len(object_names)} objects with cached model...")
    print(f"[SAM3D-BATCH] Log: {log_path}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["LIDRA_SKIP_INIT"] = "1"
    conda_prefix = os.path.dirname(os.path.dirname(sam3d_python))
    env["CONDA_PREFIX"] = conda_prefix

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        result = subprocess.run(cmd, cwd=str(ROOT), text=True,
                                stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"[SAM3D-BATCH] FAILED (exit code {result.returncode}, {elapsed:.1f}s)")
        print(f"[SAM3D-BATCH]   Check log: {log_path}")
        return {}

    # Read per-object results from info JSONs
    results = {}
    for name in object_names:
        info_path = os.path.join(output_dir, f"{name}_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                results[name] = json.load(f)
            glb_path = os.path.join(output_dir, f"{name}.glb")
            glb_size = os.path.getsize(glb_path) / (1024 * 1024)
            iou = results[name].get("iou", "N/A")
            print(f"[SAM3D] {name}: SUCCESS ({glb_size:.1f}MB, IoU={iou})")
        else:
            results[name] = None
            print(f"[SAM3D] {name}: FAILED")

    print(f"[SAM3D-BATCH] Total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return results


def run_sam3d_reconstruction(image_path: str, mask_path: str, output_dir: str,
                              object_name: str, config_path: str, scene_image: str = None) -> dict:
    """Step 2 (legacy): Run SAM3D reconstruction for a single object in its own subprocess."""
    glb_path = os.path.join(output_dir, f"{object_name}.glb")
    info_path = os.path.join(output_dir, f"{object_name}_info.json")
    log_path = os.path.join(output_dir, f"{object_name}_sam3d.log")
    ckpt_path = os.path.join(output_dir, f"{object_name}_checkpoint.npz")

    # Skip if already completed
    if os.path.exists(glb_path) and os.path.exists(info_path):
        print(f"[SAM3D] {object_name}: already exists, skipping")
        with open(info_path, "r") as f:
            return json.load(f)

    sam3d_python = get_python_path("sam3d_py311")
    cmd = [
        sam3d_python, "-u",
        str(ROOT / "tools" / "sam3d" / "sam3d_worker.py"),
        "--image", image_path,
        "--mask", mask_path,
        "--config", config_path,
        "--glb", glb_path,
        "--info", info_path,
        "--checkpoint", ckpt_path,
    ]
    if scene_image:
        cmd.extend(["--scene-image", scene_image])

    print(f"[SAM3D] Reconstructing {object_name}...")
    print(f"[SAM3D] Log: {log_path}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["LIDRA_SKIP_INIT"] = "1"
    conda_prefix = os.path.dirname(os.path.dirname(sam3d_python))
    env["CONDA_PREFIX"] = conda_prefix

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        result = subprocess.run(cmd, cwd=str(ROOT), text=True,
                                stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"[SAM3D] {object_name}: FAILED (exit code {result.returncode}, {elapsed:.1f}s)")
        print(f"[SAM3D]   Check log: {log_path}")
        return None

    if not os.path.exists(info_path):
        print(f"[SAM3D] {object_name}: FAILED - no info file produced ({elapsed:.1f}s)")
        return None

    with open(info_path, "r") as f:
        info = json.load(f)

    glb_size = os.path.getsize(glb_path) / (1024 * 1024)
    iou = info.get("iou", "N/A")
    print(f"[SAM3D] {object_name}: SUCCESS ({elapsed:.1f}s, {glb_size:.1f}MB, IoU={iou})")
    return info


def run_blender_import(output_dir: str, transforms: list, blender_cmd: str = None) -> str:
    """Step 3: Import all GLBs into a Blender scene."""
    if not blender_cmd:
        blender_cmd = "/usr/local/bin/blender"

    transforms_path = os.path.join(output_dir, "object_transforms.json")
    blend_path = os.path.join(output_dir, "scene.blend")
    log_path = os.path.join(output_dir, "blender_import.log")

    with open(transforms_path, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)

    import_script = str(ROOT / "tools" / "blender" / "glb_import.py")
    cmd = [blender_cmd, "-b", "-P", import_script, "--", transforms_path, blend_path]

    print(f"[BLENDER] Importing {len(transforms)} objects into scene...")
    with open(log_path, "w", encoding="utf-8") as lf:
        subprocess.run(cmd, cwd=str(ROOT), check=True, text=True,
                       stdout=lf, stderr=subprocess.STDOUT)
    print(f"[BLENDER] Scene saved: {blend_path}")
    return blend_path


def main():
    parser = argparse.ArgumentParser(description="Run SAM3D pipeline for dining scene")
    parser.add_argument("--output-dir", default="output/sam3d_dining",
                        help="Output directory for results")
    parser.add_argument("--image", default="data/static_scene/dining/target_resized.jpg",
                        help="Target image path")
    parser.add_argument("--blender-command", default="/usr/local/bin/blender",
                        help="Path to Blender executable")
    parser.add_argument("--skip-blender", action="store_true",
                        help="Skip Blender import step")
    parser.add_argument("--no-batch", action="store_true",
                        help="Use per-object subprocesses instead of batch worker")
    args = parser.parse_args()

    output_dir = os.path.join(str(ROOT), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    image_path = os.path.join(str(ROOT), args.image)
    if not os.path.exists(image_path):
        print(f"ERROR: Target image not found: {image_path}")
        sys.exit(1)

    config_path = str(ROOT / "utils" / "third_party" / "sam3d" / "checkpoints" / "hf" / "checkpoints" / "pipeline.yaml")
    if not os.path.exists(config_path):
        print(f"ERROR: SAM3D config not found: {config_path}")
        sys.exit(1)

    batch_mode = not args.no_batch
    print(f"{'='*60}")
    print(f"SAM3D Dining Scene Pipeline {'(BATCH)' if batch_mode else '(per-object)'}")
    print(f"{'='*60}")
    print(f"Image:  {image_path}")
    print(f"Output: {output_dir}")
    print(f"Config: {config_path}")
    print(f"{'='*60}")

    total_start = time.time()

    # Step 1: SAM segmentation
    masks_path = os.path.join(output_dir, "all_masks.npy")
    if os.path.exists(masks_path):
        print(f"[SAM] Masks already exist at {masks_path}, skipping segmentation")
    else:
        masks_path = run_sam_segmentation(image_path, output_dir)

    # Load masks and object names
    masks = np.load(masks_path, allow_pickle=True)
    if masks.dtype == object:
        masks = [m for m in masks]
    elif masks.ndim == 3:
        masks = [masks[i] for i in range(masks.shape[0])]
    else:
        masks = [masks]

    names_path = masks_path.replace(".npy", "_object_names.json")
    object_names = []
    if os.path.exists(names_path):
        with open(names_path, "r") as f:
            names_info = json.load(f)
            object_names = names_info.get("object_mapping", [])
    if not object_names:
        object_names = [f"object_{i}" for i in range(len(masks))]

    print(f"\n[PIPELINE] Found {len(masks)} objects: {', '.join(object_names)}")

    # Save individual mask files (needed by sam3d_worker)
    for i, (mask, name) in enumerate(zip(masks, object_names)):
        mask_path = os.path.join(output_dir, f"{name}.npy")
        if not os.path.exists(mask_path):
            np.save(mask_path, mask)
            print(f"[MASKS] Saved mask: {name} (shape: {mask.shape})")

    # Step 2: SAM3D reconstruction
    if batch_mode:
        results = run_sam3d_batch(image_path, object_names, output_dir, config_path)
    else:
        results = {}
        for name in object_names:
            mask_path = os.path.join(output_dir, f"{name}.npy")
            info = run_sam3d_reconstruction(
                image_path=image_path,
                mask_path=mask_path,
                output_dir=output_dir,
                object_name=name,
                config_path=config_path,
                scene_image=image_path,
            )
            results[name] = info

    # Build transforms for Blender import
    transforms = []
    for name, info in results.items():
        if info:
            transforms.append({
                "glb_path": info.get("glb_path"),
                "translation": info.get("translation"),
                "rotation": info.get("rotation"),
                "scale": info.get("scale"),
            })

    # Step 3: Blender import (optional)
    if not args.skip_blender and transforms:
        try:
            run_blender_import(output_dir, transforms, args.blender_command)
        except Exception as e:
            print(f"[BLENDER] Import failed: {e} (non-fatal)")

    # Summary
    total_time = time.time() - total_start
    success_count = sum(1 for v in results.values() if v is not None)
    total_count = len(results)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    for name, info in results.items():
        if info:
            iou = info.get("iou", "N/A")
            print(f"  {name}: OK (IoU={iou})")
        else:
            print(f"  {name}: FAILED")
    print(f"\nSuccessful: {success_count}/{total_count}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Output: {output_dir}")

    # Save summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image": image_path,
        "batch_mode": batch_mode,
        "total_time_seconds": total_time,
        "objects": {name: {"success": info is not None, "iou": info.get("iou") if info else None}
                    for name, info in results.items()},
        "summary": {"successful": success_count, "failed": total_count - success_count, "total": total_count},
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
