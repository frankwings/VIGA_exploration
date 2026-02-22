"""Run SAM3D dining scene test with TRELLIS2 instead of TRELLIS1.

Three-step pipeline:
1. SAM segmentation (sam conda env)
2. TRELLIS2 3D reconstruction (trellis2 conda env)
3. Pose alignment via MoGe + layout_post_optimization (sam3d_py311 conda env)

Usage:
    python run_sam3d_dining_t2.py [--output-dir OUTPUT_DIR] [--skip-sam] [--skip-trellis2] [--skip-pose]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

# Paths
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
from utils._path import path_to_cmd

# Input data
TARGET_IMAGE = os.path.join(ROOT, "data", "static_scene", "dining", "target_resized.jpg")
DEFAULT_OUTPUT = os.path.join(ROOT, "output", "sam3d_dining_t2")

# Worker scripts
SAM_WORKER = os.path.join(ROOT, "tools", "sam3d", "sam_worker.py")
TRELLIS2_WORKER = os.path.join(ROOT, "tools", "sam3d", "trellis2_worker.py")
POSE_ALIGN_WORKER = os.path.join(ROOT, "tools", "sam3d", "pose_align_worker.py")

# Python executables for each env
SAM_PYTHON = path_to_cmd.get("tools/sam3d/sam_worker.py")
TRELLIS2_PYTHON = path_to_cmd.get("tools/sam3d/trellis2_worker.py")
POSE_PYTHON = path_to_cmd.get("tools/sam3d/sam3d_worker.py")  # sam3d_py311


def run_step(python_bin, script, args_list, label, env_extra=None):
    """Run a subprocess step with logging."""
    cmd = [python_bin, "-u", script] + args_list
    print(f"\n{'='*60}", flush=True)
    print(f"[STEP] {label}", flush=True)
    print(f"[CMD]  {' '.join(cmd)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=ROOT, env=env,
        text=True, stdin=subprocess.DEVNULL,
    )
    elapsed = time.time() - t0
    print(f"\n[STEP] {label}: {'OK' if result.returncode == 0 else 'FAILED'} "
          f"({elapsed:.1f}s)", flush=True)
    if result.returncode != 0:
        print(f"[STEP] Exit code: {result.returncode}", flush=True)
        sys.exit(1)
    return elapsed


def main():
    p = argparse.ArgumentParser(description="SAM3D + TRELLIS2 dining scene test")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--target-image", default=TARGET_IMAGE)
    p.add_argument("--skip-sam", action="store_true", help="Skip SAM segmentation")
    p.add_argument("--skip-trellis2", action="store_true", help="Skip TRELLIS2 reconstruction")
    p.add_argument("--skip-pose", action="store_true", help="Skip pose alignment")
    p.add_argument("--decimation-target", type=int, default=100000)
    p.add_argument("--texture-size", type=int, default=2048)
    args = p.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    target_image = os.path.abspath(args.target_image)
    os.makedirs(output_dir, exist_ok=True)

    timings = {}
    total_start = time.time()

    # --- Step 1: SAM Segmentation ---
    all_masks_path = os.path.join(output_dir, "all_masks.npy")
    names_path = os.path.join(output_dir, "all_masks_object_names.json")

    if not args.skip_sam:
        if SAM_PYTHON is None:
            print("[ERROR] SAM worker python not configured in _path.py", flush=True)
            sys.exit(1)
        timings["sam"] = run_step(
            SAM_PYTHON, SAM_WORKER,
            ["--image", target_image, "--out", all_masks_path],
            "SAM Segmentation",
        )
    else:
        print("[SKIP] SAM segmentation", flush=True)

    # Load masks and object names
    if not os.path.exists(all_masks_path):
        print(f"[ERROR] Masks not found: {all_masks_path}", flush=True)
        sys.exit(1)

    masks = np.load(all_masks_path, allow_pickle=True)
    if masks.dtype == object:
        masks = [m for m in masks]
    elif masks.ndim == 3:
        masks = [masks[i] for i in range(masks.shape[0])]
    else:
        masks = [masks]

    if os.path.exists(names_path):
        with open(names_path, 'r', encoding='utf-8') as f:
            names_data = json.load(f)
        object_names = names_data.get("object_mapping", [])
    else:
        object_names = [f"object_{i}" for i in range(len(masks))]

    print(f"\n[INFO] {len(masks)} objects: {object_names}", flush=True)

    # Save individual mask .npy files and masked PNGs
    img = np.array(Image.open(target_image).convert("RGB"))
    for i, (name, mask) in enumerate(zip(object_names, masks)):
        mask_path = os.path.join(output_dir, f"{name}.npy")
        if not os.path.exists(mask_path):
            np.save(mask_path, mask)

        # Create masked RGBA PNG for TRELLIS2
        png_path = os.path.join(output_dir, f"{name}.png")
        if not os.path.exists(png_path):
            mask_bool = mask > 0
            if mask_bool.ndim == 3:
                mask_bool = mask_bool[..., 0]
            # RGBA with mask as alpha
            rgba = np.zeros((*img.shape[:2], 4), dtype=np.uint8)
            rgba[..., :3] = img
            rgba[..., 3] = mask_bool.astype(np.uint8) * 255
            Image.fromarray(rgba).save(png_path)

    # --- Step 2: TRELLIS2 3D Reconstruction ---
    if not args.skip_trellis2:
        if TRELLIS2_PYTHON is None:
            print("[ERROR] TRELLIS2 worker python not configured in _path.py", flush=True)
            sys.exit(1)

        # Build TRELLIS2 manifest
        t2_objects = []
        for name in object_names:
            t2_objects.append({
                "name": name,
                "image": os.path.join(output_dir, f"{name}.png"),
                "mask": os.path.join(output_dir, f"{name}.npy"),
                "glb": os.path.join(output_dir, f"{name}_pbr.glb"),
                "mesh": os.path.join(output_dir, f"{name}_mesh.npz"),
            })
        t2_manifest = {"objects": t2_objects}
        t2_manifest_path = os.path.join(output_dir, "trellis2_manifest.json")
        with open(t2_manifest_path, 'w', encoding='utf-8') as f:
            json.dump(t2_manifest, f, indent=2)

        timings["trellis2"] = run_step(
            TRELLIS2_PYTHON, TRELLIS2_WORKER,
            ["--manifest", t2_manifest_path,
             "--decimation-target", str(args.decimation_target),
             "--texture-size", str(args.texture_size)],
            "TRELLIS2 3D Reconstruction",
        )
    else:
        print("[SKIP] TRELLIS2 reconstruction", flush=True)

    # --- Step 3: Pose Alignment ---
    if not args.skip_pose:
        if POSE_PYTHON is None:
            print("[ERROR] Pose align worker python not configured in _path.py", flush=True)
            sys.exit(1)

        # Build pose alignment manifest
        pose_objects = []
        for name in object_names:
            mesh_path = os.path.join(output_dir, f"{name}_mesh.npz")
            if not os.path.exists(mesh_path):
                print(f"[WARN] Mesh not found for {name}, skipping pose alignment", flush=True)
                continue
            pose_objects.append({
                "name": name,
                "mesh": mesh_path,
                "mask": os.path.join(output_dir, f"{name}.npy"),
                "glb": os.path.join(output_dir, f"{name}.glb"),
                "pbr_glb": os.path.join(output_dir, f"{name}_pbr.glb"),
                "aligned_pbr": os.path.join(output_dir, f"{name}_pbr_aligned.glb"),
                "info": os.path.join(output_dir, f"{name}_info.json"),
            })
        pose_manifest = {
            "scene_image": target_image,
            "objects": pose_objects,
        }
        pose_manifest_path = os.path.join(output_dir, "pose_manifest.json")
        with open(pose_manifest_path, 'w', encoding='utf-8') as f:
            json.dump(pose_manifest, f, indent=2)

        timings["pose_align"] = run_step(
            POSE_PYTHON, POSE_ALIGN_WORKER,
            ["--manifest", pose_manifest_path],
            "Pose Alignment (MoGe + layout_post_opt)",
            env_extra={"LIDRA_SKIP_INIT": "1"},
        )
    else:
        print("[SKIP] Pose alignment", flush=True)

    # --- Summary ---
    total_time = time.time() - total_start

    # Build combined object_transforms.json
    transforms = []
    for name in object_names:
        info_path = os.path.join(output_dir, f"{name}_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            transforms.append(info)

    transforms_path = os.path.join(output_dir, "object_transforms.json")
    with open(transforms_path, 'w', encoding='utf-8') as f:
        json.dump(transforms, f, indent=2)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"SAM3D + TRELLIS2 Pipeline Complete", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Objects:        {len(object_names)}", flush=True)
    print(f"Aligned:        {len(transforms)}", flush=True)
    for stage, t in timings.items():
        print(f"{stage:15s}: {t:.1f}s ({t/60:.1f}min)", flush=True)
    print(f"{'total':15s}: {total_time:.1f}s ({total_time/60:.1f}min)", flush=True)
    print(f"Output:         {output_dir}", flush=True)

    # IoU summary
    if transforms:
        print(f"\nPer-object IoU:", flush=True)
        for t in transforms:
            name = t.get("object_name", "?")
            iou = t.get("iou", -1)
            print(f"  {name:25s}: {iou:.4f}", flush=True)

    # Write summary JSON
    summary = {
        "target_image": target_image,
        "output_dir": output_dir,
        "num_objects": len(object_names),
        "num_aligned": len(transforms),
        "timings": timings,
        "total_time": total_time,
        "objects": {
            t.get("object_name", "?"): {"iou": t.get("iou", -1)}
            for t in transforms
        },
    }
    with open(os.path.join(output_dir, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone!", flush=True)


if __name__ == "__main__":
    main()
