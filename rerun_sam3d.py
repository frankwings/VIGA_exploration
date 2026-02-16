"""Re-run SAM3D TRELLIS reconstruction with fixed Transform3d.

Uses existing SAM masks from a previous run. Only re-runs the TRELLIS
3D reconstruction (sam3d_worker.py) for each mask, producing fresh GLBs
with the corrected coordinate transforms.

Usage:
    python rerun_sam3d.py
"""

import json
import os
import subprocess
import sys
import time

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SAM_INIT_DIR = os.path.join(
    ROOT, "output", "static_scene", "20260210_043534", "greentea", "sam_init"
)
TARGET_IMAGE = os.path.join(ROOT, "data", "static_scene", "greentea", "target.jpg")
CONFIG = os.path.join(
    ROOT, "utils", "third_party", "sam3d", "checkpoints", "hf", "checkpoints", "pipeline.yaml"
)
OUTPUT_DIR = os.path.join(ROOT, "output", "sam3d_rerun_fixed")
SAM3D_WORKER = os.path.join(ROOT, "tools", "sam3d", "sam3d_worker.py")

# sam3d_worker runs in the sam3d_py311 conda env
PYTHON_SAM3D = r"C:\Users\kingy\miniconda3\envs\sam3d_py311\python.exe"

# Objects to reconstruct (must have matching .npy mask in SAM_INIT_DIR)
OBJECTS = [
    "green_tea_bottle",
    "green_tea_bottle_1",
    "alienware_keyboard",
    "alienware_keyboard_1",
    "envelope",
    "headphones",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"SAM init dir: {SAM_INIT_DIR}")
    print(f"Target image: {TARGET_IMAGE}")
    print(f"Config: {CONFIG}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Python (sam3d_py311): {PYTHON_SAM3D}")
    print(f"Objects: {OBJECTS}")
    print()

    # Verify all inputs exist
    assert os.path.isfile(TARGET_IMAGE), f"Target image not found: {TARGET_IMAGE}"
    assert os.path.isfile(CONFIG), f"Config not found: {CONFIG}"
    assert os.path.isfile(PYTHON_SAM3D), f"Python not found: {PYTHON_SAM3D}"

    all_transforms = []
    total_start = time.time()

    for i, obj_name in enumerate(OBJECTS):
        mask_path = os.path.join(SAM_INIT_DIR, f"{obj_name}.npy")
        png_path = os.path.join(SAM_INIT_DIR, f"{obj_name}.png")
        glb_path = os.path.join(OUTPUT_DIR, f"{obj_name}.glb")
        info_path = os.path.join(OUTPUT_DIR, f"{obj_name}.json")
        log_path = os.path.join(OUTPUT_DIR, f"{obj_name}_sam3d.log")

        assert os.path.isfile(mask_path), f"Mask not found: {mask_path}"
        assert os.path.isfile(png_path), f"PNG not found: {png_path}"

        print(f"[{i+1}/{len(OBJECTS)}] Reconstructing {obj_name}...")
        obj_start = time.time()

        cmd = [
            PYTHON_SAM3D,
            SAM3D_WORKER,
            f"--image={TARGET_IMAGE}",
            f"--mask={mask_path}",
            f"--config={CONFIG}",
            f"--glb={glb_path}",
            f"--info={info_path}",
        ]

        # Use log file for stdout/stderr to avoid pipe deadlock on Windows
        with open(log_path, "w", encoding="utf-8") as log_f:
            result = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1 hour max per object
            )

        elapsed = time.time() - obj_start

        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
            print(f"  See log: {log_path}")
            continue

        # Read transform info
        if os.path.isfile(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            info["object_name"] = obj_name
            all_transforms.append(info)
            print(f"  OK ({elapsed:.0f}s) -> {glb_path}")
            if "intrinsics" in info:
                K = info["intrinsics"]
                print(f"  MoGe intrinsics: fx={K[0][0]:.1f}, fy={K[1][1]:.1f}, cx={K[0][2]:.1f}, cy={K[1][2]:.1f}")
        else:
            print(f"  OK ({elapsed:.0f}s) but no info JSON")

        print()

    total_elapsed = time.time() - total_start

    # Save combined transforms
    transforms_path = os.path.join(OUTPUT_DIR, "object_transforms.json")
    with open(transforms_path, "w", encoding="utf-8") as f:
        json.dump(all_transforms, f, indent=2)

    print(f"Done! {len(all_transforms)}/{len(OBJECTS)} objects reconstructed in {total_elapsed/60:.1f} minutes")
    print(f"Transforms: {transforms_path}")
    print(f"GLBs: {OUTPUT_DIR}/*.glb")


if __name__ == "__main__":
    main()
