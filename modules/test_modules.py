"""Integration tests for the 5-module SAM3D pipeline.

Validates manifest structure, output file existence, and data correctness
for each module. Supports two modes:

  1. Full run — runs all 5 modules from scratch, then validates
  2. Validation-only — checks existing outputs without re-running

Usage:
    # Validate existing outputs (fast, no GPU)
    python modules/test_modules.py \
        --reuse-dir output/modular_dining_v4/ \
        --image data/static_scene/dining/target_resized.jpg

    # Full run from scratch
    python modules/test_modules.py \
        --image data/static_scene/dining/target_resized.jpg \
        --output-dir output/test_modules/ \
        --trellis-version 1 \
        --vlm-model gemini-2.5-flash \
        --blender-command /usr/local/bin/blender

Conda env: agent (Python 3.10, orchestrator only)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

# ---------------------------------------------------------------------------
# Helpers (same as run_all.py)
# ---------------------------------------------------------------------------

def get_python_path(env_name: str) -> str:
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


def run_module(python_path: str, module_path: str, args: list,
               log_path: str, extra_env: dict = None) -> int:
    cmd = [python_path, "-u", module_path] + args
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_path}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    conda_prefix = os.path.dirname(os.path.dirname(python_path))
    env["CONDA_PREFIX"] = conda_prefix
    if extra_env:
        env.update(extra_env)

    with open(log_path, "w", encoding="utf-8") as lf:
        result = subprocess.run(
            cmd, cwd=str(ROOT), text=True,
            stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, env=env,
        )
    return result.returncode


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_file_exists(path: str, min_size: int = 0) -> str | None:
    """Return error string if file missing or too small, else None."""
    if not os.path.exists(path):
        return f"file not found: {path}"
    if min_size > 0 and os.path.getsize(path) < min_size:
        return f"file too small ({os.path.getsize(path)} bytes < {min_size}): {path}"
    return None


def check_npz_keys(path: str, required_keys: list) -> str | None:
    """Return error string if npz missing required keys."""
    import numpy as np
    try:
        data = np.load(path)
        for key in required_keys:
            if key not in data:
                return f"npz missing key '{key}': {path} (has: {list(data.keys())})"
        return None
    except Exception as e:
        return f"failed to load npz {path}: {e}"


# ---------------------------------------------------------------------------
# Per-module test functions
# ---------------------------------------------------------------------------

def test_segment(output_dir: str) -> tuple[bool, list[str]]:
    """Validate segment module outputs."""
    errors = []
    manifest_path = os.path.join(output_dir, "segment", "segment_manifest.json")

    err = check_file_exists(manifest_path)
    if err:
        return False, [err]

    m = load_manifest(manifest_path)

    # Required top-level keys
    for key in ["image", "image_shape", "num_masks", "masks"]:
        if key not in m:
            errors.append(f"manifest missing key '{key}'")

    if errors:
        return False, errors

    # image_shape should be [H, W]
    if not isinstance(m["image_shape"], list) or len(m["image_shape"]) != 2:
        errors.append(f"image_shape should be [H, W], got: {m['image_shape']}")

    masks = m["masks"]
    if not isinstance(masks, list) or len(masks) == 0:
        errors.append(f"masks array is empty or not a list")
        return False, errors

    if m["num_masks"] != len(masks):
        errors.append(f"num_masks ({m['num_masks']}) != len(masks) ({len(masks)})")

    # Validate each mask entry
    for i, mask in enumerate(masks):
        for key in ["id", "npy_path", "png_path", "area_pixels", "area_ratio"]:
            if key not in mask:
                errors.append(f"mask[{i}] missing key '{key}'")

        if "npy_path" in mask:
            err = check_file_exists(mask["npy_path"])
            if err:
                errors.append(f"mask[{i}]: {err}")
            else:
                # Check npy shape
                import numpy as np
                arr = np.load(mask["npy_path"])
                if arr.ndim != 2:
                    errors.append(f"mask[{i}] npy has {arr.ndim} dims, expected 2")

        if "png_path" in mask:
            err = check_file_exists(mask["png_path"])
            if err:
                errors.append(f"mask[{i}]: {err}")

        if "area_ratio" in mask:
            ar = mask["area_ratio"]
            if not isinstance(ar, (int, float)) or ar < 0 or ar > 1:
                errors.append(f"mask[{i}] area_ratio={ar} not in [0,1]")

    return len(errors) == 0, errors


def test_recognize(output_dir: str) -> tuple[bool, list[str]]:
    """Validate recognize module outputs."""
    errors = []
    manifest_path = os.path.join(output_dir, "recognize", "recognize_manifest.json")

    err = check_file_exists(manifest_path)
    if err:
        return False, [err]

    m = load_manifest(manifest_path)

    for key in ["image", "objects"]:
        if key not in m:
            errors.append(f"manifest missing key '{key}'")

    if errors:
        return False, errors

    objects = m["objects"]
    if not isinstance(objects, list) or len(objects) == 0:
        errors.append("objects array is empty or not a list")
        return False, errors

    # Check for duplicate names
    names = [obj.get("name", "") for obj in objects]
    dupes = [n for n in set(names) if names.count(n) > 1]
    if dupes:
        errors.append(f"duplicate object names: {dupes}")

    for i, obj in enumerate(objects):
        for key in ["name", "mask_id", "npy_path", "png_path"]:
            if key not in obj:
                errors.append(f"objects[{i}] missing key '{key}'")

        if "name" in obj and (not isinstance(obj["name"], str) or len(obj["name"]) == 0):
            errors.append(f"objects[{i}] name is empty")

        if "npy_path" in obj:
            err = check_file_exists(obj["npy_path"])
            if err:
                errors.append(f"objects[{i}] ({obj.get('name', '?')}): {err}")

        if "png_path" in obj:
            err = check_file_exists(obj["png_path"])
            if err:
                errors.append(f"objects[{i}] ({obj.get('name', '?')}): {err}")

    return len(errors) == 0, errors


def test_monodepth(output_dir: str) -> tuple[bool, list[str]]:
    """Validate monodepth module outputs."""
    import numpy as np
    errors = []
    manifest_path = os.path.join(output_dir, "monodepth", "monodepth_manifest.json")

    err = check_file_exists(manifest_path)
    if err:
        return False, [err]

    m = load_manifest(manifest_path)

    for key in ["image", "pointmap_path", "pointmap_shape", "intrinsics"]:
        if key not in m:
            errors.append(f"manifest missing key '{key}'")

    if errors:
        return False, errors

    # Check pointmap file
    ptmap_path = m["pointmap_path"]
    err = check_file_exists(ptmap_path)
    if err:
        errors.append(err)
    else:
        data = np.load(ptmap_path)
        # Key is "points" in the actual module output
        ptmap_key = None
        for candidate in ["points", "pointmap"]:
            if candidate in data:
                ptmap_key = candidate
                break
        if ptmap_key is None:
            errors.append(f"pointmap npz has no 'points' or 'pointmap' key (has: {list(data.keys())})")
        else:
            pts = data[ptmap_key]
            if pts.ndim != 3 or pts.shape[0] != 3:
                errors.append(f"pointmap shape={pts.shape}, expected (3, H, W)")
            if np.any(np.isnan(pts)):
                nan_count = np.count_nonzero(np.isnan(pts))
                errors.append(f"pointmap has {nan_count} NaN values")
            if np.any(np.isinf(pts)):
                inf_count = np.count_nonzero(np.isinf(pts))
                errors.append(f"pointmap has {inf_count} Inf values")

    # Check intrinsics is 3x3
    intr = m["intrinsics"]
    if not isinstance(intr, list) or len(intr) != 3:
        errors.append(f"intrinsics is not 3x3 (outer len={len(intr) if isinstance(intr, list) else type(intr)})")
    else:
        for row_i, row in enumerate(intr):
            if not isinstance(row, list) or len(row) != 3:
                errors.append(f"intrinsics[{row_i}] not length 3")

    # Check pointmap_shape
    shape = m["pointmap_shape"]
    if not isinstance(shape, list) or len(shape) != 2:
        errors.append(f"pointmap_shape should be [H, W], got: {shape}")

    return len(errors) == 0, errors


def test_reconstruction_3d(output_dir: str) -> tuple[bool, list[str]]:
    """Validate 3D reconstruction module outputs."""
    errors = []
    manifest_path = os.path.join(output_dir, "3d_reconstruction", "reconstruction_3d_manifest.json")

    err = check_file_exists(manifest_path)
    if err:
        return False, [err]

    m = load_manifest(manifest_path)

    if "objects" not in m:
        errors.append("manifest missing key 'objects'")
        return False, errors

    objects = m["objects"]
    if not isinstance(objects, list) or len(objects) == 0:
        errors.append("objects array is empty or not a list")
        return False, errors

    for i, obj in enumerate(objects):
        name = obj.get("name", f"obj_{i}")

        # Required fields
        for key in ["name", "glb_path"]:
            if key not in obj:
                errors.append(f"objects[{i}] ({name}) missing key '{key}'")

        # GLB file check (>10KB)
        if "glb_path" in obj:
            err = check_file_exists(obj["glb_path"], min_size=10240)
            if err:
                errors.append(f"objects[{i}] ({name}): {err}")

        # Mesh path check (checkpoint/mesh npz)
        mesh_key = None
        for candidate in ["checkpoint_path", "mesh_path"]:
            if candidate in obj and obj[candidate]:
                mesh_key = candidate
                break
        if mesh_key:
            mesh_path = obj[mesh_key]
            err = check_file_exists(mesh_path)
            if err:
                errors.append(f"objects[{i}] ({name}): {err}")
            else:
                err = check_npz_keys(mesh_path, ["vertices", "faces"])
                if err:
                    errors.append(f"objects[{i}] ({name}): {err}")

        # Vertex/face counts should be positive
        for count_key in ["vertices_count", "faces_count"]:
            if count_key in obj:
                val = obj[count_key]
                if not isinstance(val, int) or val <= 0:
                    errors.append(f"objects[{i}] ({name}) {count_key}={val} should be positive int")

    return len(errors) == 0, errors


def test_registration_2d3d(output_dir: str) -> tuple[bool, list[str]]:
    """Validate 2D-3D registration module outputs."""
    errors = []
    manifest_path = os.path.join(output_dir, "2d3d_registration", "registration_2d3d_manifest.json")

    err = check_file_exists(manifest_path)
    if err:
        return False, [err]

    m = load_manifest(manifest_path)

    for key in ["objects", "intrinsics", "pointmap_shape"]:
        if key not in m:
            errors.append(f"manifest missing key '{key}'")

    if "objects" not in m:
        return False, errors

    objects = m["objects"]
    if not isinstance(objects, list) or len(objects) == 0:
        errors.append("objects array is empty or not a list")
        return False, errors

    iou_values = []
    for i, obj in enumerate(objects):
        name = obj.get("name", f"obj_{i}")

        for key in ["name", "aligned_glb", "iou"]:
            if key not in obj:
                errors.append(f"objects[{i}] ({name}) missing key '{key}'")

        # Aligned GLB check (>10KB)
        if "aligned_glb" in obj:
            err = check_file_exists(obj["aligned_glb"], min_size=10240)
            if err:
                errors.append(f"objects[{i}] ({name}): {err}")

        # IoU check
        if "iou" in obj:
            iou = obj["iou"]
            if not isinstance(iou, (int, float)):
                errors.append(f"objects[{i}] ({name}) iou={iou} not a number")
            elif iou < -1 or iou > 1:
                errors.append(f"objects[{i}] ({name}) iou={iou} not in [-1, 1]")
            else:
                iou_values.append(iou)

        # Transform fields (optional but check if present)
        if "translation" in obj:
            t = obj["translation"]
            if not isinstance(t, list) or len(t) != 3:
                errors.append(f"objects[{i}] ({name}) translation not [x,y,z]")
        if "rotation" in obj:
            r = obj["rotation"]
            if not isinstance(r, list) or len(r) != 4:
                errors.append(f"objects[{i}] ({name}) rotation not [qw,qx,qy,qz]")
        if "scale" in obj:
            s = obj["scale"]
            if not isinstance(s, list) or len(s) != 3:
                errors.append(f"objects[{i}] ({name}) scale not [sx,sy,sz]")

    # At least one object should have IoU > 0.2
    if iou_values and max(iou_values) < 0.2:
        errors.append(f"no object has IoU > 0.2 (max={max(iou_values):.3f})")

    # Check intrinsics
    if "intrinsics" in m:
        intr = m["intrinsics"]
        if not isinstance(intr, list) or len(intr) != 3:
            errors.append("intrinsics not 3x3")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODULES = [
    ("1. Segment",            "segment",           test_segment),
    ("2. Recognize",          "recognize",         test_recognize),
    ("3. MonoDepth",          "monodepth",         test_monodepth),
    ("4. 3D Reconstruction",  "3d_reconstruction", test_reconstruction_3d),
    ("5. 2D-3D Registration", "2d3d_registration", test_registration_2d3d),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Integration tests for modular SAM3D pipeline")
    parser.add_argument("--image", required=True, help="Path to input scene image")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for fresh run (omit for validation-only)")
    parser.add_argument("--reuse-dir", default=None,
                        help="Existing output directory to validate (skip running modules)")
    parser.add_argument("--trellis-version", choices=["1", "2"], default="1")
    parser.add_argument("--vlm-model", default="gemini-2.5-flash")
    parser.add_argument("--blender-command", default="/usr/local/bin/blender")
    parser.add_argument("--max-faces", type=int, default=10000)
    parser.add_argument("--sam-checkpoint", default=None)
    args = parser.parse_args()

    if args.reuse_dir is None and args.output_dir is None:
        parser.error("must specify either --output-dir (fresh run) or --reuse-dir (validate existing)")

    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)

    # Determine output root
    if args.reuse_dir:
        output_root = os.path.abspath(args.reuse_dir)
        run_mode = False
        print(f"Mode: VALIDATION ONLY (reusing {output_root})")
    else:
        output_root = os.path.abspath(args.output_dir)
        run_mode = True
        print(f"Mode: FULL RUN (output: {output_root})")

    print(f"Image: {image_path}")
    print("=" * 60)

    # If full run, execute each module first
    if run_mode:
        modules_dir = str(ROOT / "modules")

        dirs = {
            "segment": os.path.join(output_root, "segment"),
            "recognize": os.path.join(output_root, "recognize"),
            "monodepth": os.path.join(output_root, "monodepth"),
            "3d_reconstruction": os.path.join(output_root, "3d_reconstruction"),
            "2d3d_registration": os.path.join(output_root, "2d3d_registration"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        seg_manifest = os.path.join(dirs["segment"], "segment_manifest.json")
        rec_manifest = os.path.join(dirs["recognize"], "recognize_manifest.json")
        depth_manifest = os.path.join(dirs["monodepth"], "monodepth_manifest.json")
        recon_manifest = os.path.join(dirs["3d_reconstruction"], "reconstruction_3d_manifest.json")

        # Module 1: Segment
        print("\n[RUN] Module 1: Segment")
        t0 = time.time()
        sam_python = get_python_path("sam")
        seg_args = ["--image", image_path, "--output-dir", dirs["segment"]]
        if args.sam_checkpoint:
            seg_args += ["--checkpoint", args.sam_checkpoint]
        rc = run_module(
            sam_python,
            os.path.join(modules_dir, "segment.py"),
            seg_args,
            os.path.join(output_root, "test_segment.log"),
        )
        print(f"  Exit code: {rc} ({time.time() - t0:.1f}s)")
        if rc != 0:
            print("  FAILED — check test_segment.log")

        # Module 2: Recognize
        print("\n[RUN] Module 2: Recognize")
        t0 = time.time()
        agent_python = get_python_path("agent")
        rc = run_module(
            agent_python,
            os.path.join(modules_dir, "recognize.py"),
            ["--input-manifest", seg_manifest, "--output-dir", dirs["recognize"],
             "--model", args.vlm_model],
            os.path.join(output_root, "test_recognize.log"),
        )
        print(f"  Exit code: {rc} ({time.time() - t0:.1f}s)")
        if rc != 0:
            print("  FAILED — check test_recognize.log")

        # Module 3: MonoDepth
        print("\n[RUN] Module 3: MonoDepth")
        t0 = time.time()
        sam3d_python = get_python_path("sam3d_py311")
        rc = run_module(
            sam3d_python,
            os.path.join(modules_dir, "monodepth.py"),
            ["--image", image_path, "--output-dir", dirs["monodepth"]],
            os.path.join(output_root, "test_monodepth.log"),
            extra_env={"LIDRA_SKIP_INIT": "1"},
        )
        print(f"  Exit code: {rc} ({time.time() - t0:.1f}s)")
        if rc != 0:
            print("  FAILED — check test_monodepth.log")

        # Module 4: 3D Reconstruction
        print("\n[RUN] Module 4: 3D Reconstruction")
        t0 = time.time()
        recon_args = [
            "--input-manifest", rec_manifest,
            "--scene-image", image_path,
            "--output-dir", dirs["3d_reconstruction"],
            "--trellis-version", args.trellis_version,
        ]
        if args.max_faces > 0:
            recon_args += ["--max-faces", str(args.max_faces)]
        rc = run_module(
            agent_python,
            os.path.join(modules_dir, "reconstruction_3d.py"),
            recon_args,
            os.path.join(output_root, "test_3d_reconstruction.log"),
        )
        print(f"  Exit code: {rc} ({time.time() - t0:.1f}s)")
        if rc != 0:
            print("  FAILED — check test_3d_reconstruction.log")

        # Module 5: 2D-3D Registration
        print("\n[RUN] Module 5: 2D-3D Registration")
        t0 = time.time()
        rc = run_module(
            agent_python,
            os.path.join(modules_dir, "registration_2d3d.py"),
            [
                "--reconstruct-manifest", recon_manifest,
                "--recognize-manifest", rec_manifest,
                "--monodepth-manifest", depth_manifest,
                "--output-dir", dirs["2d3d_registration"],
                "--blender-command", args.blender_command,
            ],
            os.path.join(output_root, "test_2d3d_registration.log"),
        )
        print(f"  Exit code: {rc} ({time.time() - t0:.1f}s)")
        if rc != 0:
            print("  FAILED — check test_2d3d_registration.log")

    # ---------------------------------------------------------------------------
    # Validation phase
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    results = []
    for label, subdir, test_fn in MODULES:
        print(f"\n[TEST] {label}")
        t0 = time.time()
        try:
            passed, details = test_fn(output_root)
        except Exception as e:
            passed = False
            details = [f"exception: {e}"]
        elapsed = time.time() - t0

        status = "PASS" if passed else "FAIL"
        results.append((label, passed, details, elapsed))

        print(f"  {status} ({elapsed:.2f}s)")
        if not passed:
            for d in details[:10]:  # cap at 10 errors shown
                print(f"    - {d}")
            if len(details) > 10:
                print(f"    ... and {len(details) - 10} more errors")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for label, passed, details, elapsed in results:
        status = "\033[32mPASS\033[0m" if passed else "\033[31mFAIL\033[0m"
        err_count = f" ({len(details)} errors)" if not passed else ""
        print(f"  {status}  {label}{err_count}")
        if not passed:
            all_passed = False

    total_pass = sum(1 for _, p, _, _ in results if p)
    total = len(results)
    print(f"\n  {total_pass}/{total} modules passed")

    if all_passed:
        print("\n  All tests passed!")
        sys.exit(0)
    else:
        print("\n  Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
