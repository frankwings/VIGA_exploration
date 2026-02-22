"""Orchestrator: Run all 5 modules in sequence.

Runs the full pipeline from a single image:
  1. Segment  — SAM mask generation
  2. Recognize — VLM object naming
  3. MonoDepth — MoGe depth estimation
  4. Reconstruct — TRELLIS 3D reconstruction
  5. Register — 2D-3D pose alignment

Each module runs in its own conda environment via subprocess. Manifests
(JSON files) are passed between modules to maintain independence.

Usage:
    python modules/run_all.py \
        --image <target.jpg> \
        --output-dir <output/pipeline_run/> \
        --trellis-version 1 \
        --vlm-model gpt-4o

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


def run_module(python_path: str, module_path: str, args: list,
               log_path: str, extra_env: dict = None) -> int:
    """Run a module as a subprocess.

    Returns the exit code.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full modular pipeline")
    parser.add_argument("--image", required=True, help="Path to input scene image")
    parser.add_argument("--output-dir", required=True, help="Root output directory")
    parser.add_argument("--trellis-version", choices=["1", "2"], default="1",
                        help="TRELLIS version (1 or 2)")
    parser.add_argument("--vlm-model", default="gpt-4o",
                        help="VLM model for object naming")
    parser.add_argument("--blender-command", default="/usr/local/bin/blender",
                        help="Path to Blender executable")
    parser.add_argument("--sam-checkpoint", default=None,
                        help="SAM ViT-H checkpoint path")
    parser.add_argument("--skip-segment", action="store_true",
                        help="Skip segmentation (reuse existing)")
    parser.add_argument("--skip-recognize", action="store_true",
                        help="Skip recognition (reuse existing)")
    parser.add_argument("--skip-monodepth", action="store_true",
                        help="Skip mono depth (reuse existing)")
    parser.add_argument("--skip-reconstruct", action="store_true",
                        help="Skip reconstruction (reuse existing)")
    parser.add_argument("--skip-register", action="store_true",
                        help="Skip registration (reuse existing)")
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    output_root = os.path.abspath(args.output_dir)

    # Sub-directories for each module
    dirs = {
        "segment": os.path.join(output_root, "segment"),
        "recognize": os.path.join(output_root, "recognize"),
        "monodepth": os.path.join(output_root, "monodepth"),
        "reconstruct": os.path.join(output_root, "reconstruct"),
        "register": os.path.join(output_root, "register"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    modules_dir = str(ROOT / "modules")

    print("=" * 60)
    print("Modular SAM3D Pipeline")
    print("=" * 60)
    print(f"Image:    {image_path}")
    print(f"Output:   {output_root}")
    print(f"TRELLIS:  v{args.trellis_version}")
    print(f"VLM:      {args.vlm_model}")
    print(f"Blender:  {args.blender_command}")
    print("=" * 60)

    total_start = time.time()
    timings = {}

    # -----------------------------------------------------------------------
    # Module 1: Segment
    # -----------------------------------------------------------------------
    seg_manifest = os.path.join(dirs["segment"], "segment_manifest.json")
    if not args.skip_segment:
        print("\n[1/5] SEGMENT")
        t0 = time.time()
        sam_python = get_python_path("sam")
        module_args = [
            "--image", image_path,
            "--output-dir", dirs["segment"],
        ]
        if args.sam_checkpoint:
            module_args += ["--checkpoint", args.sam_checkpoint]

        rc = run_module(
            sam_python,
            os.path.join(modules_dir, "segment.py"),
            module_args,
            os.path.join(output_root, "segment.log"),
        )
        timings["segment"] = time.time() - t0
        if rc != 0:
            print(f"[1/5] SEGMENT FAILED (exit {rc})")
            sys.exit(1)
        print(f"[1/5] SEGMENT done ({timings['segment']:.1f}s)")
    else:
        print("\n[1/5] SEGMENT (skipped)")

    # -----------------------------------------------------------------------
    # Module 2: Recognize
    # -----------------------------------------------------------------------
    rec_manifest = os.path.join(dirs["recognize"], "recognize_manifest.json")
    if not args.skip_recognize:
        print("\n[2/5] RECOGNIZE")
        t0 = time.time()
        agent_python = get_python_path("agent")
        rc = run_module(
            agent_python,
            os.path.join(modules_dir, "recognize.py"),
            [
                "--input-manifest", seg_manifest,
                "--output-dir", dirs["recognize"],
                "--model", args.vlm_model,
            ],
            os.path.join(output_root, "recognize.log"),
        )
        timings["recognize"] = time.time() - t0
        if rc != 0:
            print(f"[2/5] RECOGNIZE FAILED (exit {rc})")
            sys.exit(1)
        print(f"[2/5] RECOGNIZE done ({timings['recognize']:.1f}s)")
    else:
        print("\n[2/5] RECOGNIZE (skipped)")

    # -----------------------------------------------------------------------
    # Module 3: MonoDepth
    # -----------------------------------------------------------------------
    depth_manifest = os.path.join(dirs["monodepth"], "monodepth_manifest.json")
    if not args.skip_monodepth:
        print("\n[3/5] MONODEPTH")
        t0 = time.time()
        sam3d_python = get_python_path("sam3d_py311")
        rc = run_module(
            sam3d_python,
            os.path.join(modules_dir, "monodepth.py"),
            ["--image", image_path, "--output-dir", dirs["monodepth"]],
            os.path.join(output_root, "monodepth.log"),
            extra_env={"LIDRA_SKIP_INIT": "1"},
        )
        timings["monodepth"] = time.time() - t0
        if rc != 0:
            print(f"[3/5] MONODEPTH FAILED (exit {rc})")
            sys.exit(1)
        print(f"[3/5] MONODEPTH done ({timings['monodepth']:.1f}s)")
    else:
        print("\n[3/5] MONODEPTH (skipped)")

    # -----------------------------------------------------------------------
    # Module 4: Reconstruct
    # -----------------------------------------------------------------------
    recon_manifest = os.path.join(dirs["reconstruct"], "reconstruct_manifest.json")
    if not args.skip_reconstruct:
        print("\n[4/5] RECONSTRUCT")
        t0 = time.time()
        agent_python = get_python_path("agent")
        rc = run_module(
            agent_python,
            os.path.join(modules_dir, "reconstruct.py"),
            [
                "--input-manifest", rec_manifest,
                "--scene-image", image_path,
                "--output-dir", dirs["reconstruct"],
                "--trellis-version", args.trellis_version,
            ],
            os.path.join(output_root, "reconstruct.log"),
        )
        timings["reconstruct"] = time.time() - t0
        if rc != 0:
            print(f"[4/5] RECONSTRUCT FAILED (exit {rc})")
            sys.exit(1)
        print(f"[4/5] RECONSTRUCT done ({timings['reconstruct']:.1f}s)")
    else:
        print("\n[4/5] RECONSTRUCT (skipped)")

    # -----------------------------------------------------------------------
    # Module 5: Register
    # -----------------------------------------------------------------------
    reg_manifest = os.path.join(dirs["register"], "register_manifest.json")
    if not args.skip_register:
        print("\n[5/5] REGISTER")
        t0 = time.time()
        agent_python = get_python_path("agent")
        rc = run_module(
            agent_python,
            os.path.join(modules_dir, "register.py"),
            [
                "--reconstruct-manifest", recon_manifest,
                "--recognize-manifest", rec_manifest,
                "--monodepth-manifest", depth_manifest,
                "--output-dir", dirs["register"],
                "--blender-command", args.blender_command,
            ],
            os.path.join(output_root, "register.log"),
        )
        timings["register"] = time.time() - t0
        if rc != 0:
            print(f"[5/5] REGISTER FAILED (exit {rc})")
            sys.exit(1)
        print(f"[5/5] REGISTER done ({timings['register']:.1f}s)")
    else:
        print("\n[5/5] REGISTER (skipped)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    for module, t in timings.items():
        print(f"  {module:15s} {t:7.1f}s ({t/60:.1f} min)")
    print(f"  {'TOTAL':15s} {total_time:7.1f}s ({total_time/60:.1f} min)")
    print(f"\nOutput: {output_root}")

    # Save pipeline summary
    summary = {
        "image": image_path,
        "output_dir": output_root,
        "trellis_version": args.trellis_version,
        "vlm_model": args.vlm_model,
        "total_time_seconds": total_time,
        "timings": timings,
        "manifests": {
            "segment": seg_manifest,
            "recognize": rec_manifest,
            "monodepth": depth_manifest,
            "reconstruct": recon_manifest,
            "register": reg_manifest,
        },
    }
    summary_path = os.path.join(output_root, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
