"""Orchestrator: Run all modules in sequence.

Runs the full pipeline from a single image:
  1.  Segment            — SAM mask generation
  2.  Recognize          — VLM object naming
  3.  MonoDepth          — MoGe depth estimation
  4a. 3DReconstruction   — TRELLIS 3D mesh generation
  4b. SS Pose (TRELLIS2) — Meta's SS model predicts initial rotation/translation/scale
  5.  2D3DRegistration   — ICP pose alignment

Module 4b only runs for TRELLIS2 objects (which lack built-in pose prediction).
TRELLIS1 objects already have SS pose from the integrated SAM3D pipeline.

Each module runs in its own conda environment via subprocess. Manifests
(JSON files) are passed between modules to maintain independence.

Usage:
    python modules/run_all.py \\
        --image <target.jpg> \\
        --output-dir <output/pipeline_run/> \\
        --trellis-version 1 \\
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


def _merge_ss_pose(recon_manifest_path: str, ss_pose_manifest_path: str,
                    output_root: str) -> str:
    """Merge SS pose checkpoint paths into the reconstruction manifest.

    Returns the path to the merged manifest. Module 5 reads this to get
    checkpoint_path for each object (used as initial pose in ICP).
    """
    with open(recon_manifest_path, "r", encoding="utf-8") as f:
        recon = json.load(f)
    with open(ss_pose_manifest_path, "r", encoding="utf-8") as f:
        ss_pose = json.load(f)

    checkpoint_map = ss_pose.get("objects", {})
    merged_count = 0
    for obj in recon["objects"]:
        name = obj["name"]
        if name in checkpoint_map:
            obj["checkpoint_path"] = checkpoint_map[name]
            merged_count += 1
            print(f"  Merged SS pose for {name}")

    merged_path = os.path.join(output_root, "reconstruction_3d_with_ss_pose.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(recon, f, indent=2)

    print(f"  Merged {merged_count} SS pose checkpoints → {merged_path}")
    return merged_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full modular pipeline")
    parser.add_argument("--image", required=True, help="Path to input scene image")
    parser.add_argument("--output-dir", required=True, help="Root output directory")
    parser.add_argument("--trellis-version", choices=["1", "2"], default="1",
                        help="TRELLIS version (1 or 2)")
    parser.add_argument("--vlm-model", default="gemini-2.5-flash",
                        help="VLM model for object naming")
    parser.add_argument("--blender-command", default="/usr/local/bin/blender",
                        help="Path to Blender executable")
    parser.add_argument("--max-faces", type=int, default=0,
                        help="Max faces per reconstructed mesh (0 = no limit, e.g. 10000)")
    parser.add_argument("--sam-checkpoint", default=None,
                        help="SAM ViT-H checkpoint path")
    parser.add_argument("--skip-segment", action="store_true",
                        help="Skip segmentation (reuse existing)")
    parser.add_argument("--skip-recognize", action="store_true",
                        help="Skip recognition (reuse existing)")
    parser.add_argument("--skip-monodepth", action="store_true",
                        help="Skip mono depth (reuse existing)")
    parser.add_argument("--skip-3d-reconstruction", action="store_true",
                        help="Skip 3D reconstruction (reuse existing)")
    parser.add_argument("--skip-ss-pose", action="store_true",
                        help="Skip SS pose estimation (reuse existing)")
    parser.add_argument("--skip-2d3d-registration", action="store_true",
                        help="Skip 2D-3D registration (reuse existing)")
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    output_root = os.path.abspath(args.output_dir)

    # Sub-directories for each module
    dirs = {
        "segment": os.path.join(output_root, "segment"),
        "recognize": os.path.join(output_root, "recognize"),
        "monodepth": os.path.join(output_root, "monodepth"),
        "3d_reconstruction": os.path.join(output_root, "3d_reconstruction"),
        "ss_pose": os.path.join(output_root, "ss_pose"),
        "2d3d_registration": os.path.join(output_root, "2d3d_registration"),
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
    print(f"Max faces: {args.max_faces if args.max_faces > 0 else 'unlimited'}")
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
    # Module 4: 3DReconstruction
    # -----------------------------------------------------------------------
    recon_manifest = os.path.join(dirs["3d_reconstruction"], "reconstruction_3d_manifest.json")
    if not args.skip_3d_reconstruction:
        print("\n[4/5] 3D_RECONSTRUCTION")
        t0 = time.time()
        agent_python = get_python_path("agent")
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
            os.path.join(output_root, "3d_reconstruction.log"),
        )
        timings["3d_reconstruction"] = time.time() - t0
        if rc != 0:
            print(f"[4/5] 3D_RECONSTRUCTION FAILED (exit {rc})")
            sys.exit(1)
        print(f"[4/5] 3D_RECONSTRUCTION done ({timings['3d_reconstruction']:.1f}s)")
    else:
        print("\n[4/5] 3D_RECONSTRUCTION (skipped)")

    # -----------------------------------------------------------------------
    # Module 4b: SS Pose Estimation (TRELLIS2 only)
    # -----------------------------------------------------------------------
    # For TRELLIS2, objects lack built-in pose prediction. Module 4b uses
    # Meta's SS model to predict initial rotation/translation/scale.
    # For TRELLIS1, the SS pose is already captured in sam3d_batch_worker.
    ss_pose_manifest = os.path.join(dirs["ss_pose"], "ss_pose_manifest.json")
    recon_manifest_for_reg = recon_manifest  # default: pass reconstruction manifest as-is

    if args.trellis_version == "2" and not args.skip_ss_pose:
        print("\n[4b] SS_POSE (TRELLIS2)")
        t0 = time.time()
        agent_python = get_python_path("agent")
        rc = run_module(
            agent_python,
            os.path.join(modules_dir, "ss_pose.py"),
            [
                "--reconstruct-manifest", recon_manifest,
                "--recognize-manifest", rec_manifest,
                "--monodepth-manifest", depth_manifest,
                "--scene-image", image_path,
                "--output-dir", dirs["ss_pose"],
            ],
            os.path.join(output_root, "ss_pose.log"),
        )
        timings["ss_pose"] = time.time() - t0
        if rc != 0:
            print(f"[4b] SS_POSE FAILED (exit {rc})")
            # Non-fatal: registration will fall back to multi-start rotation
            print("[4b] Continuing with heuristic pose fallback...")
        else:
            print(f"[4b] SS_POSE done ({timings['ss_pose']:.1f}s)")
            # Merge SS pose checkpoints into reconstruction manifest for Module 5
            recon_manifest_for_reg = _merge_ss_pose(
                recon_manifest, ss_pose_manifest, output_root
            )
    elif args.trellis_version == "2":
        print("\n[4b] SS_POSE (skipped)")
        # Check if a previous run already merged the manifest
        merged_path = os.path.join(output_root, "reconstruction_3d_with_ss_pose.json")
        if os.path.exists(merged_path):
            recon_manifest_for_reg = merged_path
    else:
        # TRELLIS1: SS pose already in reconstruction checkpoint — no 4b needed
        pass

    # -----------------------------------------------------------------------
    # Module 5: 2D3DRegistration
    # -----------------------------------------------------------------------
    reg_manifest = os.path.join(dirs["2d3d_registration"], "registration_2d3d_manifest.json")
    if not args.skip_2d3d_registration:
        print("\n[5/5] 2D3D_REGISTRATION")
        t0 = time.time()
        agent_python = get_python_path("agent")
        rc = run_module(
            agent_python,
            os.path.join(modules_dir, "registration_2d3d.py"),
            [
                "--reconstruct-manifest", recon_manifest_for_reg,
                "--recognize-manifest", rec_manifest,
                "--monodepth-manifest", depth_manifest,
                "--output-dir", dirs["2d3d_registration"],
                "--blender-command", args.blender_command,
            ],
            os.path.join(output_root, "2d3d_registration.log"),
        )
        timings["2d3d_registration"] = time.time() - t0
        if rc != 0:
            print(f"[5/5] 2D3D_REGISTRATION FAILED (exit {rc})")
            sys.exit(1)
        print(f"[5/5] 2D3D_REGISTRATION done ({timings['2d3d_registration']:.1f}s)")
    else:
        print("\n[5/5] 2D3D_REGISTRATION (skipped)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    for module, t in timings.items():
        print(f"  {module:20s} {t:7.1f}s ({t/60:.1f} min)")
    print(f"  {'TOTAL':20s} {total_time:7.1f}s ({total_time/60:.1f} min)")
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
            "3d_reconstruction": recon_manifest,
            "ss_pose": ss_pose_manifest if args.trellis_version == "2" else None,
            "2d3d_registration": reg_manifest,
        },
    }
    summary_path = os.path.join(output_root, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
