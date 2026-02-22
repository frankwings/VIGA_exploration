"""Module 5: 2D-3D Registration (Pose Alignment).

Aligns each TRELLIS mesh to the 2D image using MoGe pointmap and
layout_post_optimization (ICP + gradient refinement).

This module is an orchestrator that calls pose_align_worker.py in the
sam3d_py311 conda environment, then optionally renders the full scene
comparison using Blender.

Usage:
    python modules/register.py \
        --reconstruct-manifest <output/reconstruct/reconstruct_manifest.json> \
        --recognize-manifest <output/recognize/recognize_manifest.json> \
        --monodepth-manifest <output/monodepth/monodepth_manifest.json> \
        --output-dir <output/register/> \
        --blender-command /usr/local/bin/blender

Output:
    register_manifest.json        — manifest with per-object aligned poses
    object_transforms.json        — combined transforms for Blender scene
    {name}.glb                    — aligned GLB meshes
    {name}_info.json              — per-object pose info
    viz/full_scene_comparison.png — target vs 3D render side-by-side
    viz/full_scene_render.png     — 3D render only
    viz/{name}_y_rotation.gif     — per-object turntable GIF
    viz/{name}_x_rotation.gif     — per-object tumble GIF

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
    then aligns each object using layout_post_optimization.
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

    print(f"[REGISTER] Running pose alignment ({len(objects)} objects)...")
    print(f"[REGISTER] Log: {log_path}")

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
        print(f"[REGISTER] Pose alignment FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        print(f"[REGISTER] Check log: {log_path}")

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
            print(f"[REGISTER] {name}: IoU={iou}, GLB {glb_mb:.1f}MB")
        else:
            results[name] = None
            print(f"[REGISTER] {name}: FAILED (no info JSON)")

    print(f"[REGISTER] Pose alignment completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
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
    print(f"[REGISTER] Object transforms: {path}")
    return path


def render_full_scene(output_dir: str, scene_image: str,
                      blender_cmd: str) -> None:
    """Render the full scene comparison using Blender + render_full_scene.py."""
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    transforms_path = os.path.join(output_dir, "object_transforms.json")
    if not os.path.exists(transforms_path):
        print("[REGISTER] No object_transforms.json, skipping scene render")
        return

    # Read transforms to get intrinsics
    with open(transforms_path, "r", encoding="utf-8") as f:
        transforms = json.load(f)

    intrinsics = transforms.get("_intrinsics")
    pm_shape = transforms.get("_pointmap_shape", [0, 0])
    if not intrinsics:
        print("[REGISTER] No intrinsics in transforms, skipping scene render")
        return

    # Save intrinsics as NPZ for render_full_scene.py
    pm_h, pm_w = pm_shape
    fx = intrinsics[0][0] * pm_w
    fy = intrinsics[1][1] * pm_h
    cx = intrinsics[0][2] * pm_w
    cy = intrinsics[1][2] * pm_h

    moge_npz_path = os.path.join(output_dir, "moge_intrinsics.npz")
    np.savez(moge_npz_path, fx=fx, fy=fy, cx=cx, cy=cy, width=pm_w, height=pm_h)

    render_path = os.path.join(viz_dir, "full_scene_render.png")
    render_script = str(ROOT / "render_full_scene.py")
    log_path = os.path.join(output_dir, "blender_scene.log")

    cmd = [
        blender_cmd, "-b", "-P", render_script, "--",
        output_dir, moge_npz_path, render_path,
    ]

    print("[REGISTER] Rendering full scene with Blender...")
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            subprocess.run(
                cmd, cwd=str(ROOT), text=True, check=True,
                stdout=lf, stderr=subprocess.STDOUT,
            )
        print(f"[REGISTER] Scene render: {render_path}")

        # Create side-by-side comparison
        from PIL import Image
        if os.path.exists(render_path) and os.path.exists(scene_image):
            target = Image.open(scene_image).convert("RGB")
            render = Image.open(render_path).convert("RGB")

            # Match heights
            th, tw = target.size[1], target.size[0]
            render = render.resize((tw, th), Image.LANCZOS)

            comparison = Image.new("RGB", (tw * 2, th))
            comparison.paste(target, (0, 0))
            comparison.paste(render, (tw, 0))

            comp_path = os.path.join(viz_dir, "full_scene_comparison.png")
            comparison.save(comp_path)
            print(f"[REGISTER] Comparison: {comp_path}")
    except Exception as e:
        print(f"[REGISTER] Scene render failed: {e}")


def render_rotation_gifs(output_dir: str, object_names: list,
                         blender_cmd: str) -> None:
    """Render per-object rotation GIFs using blender_render_rotation.py."""
    viz_dir = os.path.join(output_dir, "viz")
    rotation_script = str(ROOT / "tools" / "blender_render_rotation.py")

    for name in object_names:
        glb_path = os.path.join(output_dir, f"{name}.glb")
        if not os.path.exists(glb_path):
            continue

        frames_dir = os.path.join(viz_dir, f"{name}_frames")
        os.makedirs(frames_dir, exist_ok=True)

        log_path = os.path.join(output_dir, f"rotation_{name}.log")
        cmd = [
            blender_cmd, "-b", "-P", rotation_script, "--",
            glb_path, frames_dir,
            "--frames", "24",
            "--resolution", "512",
        ]

        print(f"[REGISTER] Rendering rotation: {name}...")
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                subprocess.run(
                    cmd, cwd=str(ROOT), text=True, check=True,
                    stdout=lf, stderr=subprocess.STDOUT,
                )

            # Create GIFs from frames
            _create_pingpong_gif(frames_dir, f"{name}_y_*.png",
                                 os.path.join(viz_dir, f"{name}_y_rotation.gif"))
            _create_pingpong_gif(frames_dir, f"{name}_x_*.png",
                                 os.path.join(viz_dir, f"{name}_x_rotation.gif"))
        except Exception as e:
            print(f"[REGISTER] Rotation render failed for {name}: {e}")


def _create_pingpong_gif(frame_dir: str, pattern: str, output_path: str,
                         duration: int = 80) -> None:
    """Create ping-pong GIF from frame directory."""
    from pathlib import Path as P
    from PIL import Image

    frames = sorted(P(frame_dir).glob(pattern))
    if not frames:
        return

    images = [Image.open(f) for f in frames]
    pingpong = images + images[-2:0:-1]

    pingpong[0].save(
        str(output_path), save_all=True, append_images=pingpong[1:],
        duration=duration, loop=0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Module 5: 2D-3D Registration")
    parser.add_argument("--reconstruct-manifest", required=True,
                        help="Path to reconstruct_manifest.json from Module 4")
    parser.add_argument("--recognize-manifest", required=True,
                        help="Path to recognize_manifest.json from Module 2")
    parser.add_argument("--monodepth-manifest", required=True,
                        help="Path to monodepth_manifest.json from Module 3")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--blender-command", default="/usr/local/bin/blender",
                        help="Path to Blender executable")
    parser.add_argument("--skip-scene-render", action="store_true",
                        help="Skip Blender scene rendering")
    parser.add_argument("--skip-rotation-gifs", action="store_true",
                        help="Skip per-object rotation GIF rendering")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

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
            print(f"[REGISTER] WARNING: {name} not in recognize manifest, skipping")
            continue
        objects.append({
            "name": name,
            "mesh_path": recon_obj["mesh_path"],
            "npy_path": rec_obj["npy_path"],
        })

    print(f"[REGISTER] Scene image: {scene_image}")
    print(f"[REGISTER] {len(objects)} objects to align")

    # Step 1: Run pose alignment
    results = run_pose_alignment(scene_image, objects, output_dir)

    # Step 2: Build combined transforms
    build_object_transforms(results, output_dir)

    # Step 3: Render full scene (optional)
    if not args.skip_scene_render:
        render_full_scene(output_dir, scene_image, args.blender_command)

    # Step 4: Render rotation GIFs (optional)
    successful_names = [n for n, r in results.items() if r is not None]
    if not args.skip_rotation_gifs and successful_names:
        render_rotation_gifs(output_dir, successful_names, args.blender_command)

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
    manifest_path = os.path.join(output_dir, "register_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    success = len(obj_entries)
    total = len(objects)
    avg_iou = np.mean([o["iou"] for o in obj_entries if o["iou"] > 0]) if obj_entries else 0

    print(f"\n[REGISTER] Done: {success}/{total} objects aligned")
    print(f"[REGISTER] Average IoU: {avg_iou:.2f}")
    print(f"[REGISTER] Manifest: {manifest_path}")
    for o in obj_entries:
        print(f"  {o['name']}: IoU={o['iou']:.4f}")


if __name__ == "__main__":
    main()
