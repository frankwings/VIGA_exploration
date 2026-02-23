"""Module 5: 2D3DRegistration (ICP Pose Alignment).

Aligns each TRELLIS mesh to the 2D image using MoGe pointmap and
layout_post_optimization (ICP + gradient refinement). Produces aligned GLBs,
per-object pose transforms with IoU scores, and a 3D-to-2D projection overlay
comparing the reconstructed scene against the original image.

This module is an orchestrator that calls pose_align_worker.py in the
sam3d_py311 conda environment, then renders the scene overlay via Blender.

Usage:
    python modules/registration_2d3d.py \\
        --reconstruct-manifest <output/3d_reconstruction/reconstruction_3d_manifest.json> \\
        --recognize-manifest <output/recognize/recognize_manifest.json> \\
        --monodepth-manifest <output/monodepth/monodepth_manifest.json> \\
        --output-dir <output/2d3d_registration/> \\
        --blender-command /usr/local/bin/blender

Output:
    registration_2d3d_manifest.json — manifest with per-object aligned poses + IoU
    object_transforms.json          — combined transforms for downstream use
    {name}.glb                      — aligned GLB meshes (with vertex colors)
    {name}_info.json                — per-object pose info
    viz/scene_render.png            — 3D render with vertex colors (RGBA)
    viz/projection_overlay.png      — vertex-color 3D render overlaid on original
    viz/side_by_side.png            — original vs vertex-color 3D render
    viz/flat_scene_render.png       — flat-shaded 3D render (distinct colors per object)
    viz/flat_projection_overlay.png — flat-shaded overlay on original
    viz/flat_side_by_side.png       — original vs flat-shaded 3D render
    viz/rotation_gifs/{name}_y_rotation.gif — per-object Y-axis turntable GIF
    viz/rotation_gifs/{name}_x_rotation.gif — per-object X-axis tumble GIF

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
        entry = {
            "name": name,
            "mesh": obj["mesh_path"],
            "mask": obj["npy_path"],
            "glb": os.path.join(output_dir, f"{name}.glb"),
            "info": os.path.join(output_dir, f"{name}_info.json"),
        }
        # Pass original textured GLB so alignment preserves textures
        if obj.get("canonical_glb"):
            entry["canonical_glb"] = obj["canonical_glb"]
        # Pass TRELLIS checkpoint for SS initial pose (if available)
        if obj.get("checkpoint_path"):
            entry["checkpoint"] = obj["checkpoint_path"]
        pose_manifest["objects"].append(entry)

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


def build_object_transforms(results: dict, output_dir: str,
                             min_iou: float = 0.15) -> str:
    """Build combined object_transforms.json from per-object info JSONs.

    Objects with IoU below min_iou are excluded from the scene render
    to prevent badly-aligned large meshes from blocking the camera view.
    """
    transforms = {}
    for name, info in results.items():
        if info is None:
            continue
        iou = info.get("iou", -1)
        if iou < min_iou:
            print(f"{TAG} Excluding {name} from scene render (IoU={iou:.4f} < {min_iou})")
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


def _run_blender_render(blender_cmd: str, render_script: str,
                        output_dir: str, moge_npz_path: str,
                        render_path: str, log_path: str,
                        extra_flags: list) -> bool:
    """Run a single Blender render with given flags. Returns True on success."""
    cmd = [
        blender_cmd, "-b", "-P", render_script, "--",
        output_dir, moge_npz_path, render_path,
    ] + extra_flags

    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            subprocess.run(
                cmd, cwd=str(ROOT), text=True, check=True,
                stdout=lf, stderr=subprocess.STDOUT,
            )
        return os.path.exists(render_path)
    except Exception as e:
        print(f"{TAG} Blender render failed: {e}")
        return False


def _make_overlay_images(render_path: str, scene_image: str, viz_dir: str,
                         prefix: str = "") -> None:
    """Create overlay and side-by-side images from a render."""
    from PIL import Image

    render_img = Image.open(render_path).convert("RGBA")
    target_img = Image.open(scene_image).convert("RGB")

    tw, th = target_img.size
    render_img = render_img.resize((tw, th), Image.LANCZOS)

    # Projection overlay: blend 3D render on top of original
    target_rgba = target_img.copy().convert("RGBA")
    overlay = Image.alpha_composite(target_rgba, render_img)
    overlay_path = os.path.join(viz_dir, f"{prefix}projection_overlay.png")
    overlay.convert("RGB").save(overlay_path)
    print(f"{TAG} Overlay: {overlay_path}")

    # Side-by-side comparison
    render_rgb = Image.new("RGB", (tw, th), (255, 255, 255))
    render_rgb.paste(render_img, mask=render_img.split()[3])
    comparison = Image.new("RGB", (tw * 2, th))
    comparison.paste(target_img, (0, 0))
    comparison.paste(render_rgb, (tw, 0))
    sbs_path = os.path.join(viz_dir, f"{prefix}side_by_side.png")
    comparison.save(sbs_path)
    print(f"{TAG} Side-by-side: {sbs_path}")


def render_scene_overlay(output_dir: str, scene_image: str,
                         depth_manifest: dict, blender_cmd: str) -> None:
    """Render 3D scene with Blender and create overlay comparisons.

    Produces two sets of visualizations:
      1. Textured render (native PBR materials from TRELLIS GLBs):
         viz/scene_render.png, viz/projection_overlay.png, viz/side_by_side.png
      2. Flat-shaded render (distinct colors per object, for alignment eval):
         viz/flat_scene_render.png, viz/flat_projection_overlay.png,
         viz/flat_side_by_side.png
    """
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    transforms_path = os.path.join(output_dir, "object_transforms.json")
    if not os.path.exists(transforms_path):
        print(f"{TAG} No object_transforms.json, skipping scene render")
        return

    # Get intrinsics from monodepth manifest (normalized) and convert to pixels.
    # MoGe's normalized intrinsics (fx_norm, fy_norm) convert to pixel space as:
    #   fx_px = fx_norm * width, fy_px = fy_norm * height
    # For a single physical focal length, fx_norm*W ≈ fy_norm*H (isotropic in pixels).
    intrinsics = depth_manifest.get("intrinsics")
    pm_shape = depth_manifest.get("pointmap_shape", [0, 0])
    if not intrinsics or pm_shape == [0, 0]:
        print(f"{TAG} No intrinsics/pointmap_shape, skipping scene render")
        return

    pm_h, pm_w = pm_shape
    fx = intrinsics[0][0] * pm_w
    fy = intrinsics[1][1] * pm_h
    cx = intrinsics[0][2] * pm_w
    cy = intrinsics[1][2] * pm_h
    print(f"{TAG} Render intrinsics: fx_px={fx:.1f} fy_px={fy:.1f} "
          f"cx={cx:.1f} cy={cy:.1f} ({pm_w}x{pm_h})")

    # Save intrinsics as NPZ for render_full_scene.py
    moge_npz_path = os.path.join(output_dir, "moge_intrinsics.npz")
    intrinsics_px = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    np.savez(moge_npz_path, intrinsics_px=intrinsics_px,
             image_width=pm_w, image_height=pm_h)

    render_script = str(ROOT / "render_full_scene.py")

    # Render 1: Textured (native PBR materials from TRELLIS GLBs)
    render_path = os.path.join(viz_dir, "scene_render.png")
    log_path = os.path.join(output_dir, "blender_scene.log")
    print(f"{TAG} Rendering textured 3D scene...")
    if _run_blender_render(blender_cmd, render_script, output_dir,
                           moge_npz_path, render_path, log_path,
                           []):
        _make_overlay_images(render_path, scene_image, viz_dir)
    else:
        print(f"{TAG} Textured render failed or produced no output")

    # Render 2: Flat-shaded (distinct colors per object)
    flat_render_path = os.path.join(viz_dir, "flat_scene_render.png")
    flat_log_path = os.path.join(output_dir, "blender_scene_flat.log")
    print(f"{TAG} Rendering flat-shaded scene...")
    if _run_blender_render(blender_cmd, render_script, output_dir,
                           moge_npz_path, flat_render_path, flat_log_path,
                           ["--flat"]):
        _make_overlay_images(flat_render_path, scene_image, viz_dir, prefix="flat_")
    else:
        print(f"{TAG} Flat render failed or produced no output")


def render_rotation_gifs(results: dict, output_dir: str,
                         blender_cmd: str, frames: int = 24,
                         resolution: int = 512) -> None:
    """Render per-object rotation GIFs (Y turntable + X tumble).

    For each aligned GLB, calls blender_render_rotation.py to render frames,
    then creates ping-pong GIFs using PIL.
    """
    from PIL import Image

    gifs_dir = os.path.join(output_dir, "viz", "rotation_gifs")
    os.makedirs(gifs_dir, exist_ok=True)

    render_script = str(ROOT / "tools" / "blender_render_rotation.py")

    for name, info in results.items():
        if info is None:
            continue
        glb_path = os.path.join(output_dir, f"{name}.glb")
        if not os.path.exists(glb_path):
            print(f"{TAG} No GLB for {name}, skipping rotation render")
            continue

        frames_dir = os.path.join(gifs_dir, f"{name}_frames")
        os.makedirs(frames_dir, exist_ok=True)

        cmd = [
            blender_cmd, "--background", "--python", render_script, "--",
            glb_path, frames_dir,
            "--frames", str(frames),
            "--resolution", str(resolution),
        ]
        log_path = os.path.join(gifs_dir, f"{name}_render.log")

        print(f"{TAG} Rendering rotation: {name}...")
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                subprocess.run(
                    cmd, cwd=str(ROOT), text=True, check=True,
                    stdout=lf, stderr=subprocess.STDOUT,
                )
        except Exception as e:
            print(f"{TAG} Rotation render failed for {name}: {e}")
            continue

        # Create ping-pong GIFs (forward + reverse) for both axes
        basename = Path(glb_path).stem
        for axis in ("y", "x"):
            frame_files = sorted(Path(frames_dir).glob(f"{basename}_{axis}_*.png"))
            if not frame_files:
                print(f"{TAG} No {axis}-frames for {name}")
                continue

            images = [Image.open(f) for f in frame_files]
            pingpong = images + images[-2:0:-1]
            gif_path = os.path.join(gifs_dir, f"{name}_{axis}_rotation.gif")
            pingpong[0].save(
                gif_path, save_all=True, append_images=pingpong[1:],
                duration=80, loop=0,
            )
            print(f"{TAG} GIF: {gif_path} ({len(pingpong)} frames)")


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
    parser.add_argument("--blender-command", default="/usr/local/bin/blender",
                        help="Path to Blender executable (for scene overlay)")
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
        obj_entry = {
            "name": name,
            "mesh_path": recon_obj["mesh_path"],
            "npy_path": rec_obj["npy_path"],
        }
        # Pass canonical (pre-transform, Z-up) GLB for texture-preserving alignment
        if recon_obj.get("canonical_glb_path"):
            obj_entry["canonical_glb"] = recon_obj["canonical_glb_path"]
        objects.append(obj_entry)

    print(f"{TAG} Scene image: {scene_image}")
    print(f"{TAG} {len(objects)} objects to align")

    # Step 1: Run pose alignment (ICP + gradient refinement)
    results = run_pose_alignment(scene_image, objects, output_dir)

    # Step 2: Build combined transforms
    build_object_transforms(results, output_dir)

    # Step 3: Render 3D-to-2D projection overlay
    render_scene_overlay(output_dir, scene_image, depth_manifest,
                         args.blender_command)

    # Step 4: Render per-object rotation GIFs (Y turntable + X tumble)
    render_rotation_gifs(results, output_dir, args.blender_command)

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
