"""Alignment diagnostic: run MoGe on each segment, render GLB with MoGe camera, compare side-by-side.

Usage:
    C:/Users/kingy/miniconda3/envs/sam3d_py311/python.exe diagnose_alignment.py

Orchestrates:
1. MoGe inference on each segmented PNG (same Python env since sam3d_py311 has MoGe + compatible CUDA)
2. Create corrected GLB files (add missing translation, convert to OpenCV camera space)
3. Blender headless render of each corrected GLB using MoGe intrinsics
4. Side-by-side comparison image generation

Uses sam3d_py311 env (has MoGe + trimesh + torch with CUDA 12.8 support for RTX 5080).
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import trimesh
from PIL import Image


# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAM_INIT_DIR = os.path.join(
    PROJECT_ROOT,
    "output", "static_scene", "20260210_043534", "greentea", "sam_init"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "alignment_diagnostic")
BLENDER = "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe"
RENDER_SCRIPT = os.path.join(PROJECT_ROOT, "diagnose_render_glb.py")

OBJECTS = [
    "green_tea_bottle",
    "green_tea_bottle_1",
    "alienware_keyboard",
    "alienware_keyboard_1",
    "envelope",
    "headphones",
]


def run_moge(png_path, npz_path):
    """Run MoGe directly (we're already in sam3d_py311 env)."""
    from moge.model.v1 import MoGeModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a module-level cache for the model
    if not hasattr(run_moge, "_model"):
        print("[MoGe] Loading model...")
        run_moge._model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
        run_moge._model.eval()

    model = run_moge._model

    img = Image.open(png_path).convert("RGB")
    W, H = img.size
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model.infer(img_tensor, force_projection=False)

    points = output["points"].cpu().numpy()
    intrinsics = output["intrinsics"].cpu().numpy()
    depth = output.get("depth")
    if depth is not None:
        depth = depth.cpu().numpy()

    fx_norm, fy_norm = intrinsics[0, 0], intrinsics[1, 1]
    cx_norm, cy_norm = intrinsics[0, 2], intrinsics[1, 2]
    fx_px, fy_px = fx_norm * W, fy_norm * H
    cx_px, cy_px = cx_norm * W, cy_norm * H

    intrinsics_px = np.array([
        [fx_px, 0.0, cx_px],
        [0.0, fy_px, cy_px],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    finite_mask = np.isfinite(points).all(axis=-1)
    pts_f = points[finite_mask]
    print(f"  MoGe: {W}x{H}, fx={fx_px:.1f} fy={fy_px:.1f}")
    print(f"  Pointmap (finite): X=[{pts_f[:,0].min():.3f},{pts_f[:,0].max():.3f}] "
          f"Y=[{pts_f[:,1].min():.3f},{pts_f[:,1].max():.3f}] "
          f"Z=[{pts_f[:,2].min():.3f},{pts_f[:,2].max():.3f}]")

    save_dict = {
        "points": points,
        "intrinsics_norm": intrinsics,
        "intrinsics_px": intrinsics_px,
        "image_width": W,
        "image_height": H,
    }
    if depth is not None:
        save_dict["depth"] = depth

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path, **save_dict)
    return True


def create_corrected_glb(glb_path, transform_data, output_glb_path):
    """Create a corrected GLB with vertices in OpenCV camera space.

    The existing GLB has TWO bugs from sam3d_worker.py:
    1. Wrong pre-transform: uses R_flip_z @ R_yup_to_zup = [[-1,0,0],[0,0,1],[0,-1,0]]
       which maps [x,y,z] -> [-x,-z,y]. The correct pre (from layout_post_optimization_utils
       get_mesh) is [[1,0,0],[0,0,1],[0,-1,0]] mapping [x,y,z] -> [x,z,-y].
       The broken pre is the NEGATION of the correct pre.
    2. Translation NOT applied (Transform3d.translate stores T in column 3
       instead of row 3, so T goes to w-component instead of xyz).

    Since broken_pre = -correct_pre:
        v_broken = v_raw @ broken_pre @ SR = -(v_raw @ correct_pre @ SR)
    And the correct result in PT3D camera space is:
        v_correct_pt3d = v_raw @ correct_pre @ SR + T = -v_broken + T

    To get OpenCV camera space: negate X and Y from PT3D.

    Args:
        glb_path: Path to the broken GLB
        transform_data: Dict with 'translation', 'rotation', 'scale'
        output_glb_path: Where to save the corrected GLB
    """
    scene = trimesh.load(glb_path)
    if hasattr(scene, 'geometry'):
        meshes = list(scene.geometry.values())
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        mesh = scene

    v = torch.tensor(mesh.vertices, dtype=torch.float32)
    T = torch.tensor(transform_data['translation'], dtype=torch.float32)

    print(f"  Broken GLB center: ({v.mean(0)[0]:.3f}, {v.mean(0)[1]:.3f}, {v.mean(0)[2]:.3f})")

    # Correct PT3D: -v_broken + T
    v_pt3d = -v + T

    # Convert PT3D (X-left, Y-up, Z-fwd) to OpenCV (X-right, Y-down, Z-fwd): negate X, Y
    v_opencv = v_pt3d.clone()
    v_opencv[:, 0] = -v_pt3d[:, 0]
    v_opencv[:, 1] = -v_pt3d[:, 1]

    print(f"  Corrected PT3D center: ({v_pt3d.mean(0)[0]:.3f}, {v_pt3d.mean(0)[1]:.3f}, {v_pt3d.mean(0)[2]:.3f})")
    print(f"  Corrected OpenCV center: ({v_opencv.mean(0)[0]:.3f}, {v_opencv.mean(0)[1]:.3f}, {v_opencv.mean(0)[2]:.3f})")
    print(f"    X=[{v_opencv[:,0].min():.3f}, {v_opencv[:,0].max():.3f}]")
    print(f"    Y=[{v_opencv[:,1].min():.3f}, {v_opencv[:,1].max():.3f}]")
    print(f"    Z=[{v_opencv[:,2].min():.3f}, {v_opencv[:,2].max():.3f}]")

    mesh.vertices = v_opencv.numpy().astype(np.float32)
    os.makedirs(os.path.dirname(output_glb_path), exist_ok=True)
    mesh.export(output_glb_path)
    return True


def run_blender_render(glb_path, npz_path, output_png):
    """Render GLB with MoGe camera intrinsics using Blender headless."""
    cmd = [
        BLENDER, "-b", "-P", RENDER_SCRIPT,
        "--", glb_path, npz_path, output_png,
    ]
    print(f"  Blender render...")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, encoding="utf-8") as f_out:
        out_path = f_out.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, encoding="utf-8") as f_err:
        err_path = f_err.name

    try:
        with open(out_path, "w", encoding="utf-8") as fo, open(err_path, "w", encoding="utf-8") as fe:
            result = subprocess.run(cmd, stdout=fo, stderr=fe, timeout=120)

        with open(out_path, "r", encoding="utf-8") as fo:
            stdout = fo.read()

        for line in stdout.split("\n"):
            if "[INFO]" in line and ("bounds" in line.lower() or "camera" in line.lower()):
                print(f"    {line.strip()}")

        if result.returncode != 0:
            with open(err_path, "r", encoding="utf-8") as fe:
                stderr = fe.read()
            print(f"  [ERROR] Blender failed (rc={result.returncode})")
            for line in stderr.strip().split("\n")[-5:]:
                print(f"    {line}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  [ERROR] Blender timed out")
        return False
    finally:
        for p in [out_path, err_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def make_comparison(input_png, render_png, output_png, label):
    """Create side-by-side [2D input | 3D render] comparison."""
    try:
        img_2d = Image.open(input_png).convert("RGBA")
        img_3d = Image.open(render_png).convert("RGBA")
    except Exception as e:
        print(f"  [WARN] Cannot create comparison: {e}")
        return

    target_h = 512
    w1 = int(img_2d.width * target_h / img_2d.height)
    w2 = int(img_3d.width * target_h / img_3d.height)
    img_2d = img_2d.resize((w1, target_h), Image.LANCZOS)
    img_3d = img_3d.resize((w2, target_h), Image.LANCZOS)

    bg1 = Image.new("RGB", (w1, target_h), (255, 255, 255))
    bg1.paste(img_2d, mask=img_2d.split()[3])
    bg2 = Image.new("RGB", (w2, target_h), (240, 240, 240))
    bg2.paste(img_3d, mask=img_3d.split()[3])

    gap = 4
    total_w = w1 + gap + w2
    label_h = 30
    canvas = Image.new("RGB", (total_w, target_h + label_h), (255, 255, 255))
    canvas.paste(bg1, (0, label_h))
    canvas.paste(bg2, (w1 + gap, label_h))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        draw.text((4, 4), f"2D Input: {label}", fill=(0, 0, 0), font=font)
        draw.text((w1 + gap + 4, 4), "3D Render (MoGe cam, fixed)", fill=(0, 0, 0), font=font)
    except ImportError:
        pass

    canvas.save(output_png)
    print(f"  Comparison: {output_png}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    transforms_path = os.path.join(SAM_INIT_DIR, "object_transforms.json")
    with open(transforms_path, "r", encoding="utf-8") as f:
        transforms = json.load(f)

    print("=" * 60)
    print("ALIGNMENT DIAGNOSTIC")
    print("=" * 60)
    print(f"Data: {SAM_INIT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    results = {}

    for i, obj_name in enumerate(OBJECTS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(OBJECTS)}] {obj_name}")
        print(f"{'='*60}")

        png_path = os.path.join(SAM_INIT_DIR, f"{obj_name}.png")
        glb_path = os.path.join(SAM_INIT_DIR, f"{obj_name}.glb")
        npz_path = os.path.join(OUTPUT_DIR, f"{obj_name}_moge.npz")
        corrected_glb = os.path.join(OUTPUT_DIR, f"{obj_name}_corrected.glb")
        render_path = os.path.join(OUTPUT_DIR, f"{obj_name}_render.png")
        compare_path = os.path.join(OUTPUT_DIR, f"{obj_name}_compare.png")

        if not os.path.exists(png_path) or not os.path.exists(glb_path):
            print(f"  [SKIP] Missing PNG or GLB")
            continue

        t = transforms[i]
        print(f"  Translation: {t['translation']}")
        print(f"  Scale: {t['scale']}")

        # Step 1: Run MoGe
        print("\n  --- MoGe ---")
        moge_ok = run_moge(png_path, npz_path)
        if not moge_ok:
            results[obj_name] = {"status": "moge_failed"}
            continue

        # Step 2: Create corrected GLB
        print("\n  --- Correcting GLB ---")
        create_corrected_glb(glb_path, t, corrected_glb)

        # Step 3: Render in Blender
        print("\n  --- Blender Render ---")
        render_ok = run_blender_render(corrected_glb, npz_path, render_path)
        if not render_ok:
            results[obj_name] = {"status": "render_failed"}
            continue

        # Step 4: Side-by-side comparison
        make_comparison(png_path, render_path, compare_path, obj_name)
        results[obj_name] = {"status": "ok"}

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "diagnostic_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {OUTPUT_DIR}")
    for name, r in results.items():
        print(f"  {name}: {r['status']}")


if __name__ == "__main__":
    main()
