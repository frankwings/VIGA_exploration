"""Regenerate comparison visualization for original vs VIGA SAM3D experiment.

Run this after both experiment runs are complete.
Usage:
    C:/Users/kingy/miniconda3/envs/agent/python.exe run_experiment_compare_only.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).parent.resolve()

# target_resized.jpg (771x1024) matches mask dimensions
TARGET_IMAGE = PROJECT_ROOT / "data" / "static_scene" / "dining" / "target_resized.jpg"
MASK_FILE    = PROJECT_ROOT / "output" / "sam3d_dining" / "wooden_chair.npy"
MOGE_NPZ     = PROJECT_ROOT / "output" / "sam3d_dining" / "target_moge.npz"
OUT_ORIGINAL = PROJECT_ROOT / "output" / "experiment_original_sam3d"
OUT_VIGA     = PROJECT_ROOT / "output" / "experiment_viga_sam3d"
OUT_CMP      = PROJECT_ROOT / "output" / "experiment_comparison"


def load_moge_intrinsics(npz_path: Path) -> dict | None:
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    K = data["intrinsics_px"].astype(np.float64)
    return {
        "K": K,
        "fx": K[0, 0], "fy": K[1, 1],
        "cx": K[0, 2], "cy": K[1, 2],
        "width": int(data["image_width"]),
        "height": int(data["image_height"]),
    }


def project_glb_onto_image(
    glb_path: Path,
    info_path: Path,
    scene_img: Image.Image,
    cam: dict,
) -> Image.Image:
    """Project GLB mesh vertices (in PyTorch3D camera space) onto scene image."""
    try:
        import trimesh
    except ImportError:
        return _label_image(scene_img.copy(), "trimesh not available", "orange")

    if not glb_path.exists():
        return _label_image(scene_img.copy(), "GLB not found\n(run failed?)", "red")
    if not info_path.exists():
        return _label_image(scene_img.copy(), "info.json missing", "red")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    T = info.get("translation", [0, 0, 0])
    R = info.get("rotation", [1, 0, 0, 0])
    has_nan = any(np.isnan(v) or np.isinf(v) for v in (T + R if isinstance(T, list) else []))
    if has_nan:
        return _label_image(
            scene_img.copy(),
            f"NaN/Inf detected in pose!\nT={[f'{v:.3f}' for v in T[:3]]}",
            "red",
        )

    # Load mesh
    try:
        scene = trimesh.load(str(glb_path), force="scene")
        if isinstance(scene, trimesh.Scene):
            mesh = trimesh.util.concatenate([
                g for g in scene.geometry.values() if hasattr(g, "vertices")
            ])
        else:
            mesh = scene
        verts = np.array(mesh.vertices, dtype=np.float64)
    except Exception as e:
        return _label_image(scene_img.copy(), f"Failed loading GLB:\n{e}", "red")

    if len(verts) == 0:
        return _label_image(scene_img.copy(), "Empty mesh", "orange")

    # Subsample for speed
    if len(verts) > 5000:
        idx = np.random.choice(len(verts), 5000, replace=False)
        verts = verts[idx]

    # GLB vertices are in PyTorch3D camera space (X-left, Y-up, Z-forward)
    # Convert to OpenCV space (X-right, Y-down, Z-forward)
    x_ocv = -verts[:, 0]
    y_ocv = -verts[:, 1]
    z_ocv =  verts[:, 2]

    valid = z_ocv > 0.01
    x_ocv, y_ocv, z_ocv = x_ocv[valid], y_ocv[valid], z_ocv[valid]

    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    u = (fx * x_ocv / z_ocv + cx).astype(np.int32)
    v = (fy * y_ocv / z_ocv + cy).astype(np.int32)

    W, H = scene_img.size
    in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[in_frame], v[in_frame]

    result = scene_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for ux, vy in zip(u, v):
        draw.ellipse([ux-3, vy-3, ux+3, vy+3], fill=(0, 200, 255, 220))
    result = Image.alpha_composite(result, overlay).convert("RGB")

    # Draw IoU if available
    iou = info.get("final_iou")
    if iou is not None:
        draw2 = ImageDraw.Draw(result)
        draw2.rectangle([0, 0, 120, 25], fill=(0, 0, 0))
        draw2.text((5, 5), f"IoU: {iou:.3f}", fill=(255, 255, 0))

    return result


def make_mask_overlay(scene_img: Image.Image, mask_npy: np.ndarray) -> Image.Image:
    """Overlay SAM mask (255=object) on scene image (green)."""
    obj_mask = mask_npy > 0  # 255 = object
    img_arr = np.array(scene_img.convert("RGB"), dtype=np.float32)
    overlay = img_arr.copy()
    overlay[obj_mask, 0] = 0
    overlay[obj_mask, 1] = 200
    overlay[obj_mask, 2] = 0
    blended = (0.6 * img_arr + 0.4 * overlay).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(blended)
    draw = ImageDraw.Draw(result)
    pct = 100.0 * obj_mask.sum() / obj_mask.size
    draw.rectangle([0, 0, 220, 25], fill=(0, 0, 0))
    draw.text((5, 5), f"SAM mask: {pct:.1f}% coverage", fill=(255, 255, 0))
    return result


def _label_image(img: Image.Image, text: str, color: str = "white") -> Image.Image:
    colors = {"red": (255, 80, 80), "orange": (255, 160, 0), "white": (255, 255, 255)}
    c = colors.get(color, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        draw.rectangle([0, i*20, img.width, i*20+20], fill=(0, 0, 0))
        draw.text((5, i*20+3), line, fill=c)
    return img


def _iou_label(info_path: Path) -> str:
    if not info_path.exists():
        return "FAILED (no info.json)"
    with open(info_path) as f:
        info = json.load(f)
    T = info.get("translation", [])
    R = info.get("rotation", [])
    all_vals = (T if isinstance(T, list) else []) + (R if isinstance(R, list) else [])
    if any(np.isnan(v) or np.isinf(v) for v in all_vals):
        return "NaN DETECTED"
    iou = info.get("final_iou")
    return f"IoU={iou:.3f}" if iou is not None else "OK (no IoU recorded)"


def main():
    scene_img_full = Image.open(str(TARGET_IMAGE)).convert("RGB")
    mask_npy = np.load(str(MASK_FILE))

    cam = load_moge_intrinsics(MOGE_NPZ)
    if cam is None:
        H, W = mask_npy.shape
        f = max(W, H) * 0.8
        cam = {"K": None, "fx": f, "fy": f, "cx": W/2, "cy": H/2, "width": W, "height": H}
        print(f"Warning: No MoGe NPZ, using estimated intrinsics")
    else:
        print(f"MoGe intrinsics: fx={cam['fx']:.1f} fy={cam['fy']:.1f} "
              f"({cam['width']}x{cam['height']})")

    # Resize scene image to MoGe resolution, then to display size
    moge_w, moge_h = cam["width"], cam["height"]
    TARGET_W = 600
    disp_scale = TARGET_W / moge_w
    disp_h = int(moge_h * disp_scale)

    scene_moge = scene_img_full.resize((moge_w, moge_h), Image.LANCZOS)
    scene_disp = scene_moge.resize((TARGET_W, disp_h), Image.LANCZOS)

    # Resize mask to display size
    mask_pil = Image.fromarray((mask_npy > 0).astype(np.uint8) * 255, mode="L")
    mask_pil = mask_pil.resize((TARGET_W, disp_h), Image.NEAREST)
    mask_rsz = np.array(mask_pil) > 0

    cam_disp = {
        "fx": cam["fx"] * disp_scale,
        "fy": cam["fy"] * disp_scale,
        "cx": cam["cx"] * disp_scale,
        "cy": cam["cy"] * disp_scale,
    }

    # Generate visualizations
    mask_vis = make_mask_overlay(scene_disp, mask_rsz)
    orig_proj = project_glb_onto_image(
        OUT_ORIGINAL / "wooden_chair.glb",
        OUT_ORIGINAL / "wooden_chair_info.json",
        scene_disp, cam_disp,
    )
    viga_proj = project_glb_onto_image(
        OUT_VIGA / "wooden_chair.glb",
        OUT_VIGA / "wooden_chair_info.json",
        scene_disp, cam_disp,
    )

    # Save per-run projections
    OUT_ORIGINAL.mkdir(parents=True, exist_ok=True)
    OUT_VIGA.mkdir(parents=True, exist_ok=True)
    mask_vis.save(str(OUT_ORIGINAL / "wooden_chair_mask.png"))
    mask_vis.save(str(OUT_VIGA / "wooden_chair_mask.png"))
    orig_proj.save(str(OUT_ORIGINAL / "wooden_chair_proj.png"))
    viga_proj.save(str(OUT_VIGA / "wooden_chair_proj.png"))

    orig_lbl = _iou_label(OUT_ORIGINAL / "wooden_chair_info.json")
    viga_lbl = _iou_label(OUT_VIGA / "wooden_chair_info.json")

    print(f"Original SAM3D: {orig_lbl}")
    print(f"VIGA SAM3D:     {viga_lbl}")

    # Build side-by-side comparison
    PAD = 8
    W, H = TARGET_W, disp_h
    images = [mask_vis, orig_proj, viga_proj]
    labels = [
        "SAM mask (green = wooden_chair)",
        f"Original Meta SAM3D\n{orig_lbl}",
        f"VIGA (--scene-image + v9 mask)\n{viga_lbl}",
    ]
    n = len(images)
    cmp_w = n * W + (n + 1) * PAD
    cmp_h = H + 80
    cmp = Image.new("RGB", (cmp_w, cmp_h), (20, 20, 20))
    draw = ImageDraw.Draw(cmp)
    draw.text((PAD, 5), "wooden_chair — Original SAM3D vs VIGA (NaN root cause experiment)", fill=(220, 220, 220))
    for i, (img, lbl) in enumerate(zip(images, labels)):
        x = PAD + i * (W + PAD)
        cmp.paste(img, (x, 45))
        for j, line in enumerate(lbl.split("\n")):
            draw.text((x + 4, 45 + H + 4 + j * 18), line, fill=(180, 220, 255))

    OUT_CMP.mkdir(parents=True, exist_ok=True)
    cmp_path = OUT_CMP / "comparison.png"
    cmp.save(str(cmp_path))
    print(f"Saved comparison: {cmp_path}")


if __name__ == "__main__":
    main()
