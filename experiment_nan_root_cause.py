#!/usr/bin/env python3
"""Empirical experiment: no-scene-image vs scene-image SAM3D (NaN root cause).

Runs SAM3D on dining scene / wooden_chair object twice using the CURRENT
submodule (v9 mask growth, occlusion-check-disabled):

  Run 1 (no-scene-image): MoGe runs on per-object masked image
                           -> pipeline_pointmap may contain NaN
                           -> run_post_optimization receives NaN pointmap
  Run 2 (scene-image):    MoGe runs on full scene image
                           -> pipeline_pointmap is always valid
                           -> run_post_optimization receives valid pointmap

NOTE: The original Meta submodule (af582ce) cannot be used directly because
its inference.py imports a broken gradio version incompatible with this env.
The key difference being tested (scene-image MoGe path) is fully captured
in the current submodule.

For each run, saves:
  {out}/wooden_chair.glb          - 3D mesh (PyTorch3D camera space)
  {out}/wooden_chair_info.json    - pose: rotation/translation/scale + IoU
  {out}/wooden_chair_log.txt      - full stdout/stderr from sam3d_worker
  {out}/wooden_chair_mask.png     - SAM mask overlaid on scene photo
  {out}/wooden_chair_proj.png     - GLB vertices projected onto scene photo

Then generates:
  output/experiment_comparison/comparison.png

Usage:
    C:/Users/kingy/miniconda3/envs/sam3d_py311/python.exe experiment_nan_root_cause.py

Must be run from the project root directory.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
SAM3D_SUBMODULE = PROJECT_ROOT / "utils" / "third_party" / "sam3d"
CONFIG = SAM3D_SUBMODULE / "checkpoints" / "hf" / "checkpoints" / "pipeline.yaml"
# Use the resized image (771x1024) — matches mask dimensions from SAM segmentation
TARGET_IMAGE = PROJECT_ROOT / "data" / "static_scene" / "dining" / "target_resized.jpg"
MASK_FILE = PROJECT_ROOT / "output" / "sam3d_dining" / "wooden_chair.npy"
MOGE_NPZ = PROJECT_ROOT / "output" / "sam3d_dining" / "target_moge.npz"

# SAM3D py311 env python (has TRELLIS + MoGe + pytorch3d)
SAM3D_PYTHON = "C:/Users/kingy/miniconda3/envs/sam3d_py311/python.exe"

# Original Meta commit (last commit before our VIGA changes started at 59ecd78)
ORIGINAL_COMMIT = "af582ce"
# Current VIGA commit
VIGA_COMMIT = "5b667c8"

# Outputs
OUT_ORIGINAL = PROJECT_ROOT / "output" / "experiment_original_sam3d"
OUT_VIGA = PROJECT_ROOT / "output" / "experiment_viga_sam3d"
OUT_COMPARISON = PROJECT_ROOT / "output" / "experiment_comparison"

# Current sam3d_worker.py path
CURRENT_WORKER = PROJECT_ROOT / "tools" / "sam3d" / "sam3d_worker.py"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Helpers: submodule checkout
# ---------------------------------------------------------------------------
def submodule_checkout(commit: str) -> None:
    log(f"Checking out SAM3D submodule to {commit}...")
    r = subprocess.run(
        ["git", "-C", str(SAM3D_SUBMODULE), "checkout", commit],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"git checkout failed:\n{r.stderr}")
    log(f"  submodule now at {commit}")


def submodule_current_commit() -> str:
    r = subprocess.run(
        ["git", "-C", str(SAM3D_SUBMODULE), "rev-parse", "HEAD"],
        capture_output=True, text=True
    )
    return r.stdout.strip()[:8]


# Path to the separate original worker script (no scene_image)
ORIGINAL_WORKER = PROJECT_ROOT / "_exp_worker_original.py"


# ---------------------------------------------------------------------------
# Run SAM3D
# ---------------------------------------------------------------------------
def run_sam3d_no_scene_image(out_dir: Path) -> bool:
    """Run SAM3D WITHOUT --scene-image: MoGe on per-object masked image.

    This reproduces the original behavior where MoGe receives the dark/masked
    per-object image and may produce NaN in the pointmap.
    Uses current submodule (v9 mask growth, occlusion check disabled).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / "wooden_chair.glb"
    info_path = out_dir / "wooden_chair_info.json"
    log_path = out_dir / "wooden_chair_log.txt"

    cmd = [
        SAM3D_PYTHON, "-u", str(CURRENT_WORKER),
        "--image", str(TARGET_IMAGE.resolve()),
        "--mask", str(MASK_FILE.resolve()),
        "--config", str(CONFIG.resolve()),
        "--glb", str(glb_path.resolve()),
        "--info", str(info_path.resolve()),
        # NO --scene-image: MoGe runs on per-object masked image
    ]
    log(f"Running SAM3D (no --scene-image): {' '.join(cmd[:3])} ...")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True,
                           cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    log(f"  Finished in {elapsed/60:.1f} min, returncode={r.returncode}")
    if r.returncode != 0:
        log(f"  FAILED -- check {log_path}")
        return False
    return True


def run_sam3d_viga(out_dir: Path) -> bool:
    """Run SAM3D with current VIGA code (--scene-image)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / "wooden_chair.glb"
    info_path = out_dir / "wooden_chair_info.json"
    log_path = out_dir / "wooden_chair_log.txt"

    cmd = [
        SAM3D_PYTHON, "-u", str(CURRENT_WORKER),
        "--image", str(TARGET_IMAGE.resolve()),
        "--mask", str(MASK_FILE.resolve()),
        "--config", str(CONFIG.resolve()),
        "--glb", str(glb_path.resolve()),
        "--info", str(info_path.resolve()),
        "--scene-image", str(TARGET_IMAGE.resolve()),
    ]
    log(f"Running VIGA SAM3D: {' '.join(cmd[:3])} ...")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True,
                           cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    log(f"  Finished in {elapsed/60:.1f} min, returncode={r.returncode}")
    if r.returncode != 0:
        log(f"  FAILED — check {log_path}")
        return False
    return True


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def load_moge_intrinsics(npz_path: Path) -> dict | None:
    """Load camera intrinsics from MoGe NPZ file."""
    if not npz_path.exists():
        log(f"  MoGe NPZ not found: {npz_path}")
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


def load_moge_scene_intrinsics_fallback(mask_shape: tuple) -> dict:
    """Estimate intrinsics from mask dimensions (fallback if no NPZ).
    mask_shape = (H, W) in numpy convention."""
    H, W = mask_shape
    f = max(W, H) * 0.8
    return {"K": None, "fx": f, "fy": f, "cx": W/2, "cy": H/2, "width": W, "height": H}


def project_glb_onto_image(
    glb_path: Path,
    info_path: Path,
    scene_img: Image.Image,
    cam: dict,
    mask_npy: np.ndarray,
) -> Image.Image:
    """Project GLB mesh vertices onto scene image and draw colored dots.

    The GLB vertices are already in PyTorch3D camera space (X-left, Y-up, Z-forward).
    We need to convert to OpenCV camera space (X-right, Y-down, Z-forward) for projection.
    """
    try:
        import trimesh
    except ImportError:
        log("  trimesh not available, skipping projection")
        return scene_img.copy()

    if not glb_path.exists():
        log(f"  GLB not found: {glb_path}")
        return _label_image(scene_img.copy(), "GLB not found\n(run failed?)", color="red")

    if not info_path.exists():
        log(f"  Info JSON not found: {info_path}")
        return _label_image(scene_img.copy(), "Info JSON not found", color="red")

    # Check for NaN/inf in info
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    T = info.get("translation", [0, 0, 0])
    R = info.get("rotation", [1, 0, 0, 0])
    if any(np.isnan(v) or np.isinf(v) for v in T + R):
        return _label_image(scene_img.copy(),
                            f"NaN/Inf in pose!\nT={[f'{v:.3f}' for v in T]}", color="red")

    # Load mesh
    scene = trimesh.load(str(glb_path), force="scene")
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            g for g in scene.geometry.values() if hasattr(g, "vertices")
        ])
    else:
        mesh = scene
    verts = np.array(mesh.vertices, dtype=np.float64)

    if len(verts) == 0:
        return _label_image(scene_img.copy(), "Empty mesh", color="orange")

    # Subsample for speed
    if len(verts) > 5000:
        idx = np.random.choice(len(verts), 5000, replace=False)
        verts = verts[idx]

    # Vertices are in PyTorch3D camera space: X-left, Y-up, Z-forward
    # Convert to OpenCV: X-right, Y-down, Z-forward
    # X_ocv = -X_p3d,  Y_ocv = -Y_p3d,  Z_ocv = Z_p3d
    x_ocv = -verts[:, 0]
    y_ocv = -verts[:, 1]
    z_ocv =  verts[:, 2]

    # Only keep points in front of camera
    valid = z_ocv > 0.01
    x_ocv, y_ocv, z_ocv = x_ocv[valid], y_ocv[valid], z_ocv[valid]

    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    u = (fx * x_ocv / z_ocv + cx).astype(np.int32)
    v = (fy * y_ocv / z_ocv + cy).astype(np.int32)

    W, H = scene_img.size
    in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[in_frame], v[in_frame]

    # Draw on image
    result = scene_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for ux, vy in zip(u, v):
        draw.ellipse([ux-2, vy-2, ux+2, vy+2], fill=(0, 200, 255, 200))
    result = Image.alpha_composite(result, overlay).convert("RGB")

    # Annotate IoU if present
    iou = info.get("final_iou")
    if iou is not None:
        draw2 = ImageDraw.Draw(result)
        draw2.text((10, 10), f"IoU: {iou:.3f}", fill=(255, 255, 0))

    return result


def make_mask_overlay(scene_img: Image.Image, mask_npy: np.ndarray) -> Image.Image:
    """Overlay the SAM mask on the scene image (green = object region)."""
    # mask_npy: 0 = object, 255 = background (or bool True = object)
    if mask_npy.dtype == bool:
        obj_mask = mask_npy
    else:
        # dining masks: 255 = object, 0 = background
        # (sam3d_worker uses mask > 0, so 255 = object)
        obj_mask = mask_npy > 0

    img_arr = np.array(scene_img.convert("RGB"), dtype=np.float32)
    overlay = img_arr.copy()
    overlay[obj_mask, 0] = 0
    overlay[obj_mask, 1] = 200
    overlay[obj_mask, 2] = 0

    blended = (0.6 * img_arr + 0.4 * overlay).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(blended)
    draw = ImageDraw.Draw(result)
    pct = 100.0 * obj_mask.sum() / obj_mask.size
    draw.text((10, 10), f"SAM mask: {pct:.1f}% coverage", fill=(255, 255, 0))
    return result


def _label_image(img: Image.Image, text: str, color: str = "white") -> Image.Image:
    """Add a text label to an image."""
    draw = ImageDraw.Draw(img)
    colors = {"red": (255, 80, 80), "orange": (255, 160, 0), "white": (255, 255, 255)}
    c = colors.get(color, (255, 255, 255))
    draw.rectangle([0, 0, img.width, 60], fill=(0, 0, 0, 180) if img.mode == "RGBA" else (0, 0, 0))
    draw.text((10, 10), text, fill=c)
    return img


def make_panel(title: str, images: list[Image.Image], labels: list[str]) -> Image.Image:
    """Make a horizontal panel of images with title and per-image labels."""
    W = images[0].width
    H = images[0].height
    PAD = 4
    LABEL_H = 30
    TITLE_H = 40
    n = len(images)
    panel_w = n * W + (n + 1) * PAD
    panel_h = H + LABEL_H + TITLE_H + 2 * PAD

    panel = Image.new("RGB", (panel_w, panel_h), (30, 30, 30))
    draw = ImageDraw.Draw(panel)
    draw.text((PAD, 6), title, fill=(220, 220, 220))

    for i, (img, lbl) in enumerate(zip(images, labels)):
        x = PAD + i * (W + PAD)
        y = TITLE_H
        panel.paste(img.resize((W, H)), (x, y))
        draw.text((x + 4, y + H + 4), lbl, fill=(180, 220, 255))

    return panel


def generate_comparison(
    scene_img: Image.Image,
    mask_npy: np.ndarray,
    cam: dict,
    out_orig: Path,
    out_viga: Path,
    out_cmp: Path,
) -> None:
    """Generate side-by-side comparison image."""
    out_cmp.mkdir(parents=True, exist_ok=True)

    TARGET_W = 600
    # Mask lives in MoGe space (cam["width"] x cam["height"]).
    # Resize the scene image to match MoGe dims first, then down to TARGET_W.
    moge_w, moge_h = cam["width"], cam["height"]
    scene_moge = scene_img.resize((moge_w, moge_h), Image.LANCZOS)
    # Now downscale to TARGET_W for display
    disp_scale = TARGET_W / moge_w
    disp_w = TARGET_W
    disp_h = int(moge_h * disp_scale)
    scene_small = scene_moge.resize((disp_w, disp_h), Image.LANCZOS)
    H_sm = scene_small.height

    # Scale mask (already in MoGe resolution) to display size
    mask_img_pil = Image.fromarray((mask_npy > 0).astype(np.uint8) * 255, mode="L")
    mask_img_pil = mask_img_pil.resize((disp_w, disp_h), Image.NEAREST)
    mask_rsz = np.array(mask_img_pil) > 0

    # Scale intrinsics from MoGe space to display space
    cam_small = {
        "fx": cam["fx"] * disp_scale,
        "fy": cam["fy"] * disp_scale,
        "cx": cam["cx"] * disp_scale,
        "cy": cam["cy"] * disp_scale,
    }

    # Build panels
    mask_vis = make_mask_overlay(scene_small, mask_rsz)

    orig_info_exists = (out_orig / "wooden_chair_info.json").exists()
    viga_info_exists = (out_viga / "wooden_chair_info.json").exists()

    orig_proj = project_glb_onto_image(
        out_orig / "wooden_chair.glb",
        out_orig / "wooden_chair_info.json",
        scene_small, cam_small, mask_rsz,
    )
    viga_proj = project_glb_onto_image(
        out_viga / "wooden_chair.glb",
        out_viga / "wooden_chair_info.json",
        scene_small, cam_small, mask_rsz,
    )

    # Save individual outputs
    mask_vis.save(str(out_orig / "wooden_chair_mask.png"))
    mask_vis.save(str(out_viga / "wooden_chair_mask.png"))
    orig_proj.save(str(out_orig / "wooden_chair_proj.png"))
    viga_proj.save(str(out_viga / "wooden_chair_proj.png"))
    log(f"  Saved mask + projection images to {out_orig} and {out_viga}")

    # Load info for labels
    def _iou_label(info_path: Path) -> str:
        if not info_path.exists():
            return "FAILED"
        with open(info_path) as f:
            info = json.load(f)
        T = info.get("translation", [0, 0, 0])
        if any(np.isnan(v) or np.isinf(v) for v in T):
            return "NaN detected!"
        iou = info.get("final_iou")
        return f"IoU={iou:.3f}" if iou is not None else "no IoU"

    orig_lbl = "No --scene-image (per-obj MoGe)"
    viga_lbl = "--scene-image (full-scene MoGe)"
    orig_iou = _iou_label(out_orig / "wooden_chair_info.json")
    viga_iou = _iou_label(out_viga / "wooden_chair_info.json")

    # Big comparison
    PAD = 8
    col_w = TARGET_W
    row_imgs = [mask_vis, orig_proj, viga_proj]
    row_labels = [
        "SAM mask overlay",
        f"No --scene-image\n({orig_iou})",
        f"VIGA projection\n({viga_iou})",
    ]
    total_w = len(row_imgs) * col_w + (len(row_imgs) + 1) * PAD
    total_h = H_sm + 80 + PAD * 2
    cmp = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    draw = ImageDraw.Draw(cmp)
    draw.text((PAD, 8), "wooden_chair — no-scene-image vs scene-image MoGe", fill=(220, 220, 220))
    for i, (img, lbl) in enumerate(zip(row_imgs, row_labels)):
        x = PAD + i * (col_w + PAD)
        cmp.paste(img, (x, 50))
        draw.text((x + 4, 50 + H_sm + 4), lbl, fill=(180, 220, 255))

    cmp_path = out_cmp / "comparison.png"
    cmp.save(str(cmp_path))
    log(f"  Saved comparison: {cmp_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log("=" * 60)
    log("Experiment: no-scene-image vs scene-image SAM3D on wooden_chair")
    log("=" * 60)

    # Verify inputs exist
    for p, name in [
        (TARGET_IMAGE, "target image"),
        (MASK_FILE, "wooden_chair mask"),
        (CONFIG, "SAM3D config"),
    ]:
        if not p.exists():
            log(f"ERROR: {name} not found: {p}")
            sys.exit(1)

    log(f"Target image: {TARGET_IMAGE}")
    log(f"Mask file:    {MASK_FILE}")
    log(f"Config:       {CONFIG}")
    log(f"Output orig:  {OUT_ORIGINAL}")
    log(f"Output VIGA:  {OUT_VIGA}")

    log(f"Submodule commit: {submodule_current_commit()}")

    # ---- RUN 1: NO --scene-image (original behavior) -------------------
    log("\n" + "=" * 40)
    log("RUN 1: SAM3D WITHOUT --scene-image")
    log("  (MoGe on per-object masked image -> potential NaN)")
    log("=" * 40)
    ok1 = run_sam3d_no_scene_image(OUT_ORIGINAL)

    # ---- RUN 2: WITH --scene-image (VIGA fix) --------------------------
    log("\n" + "=" * 40)
    log("RUN 2: SAM3D WITH --scene-image (VIGA fix)")
    log("  (MoGe on full scene image -> no NaN)")
    log("=" * 40)
    ok2 = run_sam3d_viga(OUT_VIGA)

    # ---- VISUALIZATION -------------------------------------------------
    log("\n" + "=" * 40)
    log("Generating comparison visualizations")
    log("=" * 40)

    scene_img = Image.open(str(TARGET_IMAGE)).convert("RGB")
    mask_npy = np.load(str(MASK_FILE))

    # Try to load MoGe intrinsics; fall back to estimating from image
    cam = load_moge_intrinsics(MOGE_NPZ)
    if cam is None:
        log("  No MoGe NPZ, estimating intrinsics from mask dimensions")
        cam = load_moge_scene_intrinsics_fallback(mask_npy.shape)
    else:
        log(f"  Loaded MoGe intrinsics: fx={cam['fx']:.1f} fy={cam['fy']:.1f} "
            f"cx={cam['cx']:.1f} cy={cam['cy']:.1f} ({cam['width']}x{cam['height']})")

    generate_comparison(scene_img, mask_npy, cam, OUT_ORIGINAL, OUT_VIGA, OUT_COMPARISON)

    # ---- SUMMARY -------------------------------------------------------
    log("\n" + "=" * 40)
    log("SUMMARY")
    log("=" * 40)
    log(f"Run 1 (no-scene-image): {'OK' if ok1 else 'FAILED'} -> {OUT_ORIGINAL}")
    log(f"Run 2 (scene-image):    {'OK' if ok2 else 'FAILED'} -> {OUT_VIGA}")
    log(f"Comparison:       {OUT_COMPARISON / 'comparison.png'}")

    for label, info_path in [
        ("Original", OUT_ORIGINAL / "wooden_chair_info.json"),
        ("VIGA",     OUT_VIGA     / "wooden_chair_info.json"),
    ]:
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            T = info.get("translation", [])
            R = info.get("rotation", [])
            iou = info.get("final_iou", "N/A")
            has_nan = any(np.isnan(v) or np.isinf(v) for v in T + R) if T else False
            log(f"  {label}: T={[f'{v:.4f}' for v in T][:3]}, iou={iou}, NaN={has_nan}")
        else:
            log(f"  {label}: info JSON missing (run failed)")


if __name__ == "__main__":
    main()
