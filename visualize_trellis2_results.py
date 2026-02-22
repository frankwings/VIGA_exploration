"""Visualize TRELLIS2 results: masks, rotation GIFs, 2D projection overlay.

Steps:
1. Show input image + per-object SAM masks
2. Render rotation GIFs for each GLB using Open3D
3. Project each aligned GLB's vertices to 2D using intrinsics
4. Overlay projected silhouettes on the input image

Usage:
    python visualize_trellis2_results.py [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "output", "sam3d_dining_t2")
TARGET_IMAGE = os.path.join(ROOT, "data", "static_scene", "dining", "target_resized.jpg")

# Colors for each object (RGBA)
COLORS = [
    (255, 0, 0, 128),      # red
    (0, 255, 0, 128),      # green
    (0, 0, 255, 128),      # blue
    (255, 255, 0, 128),    # yellow
    (255, 0, 255, 128),    # magenta
    (0, 255, 255, 128),    # cyan
    (255, 128, 0, 128),    # orange
    (128, 0, 255, 128),    # purple
]

COLORS_SOLID = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]


def make_mask_grid(target_img, masks, names, output_path):
    """Create a grid: target image + individual object masks."""
    n = len(names)
    cols = 3
    rows = (n + 1 + cols - 1) // cols  # +1 for the target image
    cell_w, cell_h = 400, 300
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    # First cell: target image
    target_resized = target_img.resize((cell_w, cell_h), Image.LANCZOS)
    grid.paste(target_resized, (0, 0))
    draw.text((5, 5), "Target Image", fill=(255, 255, 255))

    # Remaining cells: per-object masks overlaid on target
    for i, name in enumerate(names):
        idx = i + 1
        r, c = divmod(idx, cols)
        x, y = c * cell_w, r * cell_h

        mask = masks[i]
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask_bool = mask > 0

        # Create masked overlay
        overlay = target_img.copy().convert("RGBA")
        color_layer = Image.new("RGBA", overlay.size, COLORS[i % len(COLORS)])
        mask_pil = Image.fromarray((mask_bool * 255).astype(np.uint8), mode="L")
        overlay = Image.composite(
            Image.alpha_composite(overlay, color_layer),
            overlay,
            mask_pil,
        )
        overlay = overlay.convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
        grid.paste(overlay, (x, y))
        draw.text((x + 5, y + 5), f"{name}", fill=(255, 255, 255))

    grid.save(output_path)
    print(f"[VIZ] Mask grid → {output_path}")


def load_mesh(glb_path):
    """Load a GLB file and return a single trimesh.Trimesh."""
    scene_data = trimesh.load(glb_path)
    if isinstance(scene_data, trimesh.Scene):
        return trimesh.util.concatenate(
            [g for g in scene_data.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    return scene_data


def render_rotation_blender(glb_path, name, output_dir, n_frames=12, resolution=256):
    """Render rotation frames using Blender via blender_render_rotation.py script."""
    import subprocess
    import tempfile

    blender_script = os.path.join(ROOT, "tools", "blender_render_rotation.py")

    # Blender command — auto-detect platform
    if sys.platform == "win32":
        blender_cmd = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
    else:
        blender_cmd = "/usr/local/bin/blender"
        # GCP VM may have blender via snap or custom path
        for p in ["/usr/local/bin/blender", "/snap/bin/blender",
                   os.path.expanduser("~/blender/blender")]:
            if os.path.exists(p):
                blender_cmd = p
                break

    frame_dir = os.path.join(output_dir, f"{name}_frames")
    os.makedirs(frame_dir, exist_ok=True)

    cmd = [
        blender_cmd, "--background", "--python", blender_script,
        "--", str(glb_path), str(frame_dir), "--frames", str(n_frames),
    ]

    print(f"[VIZ] Blender rendering {name} ({n_frames} frames)...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"[VIZ] Blender stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"Blender render failed for {name}")

    # Collect Y-rotation frames (turntable) and build GIF
    # Blender script names frames by GLB stem (e.g., chair_pbr_aligned_y_00.png)
    from pathlib import Path
    glb_stem = Path(glb_path).stem  # e.g., "chair_pbr_aligned"
    y_frames = sorted(Path(frame_dir).glob(f"{glb_stem}_y_*.png"))
    if not y_frames:
        # Fallback: any file matching *_y_*.png
        y_frames = sorted(Path(frame_dir).glob("*_y_*.png"))[:n_frames]
    if not y_frames:
        # Last fallback: any png in frame dir
        y_frames = sorted(Path(frame_dir).glob("*.png"))[:n_frames]

    frames = []
    for fp in y_frames:
        img = Image.open(fp).convert("RGB")
        if img.size != (resolution, resolution):
            img = img.resize((resolution, resolution), Image.LANCZOS)
        frames.append(img)

    if not frames:
        raise RuntimeError(f"No rendered frames found for {name}")

    gif_path = os.path.join(output_dir, f"{name}_rotation.gif")
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:],
        duration=150, loop=0,
    )
    print(f"[VIZ] Rotation GIF → {gif_path}")
    return gif_path, frames


def rasterize_faces_to_mask(verts, faces, intrinsics, pm_shape, color):
    """Rasterize mesh faces to a 2D silhouette mask."""
    K = np.array(intrinsics)
    H, W = pm_shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Project all vertices
    z = verts[:, 2].copy()
    z[z < 0.01] = 0.01
    u = (-verts[:, 0] / z * fx + cx) * W
    v = (-verts[:, 1] / z * fy + cy) * H

    # Create RGBA overlay image
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Subsample faces for speed (draw every Nth face)
    n_faces = len(faces)
    step = max(1, n_faces // 5000)  # limit to ~5000 triangles for drawing speed

    for i in range(0, n_faces, step):
        f = faces[i]
        z_face = verts[f, 2]
        if (z_face < 0.01).any():
            continue

        pts = [(float(u[f[j]]), float(v[f[j]])) for j in range(3)]

        # Check if all points are within image bounds (with margin)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if max(xs) < -W or min(xs) > 2 * W or max(ys) < -H or min(ys) > 2 * H:
            continue

        draw.polygon(pts, fill=color)

    return overlay


def create_overlay(target_img, transforms, data_dir, output_path):
    """Create 2D overlay of all projected objects on the target image."""
    target_rgba = target_img.convert("RGBA")
    H_img, W_img = target_img.size[1], target_img.size[0]  # PIL: (W, H)

    for i, obj in enumerate(transforms):
        name = obj["object_name"]
        glb_path = os.path.join(data_dir, f"{name}.glb")
        if not os.path.exists(glb_path):
            print(f"[VIZ] Skipping {name}: GLB not found")
            continue

        print(f"[VIZ] Projecting {name}...", flush=True)
        intrinsics = obj["intrinsics"]
        pm_shape = obj["pointmap_shape"]

        mesh = load_mesh(glb_path)

        color = COLORS[i % len(COLORS)]
        overlay = rasterize_faces_to_mask(
            mesh.vertices, mesh.faces, intrinsics, pm_shape, color,
        )

        # Resize overlay to match target image if needed
        if overlay.size != target_rgba.size:
            overlay = overlay.resize(target_rgba.size, Image.LANCZOS)

        target_rgba = Image.alpha_composite(target_rgba, overlay)

    # Add labels
    draw = ImageDraw.Draw(target_rgba)
    y_label = 10
    for i, obj in enumerate(transforms):
        name = obj["object_name"]
        iou = obj.get("iou", -1)
        color = COLORS_SOLID[i % len(COLORS_SOLID)]
        label = f"{name} (IoU={iou:.2f})"
        draw.text((10, y_label), label, fill=color)
        y_label += 18

    result = target_rgba.convert("RGB")
    result.save(output_path)
    print(f"[VIZ] Overlay → {output_path}")


def create_summary_grid(target_img, masks, names, gif_frames, overlay_img, output_path):
    """Create a final summary image: target | masks | rotation thumbnails | overlay."""
    W_panel = 512
    H_panel = int(W_panel * target_img.size[1] / target_img.size[0])

    # Layout: 2x2 grid
    # [target image] [mask grid]
    # [rotation gifs] [overlay]
    grid_w = W_panel * 2
    grid_h = H_panel * 2
    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))

    # Top-left: target
    grid.paste(target_img.resize((W_panel, H_panel), Image.LANCZOS), (0, 0))

    # Top-right: mask grid (use first frame of each mask)
    mask_cols = 4
    mask_rows = 2
    cell_w = W_panel // mask_cols
    cell_h = H_panel // mask_rows
    for i, name in enumerate(names):
        r, c = divmod(i, mask_cols)
        x = W_panel + c * cell_w
        y = r * cell_h

        mask = masks[i]
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask_bool = mask > 0

        mask_vis = target_img.copy().convert("RGBA")
        color_layer = Image.new("RGBA", mask_vis.size, COLORS[i % len(COLORS)])
        mask_pil = Image.fromarray((mask_bool * 255).astype(np.uint8), mode="L")
        mask_vis = Image.composite(
            Image.alpha_composite(mask_vis, color_layer),
            mask_vis,
            mask_pil,
        ).convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
        grid.paste(mask_vis, (x, y))

    # Bottom-left: rotation GIF first frames (4x2 grid)
    for i, name in enumerate(names):
        r, c = divmod(i, mask_cols)
        x = c * cell_w
        y = H_panel + r * cell_h

        if name in gif_frames and gif_frames[name]:
            thumb = gif_frames[name][0].resize((cell_w, cell_h), Image.LANCZOS)
            grid.paste(thumb, (x, y))

    # Bottom-right: overlay
    grid.paste(overlay_img.resize((W_panel, H_panel), Image.LANCZOS), (W_panel, H_panel))

    grid.save(output_path)
    print(f"[VIZ] Summary grid → {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--target-image", default=TARGET_IMAGE)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    data_dir = args.data_dir
    if args.output_dir is None:
        output_dir = os.path.join(data_dir, "viz")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load target image
    target_img = Image.open(args.target_image).convert("RGB")
    print(f"[VIZ] Target image: {target_img.size}")

    # Load transforms
    transforms_path = os.path.join(data_dir, "object_transforms.json")
    with open(transforms_path, 'r', encoding='utf-8') as f:
        transforms = json.load(f)

    names = [t["object_name"] for t in transforms]
    print(f"[VIZ] Objects: {names}")

    # Load masks
    all_masks = np.load(os.path.join(data_dir, "all_masks.npy"), allow_pickle=True)
    if all_masks.dtype == object:
        masks = [m for m in all_masks]
    elif all_masks.ndim == 3:
        masks = [all_masks[i] for i in range(all_masks.shape[0])]
    else:
        masks = [all_masks]

    # Step 1: Mask grid
    print("\n[VIZ] === Step 1: Mask Grid ===")
    make_mask_grid(target_img, masks, names, os.path.join(output_dir, "mask_grid.png"))

    # Step 2: Rotation GIFs (prefer PBR-textured GLBs)
    print("\n[VIZ] === Step 2: Rotation GIFs ===")
    all_gif_frames = {}
    for name in names:
        # Prefer PBR aligned > PBR canonical > plain aligned
        pbr_aligned = os.path.join(data_dir, f"{name}_pbr_aligned.glb")
        pbr_canonical = os.path.join(data_dir, f"{name}_pbr.glb")
        plain = os.path.join(data_dir, f"{name}.glb")
        if os.path.exists(pbr_aligned):
            glb_path = pbr_aligned
        elif os.path.exists(pbr_canonical):
            glb_path = pbr_canonical
        elif os.path.exists(plain):
            glb_path = plain
        else:
            print(f"[VIZ] Skipping {name}: no GLB found")
            continue
        print(f"[VIZ] {name}: using {os.path.basename(glb_path)}")
        try:
            gif_path, frames = render_rotation_blender(
                glb_path, name, output_dir, n_frames=12, resolution=256,
            )
            all_gif_frames[name] = frames
        except Exception as e:
            print(f"[VIZ] {name}: rotation render failed: {e}")
            all_gif_frames[name] = []

    # Step 3+4: 2D projection overlay
    print("\n[VIZ] === Step 3+4: 2D Projection Overlay ===")
    overlay_path = os.path.join(output_dir, "overlay.png")
    create_overlay(target_img, transforms, data_dir, overlay_path)
    overlay_img = Image.open(overlay_path)

    # Final summary grid
    print("\n[VIZ] === Summary Grid ===")
    create_summary_grid(
        target_img, masks, names, all_gif_frames, overlay_img,
        os.path.join(output_dir, "summary.png"),
    )

    print(f"\n[VIZ] All done! Output in: {output_dir}")


if __name__ == "__main__":
    main()
