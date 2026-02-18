"""Visualize SAM3D convex_hull_v2 results: project 3D GLB vertices onto 2D scene.

Matches the format of output/sam3d_dining_v4:
  - Grayscale depth map as background for scene overlays
  - Distinct neon colors per object (same OBJECT_COLORS as reoptimize_depth.py)
  - 1 pixel per vertex, no subsampling, back-to-front depth order
  - Side-by-side comparison: 2D masks (left) vs 3D projections (right)
  - Per-object 3-panel: photo+mask | 3D projection (depth-colored) | 3D on depth bg

Usage:
    C:/Users/kingy/miniconda3/envs/sam3d_py311/python.exe visualize_sam3d_convex_hull.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent

MOGE_NPZ   = ROOT / "output/sam3d_rerun_fixed/target_moge.npz"
SCENE_IMG  = ROOT / "data/static_scene/greentea/target.png"
SAM_INIT   = ROOT / "output/test/greentea/sam_init"
GLB_DIR    = ROOT / "output/sam3d_convex_hull_v2"
INFO_DIR   = ROOT / "output/sam3d_convex_hull_v2"
OUTPUT_DIR = ROOT / "output/sam3d_convex_hull_v2/vis"

OBJECTS = [
    "green_tea_bottle",
    "ito_en_bottle",
    "alienware_keyboard",
    "headphones",
    "envelope",
]

# Same palette as reoptimize_depth.py
OBJECT_COLORS = [
    (230, 25, 75),    # red
    (60, 180, 75),    # green
    (0, 130, 200),    # blue
    (255, 225, 25),   # yellow
    (245, 130, 48),   # orange
    (145, 30, 180),   # purple
    (70, 240, 240),   # cyan
    (240, 50, 230),   # magenta
    (210, 245, 60),   # lime
    (250, 190, 212),  # pink
]

# ---------------------------------------------------------------------------
# Geometry helpers (identical to reoptimize_depth.py)
# ---------------------------------------------------------------------------

def load_glb_vertices(glb_path: Path) -> np.ndarray:
    import trimesh
    scene = trimesh.load(str(glb_path), force="scene")
    all_verts = []
    for geom in scene.geometry.values():
        if hasattr(geom, "vertices"):
            all_verts.append(np.asarray(geom.vertices, dtype=np.float32))
    return np.concatenate(all_verts, axis=0) if all_verts else np.zeros((0, 3), np.float32)


def pt3d_to_opencv(verts: np.ndarray) -> np.ndarray:
    out = verts.copy()
    out[:, 0] = -verts[:, 0]
    out[:, 1] = -verts[:, 1]
    return out


def project(verts_cv: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    z = verts_cv[:, 2]
    u = fx * verts_cv[:, 0] / (z + 1e-8) + cx
    v = fy * verts_cv[:, 1] / (z + 1e-8) + cy
    return np.stack([u, v], axis=-1), z


def _font(size: int = 13):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()

# ---------------------------------------------------------------------------
# Rendering: depth-colored projection on gray canvas (same as dining_v4)
# ---------------------------------------------------------------------------

def render_projected_vertices(
    verts_pt3d: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    depth_map: np.ndarray,
) -> np.ndarray:
    """Project all vertices and colour by depth. Light gray background.
    Identical to reoptimize_depth.py render_projected_vertices()."""
    verts_cv = pt3d_to_opencv(verts_pt3d)
    uv, z_v = project(verts_cv, fx, fy, cx, cy)

    u_int = np.round(uv[:, 0]).astype(np.int32)
    v_int = np.round(uv[:, 1]).astype(np.int32)
    valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)

    vmin, vmax = depth_map.min(), depth_map.max()
    render = np.ones((H, W, 3), dtype=np.uint8) * 240  # light gray bg

    if valid.sum() > 0:
        u_v, v_v = u_int[valid], v_int[valid]
        z_valid = z_v[valid]
        z_norm = np.clip((z_valid - vmin) / (vmax - vmin + 1e-6), 0, 1)

        r = np.clip((1.0 - z_norm) * 255, 0, 255).astype(np.uint8)
        g = np.clip(z_norm * 200 + 55, 0, 255).astype(np.uint8)
        b = np.clip(z_norm * 255, 0, 255).astype(np.uint8)

        # Back-to-front (far first)
        order = np.argsort(-z_valid)
        flat_idx = v_v[order] * W + u_v[order]
        flat = render.reshape(-1, 3)
        flat[flat_idx] = np.stack([r[order], g[order], b[order]], axis=-1)

    return render

# ---------------------------------------------------------------------------
# Rendering: colored dots on grayscale depth map (scene overlay)
# ---------------------------------------------------------------------------

def paint_scene_overlay(
    verts_dict: dict[str, np.ndarray],
    depth_bg: np.ndarray,   # (H, W, 3) uint8 grayscale
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
) -> np.ndarray:
    """Paint each object's projected vertices onto a copy of depth_bg."""
    canvas = depth_bg.copy()
    for idx, (name, verts) in enumerate(verts_dict.items()):
        if len(verts) == 0:
            continue
        color = OBJECT_COLORS[idx % len(OBJECT_COLORS)]
        verts_cv = pt3d_to_opencv(verts)
        uv, z_v = project(verts_cv, fx, fy, cx, cy)
        u_int = np.round(uv[:, 0]).astype(np.int32)
        v_int = np.round(uv[:, 1]).astype(np.int32)
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)
        if valid.sum() == 0:
            continue
        u_v, v_v = u_int[valid], v_int[valid]
        order = np.argsort(-z_v[valid])   # back-to-front
        canvas[v_v[order], u_v[order]] = color
    return canvas


def paint_masks_overlay(
    masks_dict: dict[str, np.ndarray],   # name -> (H, W) uint8 mask
    depth_bg: np.ndarray,
) -> np.ndarray:
    """Paint each object's 2D mask boundary + fill onto depth_bg."""
    canvas = depth_bg.copy()
    for idx, (name, mask) in enumerate(masks_dict.items()):
        if mask is None or not mask.any():
            continue
        color = OBJECT_COLORS[idx % len(OBJECT_COLORS)]
        mask_bool = mask > 0
        # Paint fill lightly (blend 40%)
        fill = np.array(color, dtype=np.float32)
        canvas[mask_bool] = np.clip(
            0.6 * canvas[mask_bool].astype(np.float32) + 0.4 * fill, 0, 255
        ).astype(np.uint8)
        # Paint outline solidly
        eroded = binary_erosion(mask_bool, iterations=2)
        edge = mask_bool & ~eroded
        canvas[edge] = color
    return canvas

# ---------------------------------------------------------------------------
# Scene comparison: 2D masks (left) vs 3D projections (right)
# ---------------------------------------------------------------------------

def make_scene_comparison(
    verts_dict: dict[str, np.ndarray],
    masks_dict: dict[str, np.ndarray],
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    output_path: Path,
) -> None:
    """Side-by-side: 2D masks on depth (left) | 3D projections on depth (right).

    Matches the format of sam3d_dining_v4/scene_2d_comparison.png.
    """
    dmin, dmax = depth.min(), depth.max()
    depth_norm = ((depth - dmin) / (dmax - dmin + 1e-6) * 255).astype(np.uint8)
    depth_bg = np.stack([depth_norm, depth_norm, depth_norm], axis=-1)

    left_img  = paint_masks_overlay(masks_dict, depth_bg)
    right_img = paint_scene_overlay(verts_dict, depth_bg, fx, fy, cx, cy, H, W)

    gap = 6
    header_h = 50
    canvas_w = W * 2 + gap
    canvas = np.ones((H + header_h, canvas_w, 3), dtype=np.uint8) * 30  # dark bg

    canvas[header_h:, :W]        = left_img
    canvas[header_h:, W + gap:]  = right_img

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font     = _font(14)
    font_sm  = _font(11)

    # Panel titles
    draw.text((W // 2 - 40, 4),          "2D MASKS",       fill=(255, 100, 100), font=font)
    draw.text((W + gap + W // 2 - 60, 4), "3D PROJECTIONS", fill=(100, 255, 100), font=font)

    # Horizontal legend (object name → color square)
    x0 = 8
    for idx, name in enumerate(verts_dict.keys()):
        color = OBJECT_COLORS[idx % len(OBJECT_COLORS)]
        draw.rectangle([x0, 26, x0 + 10, 38], fill=color)
        draw.text((x0 + 14, 25), name, fill=(200, 200, 200), font=font_sm)
        x0 += len(name) * 7 + 26

    img.save(str(output_path))
    print(f"  Saved: {output_path.name}")

# ---------------------------------------------------------------------------
# Per-object comparison image (3 panels, white bg — matches dining_v4)
# ---------------------------------------------------------------------------

def make_object_compare(
    obj_name: str,
    verts_pt3d: np.ndarray,
    mask_raw: np.ndarray,
    png_path: Path,
    depth: np.ndarray,
    depth_bg: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    iou: float,
    output_path: Path,
) -> None:
    """3-panel: photo+mask | depth-colored 3D projection | 3D on depth bg.

    Matches sam3d_dining_v4/{name}_compare.png format.
    """
    # --- Panel 1: object photo with mask outline (black bg) ---
    if png_path.exists():
        photo = np.array(Image.open(png_path).convert("RGB").resize((W, H), Image.LANCZOS))
    else:
        photo = np.zeros((H, W, 3), dtype=np.uint8)

    mask_bool = mask_raw > 0
    if mask_bool.any():
        eroded = binary_erosion(mask_bool, iterations=2)
        edge = mask_bool & ~eroded
        photo[edge] = [0, 255, 0]
    panel1 = photo  # black bg (object photo already has bg blacked in sam_init PNGs)

    # --- Panel 2: depth-colored projection on light gray bg ---
    panel2 = render_projected_vertices(verts_pt3d, fx, fy, cx, cy, H, W, depth)

    # --- Panel 3: single-color projection on depth map bg ---
    panel3 = depth_bg.copy()
    if len(verts_pt3d) > 0:
        verts_cv = pt3d_to_opencv(verts_pt3d)
        uv, z_v = project(verts_cv, fx, fy, cx, cy)
        u_int = np.round(uv[:, 0]).astype(np.int32)
        v_int = np.round(uv[:, 1]).astype(np.int32)
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)
        if valid.sum() > 0:
            u_v, v_v = u_int[valid], v_int[valid]
            order = np.argsort(-z_v[valid])
            idx = OBJECTS.index(obj_name) if obj_name in OBJECTS else 0
            panel3[v_v[order], u_v[order]] = OBJECT_COLORS[idx % len(OBJECT_COLORS)]

    # Assemble on white canvas with 40px header
    gap = 4
    canvas = np.ones((H + 40, W * 3 + gap * 2, 3), dtype=np.uint8) * 255
    canvas[40:, :W]                  = panel1
    canvas[40:, W + gap: W * 2 + gap] = panel2
    canvas[40:, W * 2 + gap * 2:]    = panel3

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = _font(13)

    iou_str = f"{iou:.4f}" if isinstance(iou, float) and iou >= 0 else "N/A"
    draw.text((4,               4), f"{obj_name} - Mask",              fill=(0, 0, 0),     font=font)
    draw.text((W + gap + 4,     4), f"3D Projection (depth-colored)",   fill=(180, 50, 50), font=font)
    draw.text((W * 2 + gap*2 + 4, 4), f"3D on depth map  IoU={iou_str}", fill=(50, 160, 50), font=font)

    img.save(str(output_path))
    print(f"  Saved: {output_path.name}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MoGe results...")
    npz = np.load(str(MOGE_NPZ))
    K = npz["intrinsics_px"].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    depth  = npz["depth"].astype(np.float32)
    W = int(npz["image_width"])
    H = int(npz["image_height"])
    print(f"  Camera: fx={fx:.1f} fy={fy:.1f}  cx={cx:.1f} cy={cy:.1f}")
    print(f"  Image:  {W}x{H}  depth=[{depth.min():.3f}, {depth.max():.3f}]")

    # Shared depth background for scene overlays
    dmin, dmax = depth.min(), depth.max()
    depth_norm = ((depth - dmin) / (dmax - dmin + 1e-6) * 255).astype(np.uint8)
    depth_bg   = np.stack([depth_norm, depth_norm, depth_norm], axis=-1)

    verts_dict: dict[str, np.ndarray] = {}
    masks_dict: dict[str, np.ndarray] = {}
    iou_map: dict[str, float] = {}

    for name in OBJECTS:
        print(f"\n{'='*50}")
        print(f"Object: {name}")

        glb_path  = GLB_DIR / f"{name}.glb"
        info_path = INFO_DIR / f"{name}_info.json"
        mask_path = SAM_INIT / f"{name}.npy"
        png_path  = SAM_INIT / f"{name}.png"

        # Load mask
        mask_raw = np.load(str(mask_path)).astype(np.uint8) if mask_path.exists() else np.zeros((H, W), np.uint8)
        masks_dict[name] = mask_raw

        # Load IoU
        iou = -1.0
        if info_path.exists():
            iou = float(json.loads(info_path.read_text()).get("iou", -1.0))
        iou_map[name] = iou
        print(f"  IoU: {iou:.4f}" if iou >= 0 else "  IoU: N/A")

        # Load GLB vertices (PyTorch3D camera space)
        if not glb_path.exists():
            print(f"  [SKIP] No GLB")
            verts_dict[name] = np.zeros((0, 3), np.float32)
            continue

        verts = load_glb_vertices(glb_path)
        print(f"  Vertices: {len(verts):,}")
        verts_dict[name] = verts

        if len(verts) > 0:
            vc = pt3d_to_opencv(verts)
            valid_z = vc[vc[:, 2] > 0, 2]
            if len(valid_z):
                print(f"  Depth range: [{valid_z.min():.3f}, {valid_z.max():.3f}]")

        # Per-object comparison image
        make_object_compare(
            name, verts, mask_raw, png_path,
            depth, depth_bg,
            fx, fy, cx, cy, H, W,
            iou,
            OUTPUT_DIR / f"{name}_compare.png",
        )

    # Scene comparison: 2D masks (left) vs 3D projections (right)
    print(f"\n{'='*50}")
    print("Generating scene comparison overlay...")
    make_scene_comparison(
        verts_dict, masks_dict, depth,
        fx, fy, cx, cy, H, W,
        OUTPUT_DIR / "scene_2d_comparison.png",
    )

    # Summary
    print(f"\n{'='*50}")
    print("Results summary:")
    for name in OBJECTS:
        iou = iou_map.get(name, -1.0)
        n = len(verts_dict.get(name, []))
        print(f"  {name:<25}  IoU={iou:.4f}  verts={n:,}")
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
