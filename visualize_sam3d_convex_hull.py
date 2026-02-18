"""Visualize SAM3D convex_hull_v2 results: project 3D GLB vertices onto 2D scene.

For each object, creates:
  - {name}_compare.png   : 3-panel [object photo | depth projection | scene overlay]
  - scene_overlay.png    : all objects projected onto the scene image
  - scene_overlay_depth.png : all objects projected onto the depth map

GLB vertices are in PyTorch3D camera space (X-left, Y-up, Z-forward).
Convert to OpenCV (X-right, Y-down, Z-forward) by negating X and Y, then project.

Usage:
    C:/Users/kingy/miniconda3/envs/agent/python.exe visualize_sam3d_convex_hull.py
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

# Distinct bright colors per object (RGB)
COLORS = [
    (255,  60,  60),   # red    – green_tea_bottle
    ( 60, 200,  60),   # green  – ito_en_bottle
    ( 60, 120, 255),   # blue   – alienware_keyboard
    (255, 180,   0),   # yellow – headphones
    (200,   0, 255),   # purple – envelope
]

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def load_glb_vertices(glb_path: Path) -> np.ndarray:
    """Load all mesh vertices from a GLB file (concatenated)."""
    try:
        import trimesh
        scene = trimesh.load(str(glb_path), force="scene")
        all_verts = []
        for geom in scene.geometry.values():
            if hasattr(geom, "vertices"):
                all_verts.append(np.asarray(geom.vertices, dtype=np.float32))
        return np.concatenate(all_verts, axis=0) if all_verts else np.zeros((0, 3), np.float32)
    except Exception as e:
        print(f"  [WARN] trimesh failed: {e}")
        return np.zeros((0, 3), np.float32)


def pt3d_to_opencv(verts: np.ndarray) -> np.ndarray:
    """Convert PyTorch3D (X-left, Y-up, Z-fwd) → OpenCV (X-right, Y-down, Z-fwd)."""
    out = verts.copy()
    out[:, 0] = -verts[:, 0]
    out[:, 1] = -verts[:, 1]
    return out


def project(verts_cv: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    """Project OpenCV-space 3D points to pixel coordinates.

    Returns:
        uv : (N, 2) float pixel coords
        z  : (N,)  depth
    """
    z = verts_cv[:, 2]
    u = fx * verts_cv[:, 0] / (z + 1e-8) + cx
    v = fy * verts_cv[:, 1] / (z + 1e-8) + cy
    return np.stack([u, v], axis=-1), z


def subsample(verts: np.ndarray, n: int = 50_000) -> np.ndarray:
    """Randomly subsample to at most n points."""
    if len(verts) <= n:
        return verts
    idx = np.random.default_rng(0).choice(len(verts), n, replace=False)
    return verts[idx]

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_vertices_on_canvas(
    verts_pt3d: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    depth_map: np.ndarray,
    color: tuple[int, int, int] | None = None,
    bg: np.ndarray | None = None,
    dot_radius: int = 1,
) -> np.ndarray:
    """Project vertices and paint them on a canvas.

    Args:
        verts_pt3d : (N, 3) vertices in PyTorch3D camera space
        color      : If given, use this fixed RGB color. Otherwise color by depth.
        bg         : Background image (H, W, 3) uint8. If None, gray canvas.
        dot_radius : Dot size (pixels). 1 = single pixel, 2 = 3x3 square, etc.
    """
    if bg is not None:
        canvas = bg.copy().astype(np.uint8)
    else:
        canvas = np.full((H, W, 3), 220, dtype=np.uint8)

    if len(verts_pt3d) == 0:
        return canvas

    verts = subsample(verts_pt3d)
    verts_cv = pt3d_to_opencv(verts)
    uv, z_v = project(verts_cv, fx, fy, cx, cy)

    u_int = np.round(uv[:, 0]).astype(np.int32)
    v_int = np.round(uv[:, 1]).astype(np.int32)
    valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)

    if not valid.any():
        return canvas

    u_v, v_v = u_int[valid], v_int[valid]
    z_valid = z_v[valid]

    if color is not None:
        r_arr = np.full(valid.sum(), color[0], dtype=np.uint8)
        g_arr = np.full(valid.sum(), color[1], dtype=np.uint8)
        b_arr = np.full(valid.sum(), color[2], dtype=np.uint8)
    else:
        vmin, vmax = depth_map.min(), depth_map.max()
        z_norm = np.clip((z_valid - vmin) / (vmax - vmin + 1e-6), 0, 1)
        r_arr = np.clip((1.0 - z_norm) * 255, 0, 255).astype(np.uint8)
        g_arr = np.clip(z_norm * 200 + 55, 0, 255).astype(np.uint8)
        b_arr = np.clip(z_norm * 255, 0, 255).astype(np.uint8)

    # Paint back-to-front (furthest first)
    order = np.argsort(-z_valid)

    if dot_radius <= 1:
        flat_idx = v_v[order] * W + u_v[order]
        flat = canvas.reshape(-1, 3)
        flat[flat_idx] = np.stack([r_arr[order], g_arr[order], b_arr[order]], axis=-1)
    else:
        r = dot_radius
        for i in order:
            vi, ui = int(v_v[i]), int(u_v[i])
            v0, v1 = max(0, vi - r), min(H, vi + r + 1)
            u0, u1 = max(0, ui - r), min(W, ui + r + 1)
            canvas[v0:v1, u0:u1] = [r_arr[i], g_arr[i], b_arr[i]]

    return canvas


def add_label(canvas_pil: Image.Image, text: str, x: int = 4, y: int = 4,
              color=(255, 255, 255)) -> None:
    """Draw a text label on a PIL image in place."""
    draw = ImageDraw.Draw(canvas_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    # Shadow
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=color, font=font)

# ---------------------------------------------------------------------------
# Per-object comparison image
# ---------------------------------------------------------------------------

def make_object_compare(
    obj_name: str,
    verts_pt3d: np.ndarray,
    mask_raw: np.ndarray,
    png_path: Path,
    scene_rgb: np.ndarray,
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    iou: float,
    color: tuple[int, int, int],
    output_path: Path,
) -> None:
    """3-panel comparison image for one object.

    Panel 1: Object photo with mask outline (green)
    Panel 2: GLB projected on gray canvas, depth-colored
    Panel 3: GLB projected on scene image with object color
    """
    # --- Panel 1: object photo with mask outline ---
    if png_path.exists():
        photo = np.array(Image.open(png_path).convert("RGB"))
    else:
        photo = scene_rgb.copy()

    # Draw mask outline on photo
    mask_bool = mask_raw > 0
    if mask_bool.any():
        eroded = binary_erosion(mask_bool, iterations=2)
        edge = mask_bool & ~eroded
        photo_copy = photo.copy()
        photo_copy[edge] = [0, 255, 0]
    else:
        photo_copy = photo.copy()
    panel1 = photo_copy  # (H, W, 3)

    # --- Panel 2: depth-colored projection on gray ---
    panel2 = render_vertices_on_canvas(
        verts_pt3d, fx, fy, cx, cy, H, W, depth,
        color=None, bg=None, dot_radius=1,
    )

    # --- Panel 3: fixed-color projection on scene image ---
    panel3 = render_vertices_on_canvas(
        verts_pt3d, fx, fy, cx, cy, H, W, depth,
        color=color, bg=scene_rgb, dot_radius=2,
    )

    gap = 4
    header = 32
    total_w = W * 3 + gap * 2
    canvas = np.ones((H + header, total_w, 3), dtype=np.uint8) * 40  # dark bg

    canvas[header:, :W] = panel1
    canvas[header:, W + gap: W * 2 + gap] = panel2
    canvas[header:, W * 2 + gap * 2:] = panel3

    img = Image.fromarray(canvas)
    iou_str = f"{iou:.4f}" if isinstance(iou, float) and iou >= 0 else "N/A"
    add_label(img, f"{obj_name} — Mask", x=4, y=4)
    add_label(img, "Depth projection", x=W + gap + 4, y=4)
    add_label(img, f"Scene overlay  IoU={iou_str}", x=W * 2 + gap * 2 + 4, y=4)
    img.save(str(output_path))
    print(f"  Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Scene overlay image
# ---------------------------------------------------------------------------

def make_scene_overlay(
    objects_verts: dict[str, np.ndarray],
    scene_rgb: np.ndarray,
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    output_path: Path,
    title: str = "Scene 2D Overlay",
) -> None:
    """Overlay all objects on scene image with distinct colors + legend."""
    canvas = scene_rgb.copy().astype(np.float32)  # for blending

    legend_items = []
    for (name, verts), color in zip(objects_verts.items(), COLORS):
        if len(verts) == 0:
            continue
        v = subsample(verts)
        verts_cv = pt3d_to_opencv(v)
        uv, z_v = project(verts_cv, fx, fy, cx, cy)
        u_int = np.round(uv[:, 0]).astype(np.int32)
        v_int = np.round(uv[:, 1]).astype(np.int32)
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)
        if not valid.any():
            continue
        u_v, v_v = u_int[valid], v_int[valid]
        r = 2
        for i in range(len(u_v)):
            vi, ui = int(v_v[i]), int(u_v[i])
            v0, v1 = max(0, vi - r), min(H, vi + r + 1)
            u0, u1 = max(0, ui - r), min(W, ui + r + 1)
            canvas[v0:v1, u0:u1] = (
                0.4 * canvas[v0:v1, u0:u1] +
                0.6 * np.array(color, dtype=np.float32)
            )
        legend_items.append((name, color))

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    img = Image.fromarray(canvas)

    # Draw legend
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    lx, ly = 8, 8
    for label, col in legend_items:
        draw.rectangle([lx, ly, lx + 14, ly + 14], fill=col)
        draw.text((lx + 18, ly), label, fill=(255, 255, 255), font=font)
        ly += 20

    add_label(img, title, x=W // 2 - 80, y=4, color=(255, 255, 100))
    img.save(str(output_path))
    print(f"  Saved: {output_path.name}")


def make_depth_overlay(
    objects_verts: dict[str, np.ndarray],
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    output_path: Path,
) -> None:
    """Overlay projected vertices on depth map (turbo colormap)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth_rgb = (cm.turbo(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth_rgb = (np.stack([depth_norm] * 3, axis=-1) * 255).astype(np.uint8)

    canvas = depth_rgb.copy().astype(np.float32)

    for (name, verts), color in zip(objects_verts.items(), COLORS):
        if len(verts) == 0:
            continue
        v = subsample(verts)
        verts_cv = pt3d_to_opencv(v)
        uv, z_v = project(verts_cv, fx, fy, cx, cy)
        u_int = np.round(uv[:, 0]).astype(np.int32)
        v_int = np.round(uv[:, 1]).astype(np.int32)
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)
        if not valid.any():
            continue
        u_v, v_int_v = u_int[valid], v_int[valid]
        r = 2
        for i in range(len(u_v)):
            vi, ui = int(v_int_v[i]), int(u_v[i])
            v0, v1 = max(0, vi - r), min(H, vi + r + 1)
            u0, u1 = max(0, ui - r), min(W, ui + r + 1)
            canvas[v0:v1, u0:u1] = np.array(color, dtype=np.float32)

    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    out.save(str(output_path))
    print(f"  Saved: {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load MoGe results
    print("Loading MoGe results...")
    npz = np.load(str(MOGE_NPZ))
    K = npz["intrinsics_px"].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    depth = npz["depth"].astype(np.float32)
    W = int(npz["image_width"])
    H = int(npz["image_height"])
    print(f"  Camera: fx={fx:.1f} fy={fy:.1f}  cx={cx:.1f} cy={cy:.1f}")
    print(f"  Image:  {W}x{H}  depth=[{depth.min():.3f}, {depth.max():.3f}]")

    # Load scene image
    scene_img = Image.open(str(SCENE_IMG)).convert("RGB").resize((W, H), Image.LANCZOS)
    scene_rgb = np.array(scene_img)

    objects_verts: dict[str, np.ndarray] = {}
    iou_map: dict[str, float] = {}

    for name in OBJECTS:
        print(f"\n{'='*50}")
        print(f"Object: {name}")

        glb_path  = GLB_DIR / f"{name}.glb"
        info_path = INFO_DIR / f"{name}_info.json"
        mask_path = SAM_INIT / f"{name}.npy"
        png_path  = SAM_INIT / f"{name}.png"

        if not glb_path.exists():
            print(f"  [SKIP] No GLB: {glb_path}")
            objects_verts[name] = np.zeros((0, 3), np.float32)
            continue

        # Load GLB vertices (PyTorch3D camera space)
        verts = load_glb_vertices(glb_path)
        print(f"  Vertices: {len(verts):,}")

        if len(verts) > 0:
            # Quick sanity: depth range of projected vertices
            vc = pt3d_to_opencv(verts)
            valid_z = vc[vc[:, 2] > 0, 2]
            print(f"  Depth range: [{valid_z.min():.3f}, {valid_z.max():.3f}]" if len(valid_z) else "  No valid depth")

        objects_verts[name] = verts

        # Load IoU
        iou = -1.0
        if info_path.exists():
            info = json.loads(info_path.read_text())
            iou = float(info.get("iou", -1.0))
        iou_map[name] = iou
        print(f"  IoU: {iou:.4f}" if iou >= 0 else f"  IoU: N/A")

        # Load mask and PNG
        mask_raw = np.load(str(mask_path)).astype(np.uint8) if mask_path.exists() else np.zeros((H, W), np.uint8)

        # Find this object's color
        idx = OBJECTS.index(name)
        color = COLORS[idx]

        # Per-object comparison image
        out_path = OUTPUT_DIR / f"{name}_compare.png"
        make_object_compare(
            name, verts, mask_raw, png_path,
            scene_rgb, depth,
            fx, fy, cx, cy, H, W,
            iou, color, out_path,
        )

    # Scene overlays
    print(f"\n{'='*50}")
    print("Generating scene overlays...")
    make_scene_overlay(
        objects_verts, scene_rgb, depth,
        fx, fy, cx, cy, H, W,
        OUTPUT_DIR / "scene_overlay.png",
        title="SAM3D Convex Hull v2 — All Objects",
    )
    make_depth_overlay(
        objects_verts, depth,
        fx, fy, cx, cy, H, W,
        OUTPUT_DIR / "scene_overlay_depth.png",
    )

    # Summary
    print(f"\n{'='*50}")
    print("Results summary:")
    for name in OBJECTS:
        iou = iou_map.get(name, -1.0)
        n = len(objects_verts.get(name, []))
        print(f"  {name:<25}  IoU={iou:.4f}  verts={n:,}")
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
