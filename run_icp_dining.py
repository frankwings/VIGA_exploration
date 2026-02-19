#!/usr/bin/env python
"""ICP pose refinement for SAM3D dining objects — using v9 ray-cast grown masks.

For each object:
  1. Grow the SAM mask using v9 normal-consistency + 8-direction ray depth gate.
  2. Extract MoGe 3D points from the grown mask as ICP target.
  3. Run Open3D ICP to align mesh vertices → pointmap (OpenCV space).
  4. Apply post-ICP depth-scale correction (uniform scale from camera origin).
  5. Render: per-object projection, scene depth-map overlay, composite on original photo.

Outputs (in --output-dir):
  {name}_compare.png           masked photo | before projection | after projection
  {name}_overlay.png           ICP-aligned vertices blended onto original photo
  scene_depth_overlay.png      all objects projected on MoGe depth map (after ICP)
  scene_photo_overlay.png      all objects blended onto original target photo (after ICP)
  scene_photo_comparison.png   side-by-side: original photo | composite overlay
  scene_2d_comparison.png      side-by-side: before ICP | after ICP on depth map
  results_summary.json         per-object ICP stats

Usage:
    C:/Users/kingy/miniconda3/envs/sam3d_py311/python.exe run_icp_dining.py
    C:/Users/kingy/miniconda3/envs/sam3d_py311/python.exe run_icp_dining.py \\
        --data-dir output/sam3d_dining_v4 --output-dir output/sam3d_dining_icp2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:
    sys.exit("trimesh not installed.  pip install trimesh")

try:
    from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter
    from scipy.spatial import ConvexHull
except ImportError:
    sys.exit("scipy not installed.  pip install scipy")

try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d not installed.  pip install open3d")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SMOOTH_SIGMA   = 2.0    # Gaussian sigma for normal smoothing
MAX_ANGLE_DEG  = 60.0   # cap on adaptive normal threshold
EROSION_ITERS  = 2
DILATION_ITERS = 2

OBJECT_COLORS = [
    (230, 25,  75),   # red
    (60,  180, 75),   # green
    (0,   130, 200),  # blue
    (255, 225, 25),   # yellow
    (245, 130, 48),   # orange
    (145, 30,  180),  # purple
    (70,  240, 240),  # cyan
    (240, 50,  230),  # magenta
    (210, 245, 60),   # lime
    (250, 190, 212),  # pink
    (0,   128, 128),  # teal
    (220, 190, 255),  # lavender
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_moge(npz_path: Path) -> dict:
    data = np.load(npz_path)
    K = data["intrinsics_px"].astype(np.float64)
    depth  = data["depth"].astype(np.float32)
    points = data["points"].astype(np.float32)  # (H, W, 3) OpenCV space
    w, h = int(data["image_width"]), int(data["image_height"])
    return {
        "K": K, "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
        "depth": depth, "width": w, "height": h, "points": points,
    }


def load_glb_vertices(glb_path: Path) -> np.ndarray:
    scene = trimesh.load(str(glb_path), force="scene")
    all_verts = []
    for geom in scene.geometry.values():
        if hasattr(geom, "vertices"):
            all_verts.append(np.asarray(geom.vertices, dtype=np.float32))
    if not all_verts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(all_verts, axis=0)


def load_glb_scene(glb_path: Path):
    return trimesh.load(str(glb_path), force="scene")


def pt3d_to_opencv(verts: np.ndarray) -> np.ndarray:
    """PT3D (X-left, Y-up, Z-fwd) → OpenCV (X-right, Y-down, Z-fwd)."""
    out = verts.copy()
    out[:, 0] = -verts[:, 0]
    out[:, 1] = -verts[:, 1]
    return out


def project(verts_cv, fx, fy, cx, cy):
    z = verts_cv[:, 2]
    u = fx * verts_cv[:, 0] / z + cx
    v = fy * verts_cv[:, 1] / z + cy
    return np.stack([u, v], axis=-1), z


def load_target_image(data_dir: Path, H: int, W: int,
                      target_override: Path | None = None) -> np.ndarray | None:
    """Try to find and load the original target photograph."""
    if not Image:
        return None
    candidates = []
    if target_override:
        candidates.append(target_override)
    for name in ("target_resized.jpg", "target.jpg", "target.png", "target_resized.png"):
        candidates.append(data_dir / name)
        # Also check data/static_scene/<scene_name>/ inferred from data_dir stem
        # e.g. data_dir = output/sam3d_dining_v4 → scene = "dining"
        stem = data_dir.name  # e.g. "sam3d_dining_v4"
        for part in stem.split("_"):
            project_root = data_dir.parent.parent
            candidates.append(project_root / "data" / "static_scene" / part / name)
        # Try project_root/data/static_scene/dining directly
        project_root = data_dir.parent.parent
        candidates.append(project_root / "data" / "static_scene" / "dining" / name)
    for p in candidates:
        if p.exists():
            img = Image.open(p).convert("RGB").resize((W, H))
            print(f"  [INFO] Target photo: {p}")
            return np.array(img)
    return None


# ---------------------------------------------------------------------------
# V9 mask growth: normal-consistency + 8-dir ray depth gate
# ---------------------------------------------------------------------------

def _make_convex_hull_mask(mask_bool: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask_bool)
    if len(rows) < 3:
        return mask_bool.copy()
    pts = np.column_stack([cols, rows])
    try:
        hull = ConvexHull(pts)
    except Exception:
        return mask_bool.copy()
    from matplotlib.path import Path as MplPath
    hull_path = MplPath(pts[hull.vertices])
    H, W = mask_bool.shape
    rmin = max(0, rows.min() - 2);  rmax = min(H - 1, rows.max() + 2)
    cmin = max(0, cols.min() - 2);  cmax = min(W - 1, cols.max() + 2)
    yy, xx = np.mgrid[rmin:rmax+1, cmin:cmax+1]
    inside = hull_path.contains_points(
        np.column_stack([xx.ravel(), yy.ravel()])
    ).reshape(yy.shape)
    out = np.zeros_like(mask_bool)
    out[rmin:rmax+1, cmin:cmax+1] = inside
    return out


def _compute_normals(pointmap: np.ndarray) -> tuple:
    pm = gaussian_filter(pointmap.astype(np.float32),
                         sigma=[SMOOTH_SIGMA, SMOOTH_SIGMA, 0])
    dx = np.zeros_like(pm);  dy = np.zeros_like(pm)
    dx[:, 1:-1] = pm[:, 2:] - pm[:, :-2]
    dy[1:-1, :] = pm[2:, :]  - pm[:-2, :]
    n   = np.cross(dx, dy)
    nrm = np.linalg.norm(n, axis=-1, keepdims=True)
    valid = nrm[..., 0] > 1e-10
    n[valid]  /= nrm[valid]
    n[~valid]  = 0.0
    return n.astype(np.float32), valid


def _precompute_ray_first_hit(cleaned: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Directions: N(0) NE(1) E(2) SE(3) S(4) SW(5) W(6) NW(7)."""
    H, W = cleaned.shape
    mask_dep = np.where(cleaned, depth, np.nan).astype(np.float32)
    ray = np.full((H, W, 8), np.nan, dtype=np.float32)

    # Cardinal N
    ray[:, :, 0] = mask_dep.copy()
    for r in range(1, H):
        nan_r = np.isnan(ray[r, :, 0])
        ray[r, nan_r, 0] = ray[r - 1, nan_r, 0]

    # Cardinal S
    ray[:, :, 4] = mask_dep.copy()
    for r in range(H - 2, -1, -1):
        nan_r = np.isnan(ray[r, :, 4])
        ray[r, nan_r, 4] = ray[r + 1, nan_r, 4]

    # Cardinal E
    ray[:, :, 2] = mask_dep.copy()
    for c in range(W - 2, -1, -1):
        nan_c = np.isnan(ray[:, c, 2])
        ray[nan_c, c, 2] = ray[nan_c, c + 1, 2]

    # Cardinal W
    ray[:, :, 6] = mask_dep.copy()
    for c in range(1, W):
        nan_c = np.isnan(ray[:, c, 6])
        ray[nan_c, c, 6] = ray[nan_c, c - 1, 6]

    # NE (anti-diagonal r+c=k, top-right→bottom-left)
    ray[:, :, 1] = np.full((H, W), np.nan, dtype=np.float32)
    for k in range(H + W - 1):
        c_max = min(W - 1, k);  c_min = max(0, k - H + 1)
        for c in range(c_max, c_min - 1, -1):
            r = k - c
            if 0 <= r < H:
                if cleaned[r, c]:
                    ray[r, c, 1] = depth[r, c]
                else:
                    nr, nc = r - 1, c + 1
                    if 0 <= nr < H and 0 <= nc < W:
                        ray[r, c, 1] = ray[nr, nc, 1]

    # SW (anti-diagonal, bottom-left→top-right)
    ray[:, :, 5] = np.full((H, W), np.nan, dtype=np.float32)
    for k in range(H + W - 1):
        c_max = min(W - 1, k);  c_min = max(0, k - H + 1)
        for c in range(c_min, c_max + 1):
            r = k - c
            if 0 <= r < H:
                if cleaned[r, c]:
                    ray[r, c, 5] = depth[r, c]
                else:
                    nr, nc = r + 1, c - 1
                    if 0 <= nr < H and 0 <= nc < W:
                        ray[r, c, 5] = ray[nr, nc, 5]

    # SE (main diagonal r-c=k, bottom-right→top-left)
    ray[:, :, 3] = np.full((H, W), np.nan, dtype=np.float32)
    for k in range(-(W - 1), H):
        r_min = max(0, k);  r_max = min(H - 1, W - 1 + k)
        for r in range(r_max, r_min - 1, -1):
            c = r - k
            if 0 <= c < W:
                if cleaned[r, c]:
                    ray[r, c, 3] = depth[r, c]
                else:
                    nr, nc = r + 1, c + 1
                    if 0 <= nr < H and 0 <= nc < W:
                        ray[r, c, 3] = ray[nr, nc, 3]

    # NW (main diagonal, top-left→bottom-right)
    ray[:, :, 7] = np.full((H, W), np.nan, dtype=np.float32)
    for k in range(-(W - 1), H):
        r_min = max(0, k);  r_max = min(H - 1, W - 1 + k)
        for r in range(r_min, r_max + 1):
            c = r - k
            if 0 <= c < W:
                if cleaned[r, c]:
                    ray[r, c, 7] = depth[r, c]
                else:
                    nr, nc = r - 1, c - 1
                    if 0 <= nr < H and 0 <= nc < W:
                        ray[r, c, 7] = ray[nr, nc, 7]

    return ray


def grow_mask_v9(
    mask_bool: np.ndarray,
    pointmap: np.ndarray,  # (H, W, 3) — used for normals
    depth: np.ndarray,
) -> np.ndarray:
    """Grow mask toward convex hull using normal-consistency + v9 8-dir ray gate.

    Returns grown_mask (H, W) bool.
    """
    H, W = mask_bool.shape

    # Morphological opening
    cleaned = mask_bool.copy()
    if mask_bool.sum() > 50:
        eroded = binary_erosion(mask_bool, iterations=EROSION_ITERS)
        if eroded.sum() >= 10:
            cleaned = binary_dilation(eroded, iterations=DILATION_ITERS)

    hull_mask = _make_convex_hull_mask(cleaned)
    growth_region = hull_mask & ~cleaned

    if not growth_region.any() or cleaned.sum() < 3:
        return cleaned

    # Normals
    normals, valid = _compute_normals(pointmap)

    # Reference normal from cleaned mask
    inside_valid = cleaned & valid
    if inside_valid.sum() < 3:
        return cleaned

    ref = normals[inside_valid].mean(axis=0)
    ref_nrm = np.linalg.norm(ref)
    if ref_nrm < 1e-10:
        return cleaned
    ref /= ref_nrm

    # Angle deviation map
    dots = np.clip(np.abs((normals * ref).sum(axis=-1)), 0.0, 1.0)
    ang  = np.degrees(np.arccos(dots)).astype(np.float32)

    # Adaptive threshold
    ang_inside = ang[inside_valid]
    threshold_deg = float(np.clip(
        np.median(ang_inside) + 2.0 * ang_inside.std(),
        10.0, MAX_ANGLE_DEG,
    ))

    # 8-direction ray first-hit
    ray_first_hit = _precompute_ray_first_hit(cleaned, depth)  # (H, W, 8)

    # Per-growth-pixel gates
    gr_rows, gr_cols = np.where(growth_region)
    p_depths   = depth[gr_rows, gr_cols]
    p_ray_hits = ray_first_hit[gr_rows, gr_cols, :]  # (M, 8)

    dmin = np.nanmin(p_ray_hits, axis=1)
    dmax = np.nanmax(p_ray_hits, axis=1)

    has_hit   = np.isfinite(dmin)
    depth_ok  = has_hit & (dmin <= p_depths) & (p_depths <= dmax)
    normal_ok = (ang[gr_rows, gr_cols] < threshold_deg) & valid[gr_rows, gr_cols]

    accepted = normal_ok & depth_ok

    grown = cleaned.copy()
    grown[gr_rows[accepted], gr_cols[accepted]] = True
    return grown


# ---------------------------------------------------------------------------
# ICP alignment (identical to reoptimize_depth.py)
# ---------------------------------------------------------------------------

def run_icp_alignment(
    verts_pt3d: np.ndarray,
    pointmap: np.ndarray,      # (H, W, 3) OpenCV space
    grown_mask: np.ndarray,    # (H, W) bool
    icp_threshold: float = 0.3,
    depth_quantile: float = 0.9,
    max_source_points: int = 20000,
) -> tuple[np.ndarray, dict]:
    target_pts = pointmap[grown_mask]

    if len(target_pts) > 0 and depth_quantile < 1.0:
        z_vals = target_pts[:, 2]
        z_thresh = np.quantile(z_vals, depth_quantile)
        target_pts = target_pts[z_vals <= z_thresh]

    if len(target_pts) < 10:
        return verts_pt3d.copy(), {"fitness": 0.0, "rmse": 0.0,
                                   "n_target": 0, "n_source": 0}

    source_pts_cv = pt3d_to_opencv(verts_pt3d)

    if len(source_pts_cv) > max_source_points:
        idx = np.random.default_rng(42).choice(
            len(source_pts_cv), max_source_points, replace=False)
        source_pts_sub = source_pts_cv[idx]
    else:
        source_pts_sub = source_pts_cv

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(source_pts_sub.astype(np.float64))

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(target_pts.astype(np.float64))

    reg = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_correspondence_distance=icp_threshold,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )

    T_cv = reg.transformation
    M = np.diag([-1.0, -1.0, 1.0, 1.0])
    T_pt3d = M @ T_cv @ M

    ones = np.ones((len(verts_pt3d), 1), dtype=np.float64)
    verts_h = np.hstack([verts_pt3d.astype(np.float64), ones])
    verts_aligned = (verts_h @ T_pt3d.T)[:, :3].astype(np.float32)

    return verts_aligned, {
        "fitness": float(reg.fitness),
        "rmse": float(reg.inlier_rmse),
        "n_target": len(target_pts),
        "n_source": len(source_pts_sub),
        "transformation": T_pt3d.tolist(),
        "transformation_cv": T_cv.tolist(),
    }


def compute_depth_scale(
    verts_pt3d, mask_bool, depth, fx, fy, cx, cy, H, W,
) -> tuple[float, dict]:
    verts_cv = pt3d_to_opencv(verts_pt3d)
    uv, z_v  = project(verts_cv, fx, fy, cx, cy)
    u_int = np.round(uv[:, 0]).astype(np.int32)
    v_int = np.round(uv[:, 1]).astype(np.int32)
    in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)
    in_mask = np.zeros(len(verts_pt3d), dtype=bool)
    in_mask[in_bounds] = mask_bool[v_int[in_bounds], u_int[in_bounds]]
    valid = in_bounds & in_mask
    if valid.sum() < 10:
        return 1.0, {"valid_count": int(valid.sum()), "scale": 1.0}
    ratios = depth[v_int[valid], u_int[valid]] / np.maximum(z_v[valid], 1e-6)
    s = float(np.clip(np.median(ratios), 0.5, 2.0))
    return s, {"valid_count": int(valid.sum()), "scale": s,
               "ratio_std": float(np.std(ratios))}


def compute_depth_error(verts_pt3d, mask_bool, depth, fx, fy, cx, cy, H, W) -> dict:
    verts_cv = pt3d_to_opencv(verts_pt3d)
    uv, z_v  = project(verts_cv, fx, fy, cx, cy)
    u_int = np.round(uv[:, 0]).astype(np.int32)
    v_int = np.round(uv[:, 1]).astype(np.int32)
    in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)
    in_mask = np.zeros(len(verts_pt3d), dtype=bool)
    in_mask[in_bounds] = mask_bool[v_int[in_bounds], u_int[in_bounds]]
    valid = in_bounds & in_mask
    if valid.sum() == 0:
        return {"valid_count": 0, "depth_error_rel_mean": np.nan}
    z_m = depth[v_int[valid], u_int[valid]]
    abs_err = np.abs(z_v[valid] - z_m)
    rel_err = abs_err / np.maximum(z_m, 1e-6)
    return {
        "valid_count": int(valid.sum()),
        "depth_error_rel_mean": float(np.mean(rel_err)),
        "depth_error_abs_mean": float(np.mean(abs_err)),
    }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_vertices(verts_pt3d, fx, fy, cx, cy, H, W, depth_map):
    """Project vertices → colored image (depth-coded, light grey background)."""
    verts_cv = pt3d_to_opencv(verts_pt3d)
    uv, z_v  = project(verts_cv, fx, fy, cx, cy)
    u_int = np.round(uv[:, 0]).astype(np.int32)
    v_int = np.round(uv[:, 1]).astype(np.int32)
    valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)

    vmin, vmax = depth_map.min(), depth_map.max()
    render = np.ones((H, W, 3), dtype=np.uint8) * 240

    if valid.sum() > 0:
        u_v, v_v, z_valid = u_int[valid], v_int[valid], z_v[valid]
        z_norm = np.clip((z_valid - vmin) / (vmax - vmin + 1e-6), 0, 1)
        r = np.clip((1.0 - z_norm) * 255, 0, 255).astype(np.uint8)
        g = np.clip(z_norm * 200 + 55, 0, 255).astype(np.uint8)
        b = np.clip(z_norm * 255, 0, 255).astype(np.uint8)
        order = np.argsort(-z_valid)  # far to near
        render_flat = render.reshape(-1, 3)
        render_flat[v_v[order] * W + u_v[order]] = np.stack(
            [r[order], g[order], b[order]], axis=-1)
        render = render_flat.reshape(H, W, 3)
    return render


def render_scene_on_background(
    verts_dict: dict[str, np.ndarray],
    background: np.ndarray,   # (H, W, 3) uint8
    fx, fy, cx, cy, H, W,
    alpha: float = 0.6,
) -> np.ndarray:
    """Paint all objects' projected vertices onto background (alpha blend).

    Each object gets a distinct color.  A 3×3 dilation makes sparse vertices
    visible.  Vectorized: no per-vertex Python loop.
    """
    overlay  = np.zeros((H, W, 3), dtype=np.float32)
    hit_mask = np.zeros((H, W),    dtype=bool)

    for idx, (name, verts) in enumerate(verts_dict.items()):
        color = np.array(OBJECT_COLORS[idx % len(OBJECT_COLORS)], dtype=np.float32)
        verts_cv = pt3d_to_opencv(verts)
        uv, z_v  = project(verts_cv, fx, fy, cx, cy)
        u_int = np.round(uv[:, 0]).astype(np.int32)
        v_int = np.round(uv[:, 1]).astype(np.int32)
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)
        if valid.sum() == 0:
            continue
        u_v, v_v = u_int[valid], v_int[valid]
        # 3×3 dilation: paint 9 offset positions vectorized
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r_idx = np.clip(v_v + dr, 0, H - 1)
                c_idx = np.clip(u_v + dc, 0, W - 1)
                overlay[r_idx, c_idx]  = color
                hit_mask[r_idx, c_idx] = True

    bg = background.astype(np.float32)
    composite = np.where(
        hit_mask[:, :, None],
        alpha * overlay + (1.0 - alpha) * bg,
        bg,
    ).clip(0, 255).astype(np.uint8)
    return composite


def render_scene_on_depthmap(
    verts_dict: dict[str, np.ndarray],
    depth: np.ndarray,
    fx, fy, cx, cy, H, W,
    title: str,
    output_path: Path,
):
    """All objects projected on grey depth-map background."""
    dmin, dmax = depth.min(), depth.max()
    d_norm = ((depth - dmin) / (dmax - dmin + 1e-6) * 255).astype(np.uint8)
    bg = np.stack([d_norm, d_norm, d_norm], axis=-1)
    canvas = bg.copy()

    for idx, (name, verts) in enumerate(verts_dict.items()):
        color = OBJECT_COLORS[idx % len(OBJECT_COLORS)]
        verts_cv = pt3d_to_opencv(verts)
        uv, z_v  = project(verts_cv, fx, fy, cx, cy)
        u_int = np.round(uv[:, 0]).astype(np.int32)
        v_int = np.round(uv[:, 1]).astype(np.int32)
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)
        if valid.sum() == 0:
            continue
        u_v, v_v = u_int[valid], v_int[valid]
        order = np.argsort(-z_v[valid])
        canvas[v_v[order], u_v[order]] = color

    if Image:
        img_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 13)
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((4, 4), title, fill=(255, 255, 255), font=font)
        y0 = 24
        for idx, name in enumerate(verts_dict.keys()):
            c = OBJECT_COLORS[idx % len(OBJECT_COLORS)]
            draw.rectangle([8, y0, 22, y0 + 14], fill=c)
            draw.text((28, y0), name, fill=(255, 255, 255), font=font)
            y0 += 18
        img_pil.save(str(output_path))
    else:
        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        ax.imshow(canvas); ax.axis("off"); ax.set_title(title)
        fig.savefig(str(output_path), dpi=100, bbox_inches="tight"); plt.close()


def make_per_object_compare(
    target_img, mask_bool, verts_before, verts_after,
    depth, fx, fy, cx, cy, H, W,
    name, stats_b, stats_a, output_path,
):
    """3-panel: masked photo | before vertex projection | after vertex projection."""
    # Panel 1: target image with mask outline
    if target_img is not None:
        panel1 = target_img.copy()
        from scipy.ndimage import binary_erosion as be
        edge = mask_bool & ~be(mask_bool, iterations=2)
        panel1[edge] = [0, 255, 0]
    else:
        panel1 = np.zeros((H, W, 3), dtype=np.uint8)
        panel1[mask_bool] = [80, 80, 80]

    panel2 = render_vertices(verts_before, fx, fy, cx, cy, H, W, depth)
    panel3 = render_vertices(verts_after,  fx, fy, cx, cy, H, W, depth)

    gap = 4
    canvas = np.ones((H + 40, W * 3 + gap * 2, 3), dtype=np.uint8) * 255
    canvas[40:, :W]                    = panel1
    canvas[40:, W + gap: W * 2 + gap]  = panel2
    canvas[40:, W * 2 + gap * 2:]      = panel3

    if Image:
        img_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 13)
        except (OSError, IOError):
            font = ImageFont.load_default()
        rel_b = stats_b.get("depth_error_rel_mean", float("nan"))
        rel_a = stats_a.get("depth_error_rel_mean", float("nan"))
        draw.text((4, 4), f"{name} — Mask (green border)", fill=(0, 0, 0), font=font)
        draw.text((W + gap + 4, 4),
                  f"Before (rel_err={rel_b:.1%})" if np.isfinite(rel_b) else "Before",
                  fill=(180, 0, 0), font=font)
        draw.text((W * 2 + gap * 2 + 4, 4),
                  f"After (rel_err={rel_a:.1%})" if np.isfinite(rel_a) else "After",
                  fill=(0, 128, 0), font=font)
        img_pil.save(str(output_path))
    else:
        fig, ax = plt.subplots(figsize=((W * 3 + gap * 2) / 100, (H + 40) / 100), dpi=100)
        ax.imshow(canvas); ax.axis("off")
        fig.savefig(str(output_path), dpi=100, bbox_inches="tight"); plt.close()


def make_scene_before_after_depthmap(
    verts_before, verts_after, depth, fx, fy, cx, cy, H, W, output_path,
):
    """Side-by-side before/after depth-map overlay."""
    dmin, dmax = depth.min(), depth.max()
    d_norm = ((depth - dmin) / (dmax - dmin + 1e-6) * 255).astype(np.uint8)
    bg = np.stack([d_norm, d_norm, d_norm], axis=-1)
    names = list(verts_before.keys())

    def paint(vd):
        c = bg.copy()
        for i, nm in enumerate(names):
            col = OBJECT_COLORS[i % len(OBJECT_COLORS)]
            verts_cv = pt3d_to_opencv(vd[nm])
            uv, z_v  = project(verts_cv, fx, fy, cx, cy)
            u_int = np.round(uv[:, 0]).astype(np.int32)
            v_int = np.round(uv[:, 1]).astype(np.int32)
            valid = (u_int >= 0)&(u_int < W)&(v_int >= 0)&(v_int < H)&(z_v > 0)
            if valid.sum() == 0:
                continue
            order = np.argsort(-z_v[valid])
            c[v_int[valid][order], u_int[valid][order]] = col
        return c

    before_img = paint(verts_before)
    after_img  = paint(verts_after)
    gap = 6
    canvas = np.ones((H + 50, W * 2 + gap, 3), dtype=np.uint8) * 30
    canvas[50:, :W]         = before_img
    canvas[50:, W + gap:]   = after_img

    if Image:
        img_pil = Image.fromarray(canvas)
        draw    = ImageDraw.Draw(img_pil)
        try:
            font   = ImageFont.truetype("arial.ttf", 14)
            font_s = ImageFont.truetype("arial.ttf", 11)
        except (OSError, IOError):
            font = font_s = ImageFont.load_default()
        draw.text((W // 2 - 40, 4),        "BEFORE ICP", fill=(255, 100, 100), font=font)
        draw.text((W + gap + W // 2 - 40, 4), "AFTER ICP",  fill=(100, 255, 100), font=font)
        x0 = 8
        for i, nm in enumerate(names):
            col = OBJECT_COLORS[i % len(OBJECT_COLORS)]
            draw.rectangle([x0, 26, x0 + 10, 38], fill=col)
            draw.text((x0 + 14, 25), nm, fill=(200, 200, 200), font=font_s)
            x0 += len(nm) * 7 + 26
        img_pil.save(str(output_path))
    else:
        fig, ax = plt.subplots(figsize=((W * 2 + gap) / 100, (H + 50) / 100), dpi=100)
        ax.imshow(canvas); ax.axis("off")
        fig.savefig(str(output_path), dpi=100, bbox_inches="tight"); plt.close()


def make_scene_photo_comparison(target_img, overlay_img, H, W, output_path):
    """Side-by-side: original photo | composite overlay."""
    if target_img is None:
        return
    gap = 6
    canvas = np.ones((H + 40, W * 2 + gap, 3), dtype=np.uint8) * 240
    canvas[40:, :W]       = target_img
    canvas[40:, W + gap:] = overlay_img

    if Image:
        img_pil = Image.fromarray(canvas)
        draw    = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((W // 2 - 60, 8), "Original Photo",    fill=(0, 0, 0), font=font)
        draw.text((W + gap + W // 2 - 80, 8), "3D Objects Overlay", fill=(0, 80, 180), font=font)
        img_pil.save(str(output_path))
    else:
        fig, ax = plt.subplots(figsize=((W * 2 + gap) / 100, (H + 40) / 100), dpi=100)
        ax.imshow(canvas); ax.axis("off")
        fig.savefig(str(output_path), dpi=100, bbox_inches="tight"); plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",      type=Path,  default=Path("output/sam3d_dining_v4"))
    parser.add_argument("--output-dir",    type=Path,  default=None)
    parser.add_argument("--icp-threshold", type=float, default=0.3)
    parser.add_argument("--no-scale",      action="store_true")
    parser.add_argument("--overlay-alpha", type=float, default=0.6,
                        help="Alpha for 3D overlay on photo (0=invisible, 1=opaque)")
    parser.add_argument("--target-image", type=Path, default=None,
                        help="Path to the original target photo (auto-detected if not given)")
    args = parser.parse_args()

    data_dir   = args.data_dir.resolve()
    output_dir = (args.output_dir or Path(str(data_dir) + "_icp")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    moge_path   = data_dir / "target_moge.npz"
    transforms_path = data_dir / "object_transforms.json"

    moge = load_moge(moge_path)
    with open(transforms_path, encoding="utf-8") as f:
        objects = json.load(f)

    fx, fy, cx, cy = moge["fx"], moge["fy"], moge["cx"], moge["cy"]
    H, W    = moge["height"], moge["width"]
    depth   = moge["depth"]
    pointmap = moge["points"]  # (H, W, 3) OpenCV space for ICP target

    # MoGe pointmap is in OpenCV space but normals need PT3D space:
    # visualize_convex_hull_growth uses the pointmap directly for cross-product normals
    # which is fine — we just need consistent orientation, not a specific convention

    print(f"Data:      {data_dir}")
    print(f"Output:    {output_dir}")
    print(f"Objects:   {len(objects)}")
    print(f"Image:     {W}x{H}  fx={fx:.1f}")
    print(f"ICP thr:   {args.icp_threshold}")
    print()

    # Load target photograph
    target_img = load_target_image(data_dir, H, W, args.target_image)
    if target_img is None:
        print("  [INFO] Target photo not found - overlays will use depth map as background")

    # Copy MoGe data
    shutil.copy2(moge_path, output_dir / "target_moge.npz")

    # Pre-load all masks
    all_masks: dict[str, np.ndarray] = {}
    for obj in objects:
        name = obj["object_name"]
        mp = data_dir / f"{name}.npy"
        all_masks[name] = (np.load(str(mp)) > 127) if mp.exists() else np.zeros((H, W), dtype=bool)

    verts_before_all: dict[str, np.ndarray] = {}
    verts_after_all:  dict[str, np.ndarray] = {}
    stats_before_all: dict[str, dict] = {}
    stats_after_all:  dict[str, dict] = {}
    results_list = []

    # -----------------------------------------------------------------------
    # Per-object ICP loop
    # -----------------------------------------------------------------------
    for obj in objects:
        name     = obj["object_name"]
        glb_path = data_dir / f"{name}.glb"
        mask_bool = all_masks[name]

        if not glb_path.exists():
            print(f"[SKIP] {name}: GLB not found")
            continue

        print(f"{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        verts = load_glb_vertices(glb_path)
        if len(verts) == 0:
            print(f"  [SKIP] no vertices")
            continue

        # Before stats
        stats_b = compute_depth_error(verts, mask_bool, depth, fx, fy, cx, cy, H, W)
        verts_before_all[name] = verts.copy()
        stats_before_all[name] = stats_b
        print(f"  Before: rel_err={stats_b.get('depth_error_rel_mean', float('nan')):.2%}  "
              f"valid={stats_b.get('valid_count', 0)}")

        # V9 mask growth for ICP target
        print(f"  Growing mask (v9 ray-cast)...")
        grown = grow_mask_v9(mask_bool, pointmap, depth)
        print(f"  Mask: orig={mask_bool.sum():,}  grown={grown.sum():,}  "
              f"(+{grown.sum() - mask_bool.sum():,})")

        # ICP
        verts_aligned, icp_info = run_icp_alignment(
            verts, pointmap, grown, icp_threshold=args.icp_threshold)
        print(f"  ICP: fitness={icp_info['fitness']:.4f}  rmse={icp_info['rmse']:.4f}  "
              f"target={icp_info['n_target']:,}")

        # Depth-scale correction
        s, scale_info = compute_depth_scale(
            verts_aligned, mask_bool, depth, fx, fy, cx, cy, H, W)
        if args.no_scale:
            s = 1.0
        verts_scaled = (verts_aligned * s).astype(np.float32)
        print(f"  Scale: s={s:.4f}  ({scale_info['valid_count']} pts)")

        # After stats
        stats_a = compute_depth_error(verts_scaled, mask_bool, depth, fx, fy, cx, cy, H, W)

        # Rejection gate
        err_b = stats_b.get("depth_error_rel_mean", float("inf"))
        err_a = stats_a.get("depth_error_rel_mean", float("inf"))

        if np.isfinite(err_a) and err_a < err_b:
            verts_final = verts_scaled
            stats_final = stats_a
            accepted = True
            print(f"  After:  rel_err={err_a:.2%}  [ACCEPTED  D={err_b - err_a:.2%}]")
        else:
            verts_final = verts.copy()
            stats_final = stats_b
            accepted = False
            print(f"  After:  rel_err={err_a:.2%}  [REJECTED - keeping original]")

        verts_after_all[name]  = verts_final.copy()
        stats_after_all[name]  = stats_final

        # Save corrected GLB
        if accepted:
            scene_glb = load_glb_scene(glb_path)
            T_icp = np.array(icp_info["transformation"], dtype=np.float64)
            for geom in scene_glb.geometry.values():
                if hasattr(geom, "vertices"):
                    v = np.asarray(geom.vertices, dtype=np.float64)
                    ones = np.ones((len(v), 1), dtype=np.float64)
                    v_new = (np.hstack([v, ones]) @ T_icp.T)[:, :3] * s
                    geom.vertices = v_new.astype(np.float32)
            scene_glb.export(str(output_dir / f"{name}.glb"))
        else:
            shutil.copy2(glb_path, output_dir / f"{name}.glb")

        # Copy mask + png
        for ext in (".npy", ".png"):
            src = data_dir / f"{name}{ext}"
            if src.exists():
                shutil.copy2(src, output_dir / f"{name}{ext}")

        # Per-object compare image
        compare_path = output_dir / f"{name}_compare.png"
        make_per_object_compare(
            target_img, mask_bool, verts, verts_final,
            depth, fx, fy, cx, cy, H, W,
            name, stats_b, stats_final, compare_path,
        )
        print(f"  Saved: {compare_path.name}")

        # Per-object photo overlay (ICP-aligned vertices on original photo)
        if target_img is not None:
            single_vd = {name: verts_final}
            overlay_single = render_scene_on_background(
                single_vd, target_img, fx, fy, cx, cy, H, W,
                alpha=args.overlay_alpha)
            overlay_path = output_dir / f"{name}_overlay.png"
            if Image:
                Image.fromarray(overlay_single).save(str(overlay_path))
            print(f"  Saved: {overlay_path.name}")

        # Save info JSON
        info = {
            "object_name": name,
            "accepted": accepted,
            "depth_scale": s,
            "icp_info": icp_info,
            "scale_info": scale_info,
            "stats_before": stats_b,
            "stats_after": stats_final,
            "n_grown_pixels": int(grown.sum()),
            "n_orig_pixels": int(mask_bool.sum()),
        }
        with open(output_dir / f"{name}_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        results_list.append(info)
        print()

    # -----------------------------------------------------------------------
    # Scene-level renders
    # -----------------------------------------------------------------------
    print("Rendering scene-level outputs...")

    # Depth-map overlays (before / after ICP)
    render_scene_on_depthmap(
        verts_before_all, depth, fx, fy, cx, cy, H, W,
        "Before ICP (v9 mask growth)",
        output_dir / "scene_depth_before.png",
    )
    render_scene_on_depthmap(
        verts_after_all, depth, fx, fy, cx, cy, H, W,
        "After ICP (v9 mask growth)",
        output_dir / "scene_depth_after.png",
    )
    make_scene_before_after_depthmap(
        verts_before_all, verts_after_all,
        depth, fx, fy, cx, cy, H, W,
        output_dir / "scene_2d_comparison.png",
    )
    print("  Saved: scene_depth_before.png, scene_depth_after.png, scene_2d_comparison.png")

    # Photo overlays
    if target_img is not None:
        overlay_all = render_scene_on_background(
            verts_after_all, target_img, fx, fy, cx, cy, H, W,
            alpha=args.overlay_alpha)
        if Image:
            Image.fromarray(overlay_all).save(
                str(output_dir / "scene_photo_overlay.png"))
        make_scene_photo_comparison(
            target_img, overlay_all, H, W,
            output_dir / "scene_photo_comparison.png",
        )
        print("  Saved: scene_photo_overlay.png, scene_photo_comparison.png")

    # Summary JSON
    with open(output_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, default=str)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"{'Object':40s}  {'Before':>7s}  {'After':>7s}  {'Delta':>6s}  Status")
    print("-" * 75)
    n_improved = 0
    for r in results_list:
        nm = r["object_name"]
        b  = r["stats_before"].get("depth_error_rel_mean", float("nan"))
        a  = r["stats_after"].get("depth_error_rel_mean", float("nan"))
        delta = (b - a) if (np.isfinite(b) and np.isfinite(a)) else float("nan")
        ok = "ACCEPTED" if r["accepted"] else "kept"
        if np.isfinite(delta) and delta > 0:
            n_improved += 1
        b_s = f"{b:.1%}" if np.isfinite(b) else "  N/A  "
        a_s = f"{a:.1%}" if np.isfinite(a) else "  N/A  "
        d_s = f"{delta:+.1%}" if np.isfinite(delta) else "  N/A  "
        print(f"  {nm:38s}  {b_s:>7s}  {a_s:>7s}  {d_s:>6s}  {ok}")
    print(f"\n  Improved: {n_improved}/{len(results_list)} objects")
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
