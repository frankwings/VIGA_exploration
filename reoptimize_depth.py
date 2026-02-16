#!/usr/bin/env python
"""Re-optimize SAM3D object poses using ICP with grown masks.

For each object:
  1. Grow the segmentation mask toward its convex hull, stopping at
     depth edges (adaptive threshold based on local Sobel statistics).
  2. Extract 3D points from the MoGe pointmap within the grown mask.
  3. Run Open3D ICP to align mesh vertices to pointmap points.
  4. Save corrected GLBs + before/after visualizations.

Usage:
    python reoptimize_depth.py [--data-dir output/sam3d_dining] [--output-dir output/sam3d_dining_v2]

Run with:  C:/Users/kingy/miniconda3/envs/sam3d_py311/python.exe reoptimize_depth.py
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
    from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
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
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_moge(npz_path: Path) -> dict:
    data = np.load(npz_path)
    K = data["intrinsics_px"].astype(np.float64)
    depth = data["depth"].astype(np.float32)
    points = data["points"].astype(np.float32)
    w, h = int(data["image_width"]), int(data["image_height"])
    return {
        "K": K, "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
        "depth": depth, "width": w, "height": h,
        "points": points,
    }


def load_glb_vertices(glb_path: Path) -> np.ndarray:
    scene = trimesh.load(str(glb_path), force="scene")
    all_verts = []
    for geom in scene.geometry.values():
        if hasattr(geom, "vertices"):
            all_verts.append(np.asarray(geom.vertices, dtype=np.float32))
    return np.concatenate(all_verts, axis=0)


def load_glb_scene(glb_path: Path) -> trimesh.Scene:
    return trimesh.load(str(glb_path), force="scene")


def pt3d_to_opencv(verts: np.ndarray) -> np.ndarray:
    out = verts.copy()
    out[:, 0] = -verts[:, 0]
    out[:, 1] = -verts[:, 1]
    return out


def project(verts_cv, fx, fy, cx, cy):
    z = verts_cv[:, 2]
    u = fx * verts_cv[:, 0] / z + cx
    v = fy * verts_cv[:, 1] / z + cy
    return np.stack([u, v], axis=-1), z


# ---------------------------------------------------------------------------
# Mask-filtered depth error
# ---------------------------------------------------------------------------

def compute_masked_depth_error(verts_pt3d, mask, depth, fx, fy, cx, cy, H, W):
    """Compute depth error only for vertices projecting within the object mask."""
    verts_cv = pt3d_to_opencv(verts_pt3d)
    uv, z_v = project(verts_cv, fx, fy, cx, cy)

    u_int = np.round(uv[:, 0]).astype(np.int32)
    v_int = np.round(uv[:, 1]).astype(np.int32)

    in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)
    in_mask = np.zeros(len(verts_pt3d), dtype=bool)
    in_mask[in_bounds] = mask[v_int[in_bounds], u_int[in_bounds]]

    valid = in_bounds & in_mask
    if valid.sum() == 0:
        return {"valid_count": 0, "depth_error_abs_mean": np.nan}

    z_vertex = z_v[valid]
    z_moge = depth[v_int[valid], u_int[valid]]
    err = z_vertex - z_moge
    abs_err = np.abs(err)
    rel_err = abs_err / np.maximum(z_moge, 1e-6)

    return {
        "valid_count": int(valid.sum()),
        "z_vertex_median": float(np.median(z_vertex)),
        "z_moge_median": float(np.median(z_moge)),
        "depth_error_mean": float(np.mean(err)),
        "depth_error_median": float(np.median(err)),
        "depth_error_abs_mean": float(np.mean(abs_err)),
        "depth_error_abs_max": float(np.max(abs_err)),
        "depth_error_rel_mean": float(np.mean(rel_err)),
        "scale_ratio": float(np.median(z_vertex) / np.median(z_moge)),
    }


# ---------------------------------------------------------------------------
# Convex-hull mask growth with depth-edge stopping
# ---------------------------------------------------------------------------

def compute_depth_sobel(depth: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude of depth map."""
    from scipy.ndimage import sobel
    sx = sobel(depth.astype(np.float64), axis=1)
    sy = sobel(depth.astype(np.float64), axis=0)
    return np.sqrt(sx**2 + sy**2).astype(np.float32)


def make_convex_hull_mask(mask_bool: np.ndarray) -> np.ndarray:
    """Create a filled convex hull mask from a binary mask."""
    rows, cols = np.where(mask_bool)
    if len(rows) < 3:
        return mask_bool.copy()

    points_2d = np.column_stack([cols, rows])  # (N, 2) as (x, y)
    try:
        hull = ConvexHull(points_2d)
    except Exception:
        return mask_bool.copy()

    # Rasterize convex hull using matplotlib path
    from matplotlib.path import Path as MplPath
    hull_pts = points_2d[hull.vertices]
    hull_path = MplPath(hull_pts)

    H, W = mask_bool.shape
    # Test all pixels inside the bounding box of the hull
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()
    # Add small margin
    rmin, rmax = max(0, rmin - 2), min(H - 1, rmax + 2)
    cmin, cmax = max(0, cmin - 2), min(W - 1, cmax + 2)

    yy, xx = np.mgrid[rmin:rmax+1, cmin:cmax+1]
    test_pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = hull_path.contains_points(test_pts).reshape(yy.shape)

    hull_mask = np.zeros_like(mask_bool)
    hull_mask[rmin:rmax+1, cmin:cmax+1] = inside
    return hull_mask


def grow_mask_to_convex_hull(
    mask_bool: np.ndarray,
    depth: np.ndarray,
    sobel_mag: np.ndarray,
    edge_sigma: float = 2.0,
    max_iters: int = 50,
) -> np.ndarray:
    """Grow mask toward convex hull, stopping at depth edges.

    The depth-edge threshold is adaptive: computed as
        threshold = median(sobel within mask) + edge_sigma * std(sobel within mask)
    so it's based on the local Sobel statistics of the existing mask region.

    Args:
        mask_bool: Original binary mask (H, W)
        depth: MoGe depth map (H, W)
        sobel_mag: Sobel gradient magnitude of depth (H, W)
        edge_sigma: Number of std deviations above median to set the threshold
        max_iters: Maximum dilation iterations

    Returns:
        grown_mask: Expanded binary mask (H, W)
    """
    hull_mask = make_convex_hull_mask(mask_bool)

    # Compute adaptive threshold from Sobel values within the original mask
    sobel_in_mask = sobel_mag[mask_bool]
    if len(sobel_in_mask) == 0:
        return mask_bool.copy()
    sobel_median = float(np.median(sobel_in_mask))
    sobel_std = float(np.std(sobel_in_mask))
    threshold = sobel_median + edge_sigma * sobel_std

    # Region we're allowed to grow into: inside hull but not original mask
    growth_region = hull_mask & ~mask_bool

    # Pixels in the growth region that are NOT depth edges
    safe_pixels = growth_region & (sobel_mag < threshold)

    # Iterative dilation: expand mask one pixel at a time, but only into safe pixels
    struct = np.ones((3, 3), dtype=bool)
    grown = mask_bool.copy()

    for i in range(max_iters):
        dilated = binary_dilation(grown, structure=struct)
        # Only add pixels that are safe (inside hull, below edge threshold)
        new_pixels = dilated & safe_pixels & ~grown
        if not new_pixels.any():
            break
        grown = grown | new_pixels

    return grown


# ---------------------------------------------------------------------------
# Mask growth visualization
# ---------------------------------------------------------------------------

def visualize_mask_growth(
    mask_bool: np.ndarray,
    grown_mask: np.ndarray,
    hull_mask: np.ndarray,
    depth: np.ndarray,
    sobel_mag: np.ndarray,
    threshold: float,
    obj_name: str,
    output_path: Path,
):
    """4-panel visualization: depth + sobel | original mask | grown mask | overlay."""
    H, W = mask_bool.shape
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Mask Growth: {obj_name}", fontsize=14, fontweight="bold")

    # Panel 1: Depth map with Sobel edges highlighted
    ax = axes[0, 0]
    ax.imshow(depth, cmap="turbo", vmin=depth.min(), vmax=depth.max())
    edge_overlay = np.zeros((H, W, 4))
    edge_overlay[sobel_mag >= threshold] = [1, 0, 0, 0.5]  # red for edges
    ax.imshow(edge_overlay)
    ax.set_title(f"Depth + Edges (threshold={threshold:.4f})")
    ax.axis("off")

    # Panel 2: Original mask vs convex hull
    ax = axes[0, 1]
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[hull_mask] = [60, 60, 100]       # dark blue = hull
    vis[mask_bool] = [0, 200, 0]         # green = original mask
    hull_edge = hull_mask & ~binary_erosion(hull_mask, iterations=1)
    vis[hull_edge] = [255, 255, 0]       # yellow = hull boundary
    ax.imshow(vis)
    n_orig = mask_bool.sum()
    n_hull = hull_mask.sum()
    ax.set_title(f"Original mask ({n_orig:,}px) + Convex hull ({n_hull:,}px)")
    ax.axis("off")

    # Panel 3: Grown mask
    ax = axes[1, 0]
    vis2 = np.zeros((H, W, 3), dtype=np.uint8)
    vis2[grown_mask] = [0, 150, 200]     # cyan = grown
    vis2[mask_bool] = [0, 200, 0]        # green = original
    added = grown_mask & ~mask_bool
    vis2[added] = [255, 165, 0]          # orange = newly added
    ax.imshow(vis2)
    n_grown = grown_mask.sum()
    n_added = added.sum()
    ax.set_title(f"Grown mask ({n_grown:,}px, +{n_added:,} added)")
    ax.axis("off")

    # Panel 4: Sobel histogram with threshold
    ax = axes[1, 1]
    sobel_in_mask = sobel_mag[mask_bool]
    sobel_in_growth = sobel_mag[hull_mask & ~mask_bool]
    if len(sobel_in_mask) > 0:
        ax.hist(sobel_in_mask.ravel(), bins=100, alpha=0.7, color="green",
                label="Inside mask", density=True)
    if len(sobel_in_growth) > 0:
        ax.hist(sobel_in_growth.ravel(), bins=100, alpha=0.5, color="orange",
                label="Growth region", density=True)
    ax.axvline(threshold, color="red", ls="--", lw=2, label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Sobel gradient magnitude")
    ax.set_ylabel("Density")
    ax.set_title("Depth gradient distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(threshold * 5, sobel_mag.max()))

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# ICP optimization using Open3D
# ---------------------------------------------------------------------------

def run_icp_alignment(
    verts_pt3d: np.ndarray,
    pointmap: np.ndarray,
    grown_mask: np.ndarray,
    icp_threshold: float = 0.3,
    depth_quantile: float = 0.9,
    max_source_points: int = 20000,
) -> tuple[np.ndarray, dict]:
    """Run ICP to align mesh vertices to MoGe pointmap within the grown mask.

    The mesh vertices are in PyTorch3D camera space (X-left, Y-up, Z-forward).
    The MoGe pointmap is in OpenCV camera space (X-right, Y-down, Z-forward).
    ICP runs in OpenCV space; the result is converted back to PT3D space.

    Args:
        verts_pt3d: (N, 3) mesh vertices in PT3D space
        pointmap: (H, W, 3) MoGe pointmap in OpenCV camera space
        grown_mask: (H, W) binary mask (grown)
        icp_threshold: Max correspondence distance for ICP
        depth_quantile: Filter outlier target points by depth percentile
        max_source_points: Subsample source points for speed

    Returns:
        verts_aligned: (N, 3) aligned vertices in PT3D space
        info: dict with ICP stats
    """
    # Extract target points from pointmap within grown mask (OpenCV space)
    target_pts = pointmap[grown_mask]  # (M, 3)

    # Filter depth outliers
    if len(target_pts) > 0 and depth_quantile < 1.0:
        z_vals = target_pts[:, 2]  # Z = depth (forward) in OpenCV
        z_thresh = np.quantile(z_vals, depth_quantile)
        target_pts = target_pts[z_vals <= z_thresh]

    if len(target_pts) < 10:
        return verts_pt3d.copy(), {"fitness": 0, "rmse": 0, "n_target": 0, "n_source": 0}

    # Convert source vertices from PT3D to OpenCV space for ICP
    # OpenCV: X_cv = -X_pt3d, Y_cv = -Y_pt3d, Z_cv = Z_pt3d
    source_pts_cv = pt3d_to_opencv(verts_pt3d)

    # Subsample source points if too many
    if len(source_pts_cv) > max_source_points:
        idx = np.random.default_rng(42).choice(len(source_pts_cv), max_source_points, replace=False)
        source_pts_sub = source_pts_cv[idx]
    else:
        source_pts_sub = source_pts_cv

    # Create Open3D point clouds
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(source_pts_sub.astype(np.float64))

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(target_pts.astype(np.float64))

    # Run ICP in OpenCV space
    reg = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd,
        max_correspondence_distance=icp_threshold,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )

    T_cv = reg.transformation  # 4x4 in OpenCV space
    fitness = reg.fitness
    rmse = reg.inlier_rmse

    # Convert transformation from OpenCV space back to PT3D space:
    # T_pt3d = M @ T_cv @ M  where M = diag(-1, -1, 1, 1)
    M = np.diag([-1.0, -1.0, 1.0, 1.0])
    T_pt3d = M @ T_cv @ M

    # Apply transformation to ALL vertices in PT3D space
    ones = np.ones((len(verts_pt3d), 1), dtype=np.float64)
    verts_h = np.hstack([verts_pt3d.astype(np.float64), ones])  # (N, 4)
    verts_aligned = (verts_h @ T_pt3d.T)[:, :3].astype(np.float32)

    info = {
        "fitness": float(fitness),
        "rmse": float(rmse),
        "n_target": len(target_pts),
        "n_source": len(source_pts_sub),
        "transformation": T_pt3d.tolist(),
        "transformation_cv": T_cv.tolist(),
    }
    return verts_aligned, info


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def render_projected_vertices(verts_pt3d, fx, fy, cx, cy, H, W, depth_map):
    """Create a synthetic render by projecting vertices and colouring by depth."""
    verts_cv = pt3d_to_opencv(verts_pt3d)
    uv, z_v = project(verts_cv, fx, fy, cx, cy)

    u_int = np.round(uv[:, 0]).astype(np.int32)
    v_int = np.round(uv[:, 1]).astype(np.int32)
    valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)

    vmin, vmax = depth_map.min(), depth_map.max()
    render = np.ones((H, W, 3), dtype=np.uint8) * 240

    if valid.sum() > 0:
        u_v = u_int[valid]
        v_v = v_int[valid]
        z_valid = z_v[valid]
        z_norm = np.clip((z_valid - vmin) / (vmax - vmin + 1e-6), 0, 1)

        r = np.clip((1.0 - z_norm) * 255, 0, 255).astype(np.uint8)
        g = np.clip(z_norm * 200 + 55, 0, 255).astype(np.uint8)
        b = np.clip(z_norm * 255, 0, 255).astype(np.uint8)

        order = np.argsort(-z_valid)
        flat_idx = v_v[order] * W + u_v[order]
        render_flat = render.reshape(-1, 3)
        colors = np.stack([r[order], g[order], b[order]], axis=-1)
        render_flat[flat_idx] = colors
        render = render_flat.reshape(H, W, 3)

    return render


def make_comparison_image(
    input_png_path, mask, verts_before, verts_after, depth,
    fx, fy, cx, cy, H, W, obj_name, stats_before, stats_after, output_path,
):
    """Create a 3-panel comparison: input mask | before render | after render."""
    if input_png_path and input_png_path.exists() and Image:
        img = Image.open(input_png_path).convert("RGB").resize((W, H))
        panel1 = np.array(img)
        mask_bool = mask > 127 if mask.dtype == np.uint8 else mask > 0.5
        edge = mask_bool & ~binary_erosion(mask_bool, iterations=2)
        panel1[edge] = [0, 255, 0]
    else:
        panel1 = np.zeros((H, W, 3), dtype=np.uint8)
        mask_bool = mask > 127 if mask.dtype == np.uint8 else mask > 0.5
        panel1[mask_bool] = [128, 128, 128]

    panel2 = render_projected_vertices(verts_before, fx, fy, cx, cy, H, W, depth)
    panel3 = render_projected_vertices(verts_after, fx, fy, cx, cy, H, W, depth)

    gap = 4
    canvas_w = W * 3 + gap * 2
    canvas = np.ones((H + 40, canvas_w, 3), dtype=np.uint8) * 255
    canvas[40:, :W] = panel1
    canvas[40:, W + gap: W * 2 + gap] = panel2
    canvas[40:, W * 2 + gap * 2:] = panel3

    if Image:
        img_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        err_b = stats_before.get("depth_error_rel_mean", 0)
        err_a = stats_after.get("depth_error_rel_mean", 0)

        draw.text((4, 4), f"{obj_name} - Mask", fill=(0, 0, 0), font=font)
        draw.text((W + gap + 4, 4), f"Before (rel_err={err_b:.1%})", fill=(180, 0, 0), font=font)
        draw.text((W * 2 + gap * 2 + 4, 4), f"After (rel_err={err_a:.1%})", fill=(0, 128, 0), font=font)
        img_pil.save(str(output_path))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(canvas_w / 100, (H + 40) / 100), dpi=100)
        ax.imshow(canvas)
        ax.axis("off")
        fig.savefig(str(output_path), dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()


def make_dashboard(objects, stats_before_all, stats_after_all, output_dir):
    """Before/after bar chart dashboard."""
    names = list(stats_before_all.keys())
    err_b = [stats_before_all[n].get("depth_error_rel_mean", 0) * 100 for n in names]
    err_a = [stats_after_all[n].get("depth_error_rel_mean", 0) * 100 for n in names]

    order = np.argsort(err_b)[::-1]
    names_s = [names[i] for i in order]
    err_b_s = [err_b[i] for i in order]
    err_a_s = [err_a[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(names_s))
    bar_h = 0.35
    ax.barh(y + bar_h / 2, err_b_s, bar_h, color="salmon", edgecolor="gray", label="Before")
    ax.barh(y - bar_h / 2, err_a_s, bar_h, color="mediumseagreen", edgecolor="gray", label="After")

    ax.set_yticks(y)
    ax.set_yticklabels(names_s, fontsize=9)
    ax.set_xlabel("Relative Depth Error (%)")
    ax.set_title("Depth Error: Before vs After ICP Re-optimization", fontweight="bold")
    ax.legend()
    ax.axvline(5, color="green", ls="--", lw=0.8, alpha=0.4)
    ax.axvline(15, color="orange", ls="--", lw=0.8, alpha=0.4)

    for i, (b, a) in enumerate(zip(err_b_s, err_a_s)):
        improvement = b - a
        symbol = "+" if improvement > 0 else ""
        ax.text(max(b, a) + 0.5, i, f"{symbol}{improvement:.1f}pp", va="center", fontsize=7,
                color="green" if improvement > 0 else "red")

    plt.tight_layout()
    fig.savefig(str(output_dir / "depth_diagnostic_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Re-optimize SAM3D poses with ICP + grown masks")
    parser.add_argument("--data-dir", type=Path, default=Path("output/sam3d_dining"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <data-dir>_v2)")
    parser.add_argument("--icp-threshold", type=float, default=0.3,
                        help="ICP max correspondence distance")
    parser.add_argument("--edge-sigma", type=float, default=2.0,
                        help="Depth edge threshold = median + edge_sigma * std")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or Path(str(data_dir) + "_v2")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    moge_path = data_dir / "target_moge.npz"
    transforms_path = data_dir / "object_transforms.json"

    moge = load_moge(moge_path)
    with open(transforms_path, encoding="utf-8") as f:
        objects = json.load(f)

    fx, fy, cx, cy = moge["fx"], moge["fy"], moge["cx"], moge["cy"]
    H, W = moge["height"], moge["width"]
    depth = moge["depth"]
    pointmap = moge["points"]  # (H, W, 3) in PT3D camera space

    # Precompute Sobel gradient of depth map (used for all objects)
    sobel_mag = compute_depth_sobel(depth)

    # Copy MoGe npz to output dir
    shutil.copy2(moge_path, output_dir / "target_moge.npz")

    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Objects: {len(objects)}")
    print(f"Image: {W}x{H}, fx={fx:.2f}")
    print(f"ICP threshold: {args.icp_threshold}")
    print(f"Edge sigma: {args.edge_sigma}")
    print()

    new_transforms = []
    stats_before_all = {}
    stats_after_all = {}
    verts_before_all = {}
    verts_after_all = {}

    for obj in objects:
        name = obj["object_name"]
        glb_path = data_dir / f"{name}.glb"
        mask_path = data_dir / f"{name}.npy"
        png_path = data_dir / f"{name}.png"

        if not glb_path.exists():
            print(f"[SKIP] {name}: GLB not found")
            continue

        print(f"{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")

        # Load data
        verts = load_glb_vertices(glb_path)
        mask = np.load(str(mask_path)) if mask_path.exists() else np.ones((H, W), dtype=np.uint8) * 255
        mask_bool = mask > 127

        centroid = np.array(obj["translation"], dtype=np.float32)

        # --- Before stats ---
        stats_b = compute_masked_depth_error(verts, mask_bool, depth, fx, fy, cx, cy, H, W)
        stats_before_all[name] = stats_b
        verts_before_all[name] = verts.copy()

        print(f"  BEFORE: rel_err={stats_b.get('depth_error_rel_mean', 0):.2%}  "
              f"|err|={stats_b.get('depth_error_abs_mean', 0):.4f}  "
              f"ratio={stats_b.get('scale_ratio', 0):.4f}  "
              f"valid={stats_b.get('valid_count', 0)}")

        # --- Grow mask toward convex hull ---
        hull_mask = make_convex_hull_mask(mask_bool)
        grown_mask = grow_mask_to_convex_hull(
            mask_bool, depth, sobel_mag, edge_sigma=args.edge_sigma,
        )

        # Compute threshold for visualization
        sobel_in_mask = sobel_mag[mask_bool]
        if len(sobel_in_mask) > 0:
            edge_threshold = float(np.median(sobel_in_mask) + args.edge_sigma * np.std(sobel_in_mask))
        else:
            edge_threshold = 0.0

        n_orig = int(mask_bool.sum())
        n_grown = int(grown_mask.sum())
        n_hull = int(hull_mask.sum())
        print(f"  Mask: original={n_orig:,}  hull={n_hull:,}  grown={n_grown:,}  "
              f"(+{n_grown - n_orig:,} pixels, edge_thresh={edge_threshold:.4f})")

        # --- Visualize mask growth ---
        vis_path = output_dir / f"{name}_mask_growth.png"
        visualize_mask_growth(
            mask_bool, grown_mask, hull_mask, depth, sobel_mag,
            edge_threshold, name, vis_path,
        )
        print(f"  Saved: {vis_path.name}")

        # --- Run ICP ---
        verts_aligned, icp_info = run_icp_alignment(
            verts, pointmap, grown_mask,
            icp_threshold=args.icp_threshold,
        )
        verts_after_all[name] = verts_aligned.copy()

        print(f"  ICP: fitness={icp_info['fitness']:.4f}  "
              f"rmse={icp_info['rmse']:.4f}  "
              f"target_pts={icp_info['n_target']:,}  "
              f"source_pts={icp_info['n_source']:,}")

        # --- After stats ---
        stats_a = compute_masked_depth_error(verts_aligned, mask_bool, depth, fx, fy, cx, cy, H, W)

        improvement = (stats_b.get("depth_error_rel_mean", 0) - stats_a.get("depth_error_rel_mean", 0))
        symbol = "+" if improvement > 0 else ""

        # Safety: skip ICP if it made things worse
        applied_icp = True
        if improvement < 0:
            print(f"  ICP made depth worse ({improvement:+.2%}), keeping original pose")
            verts_aligned = verts.copy()
            stats_a = stats_b
            icp_info["transformation"] = np.eye(4).tolist()
            icp_info["skipped"] = True
            applied_icp = False
        else:
            print(f"  AFTER:  rel_err={stats_a.get('depth_error_rel_mean', 0):.2%}  "
                  f"|err|={stats_a.get('depth_error_abs_mean', 0):.4f}  "
                  f"ratio={stats_a.get('scale_ratio', 0):.4f}  "
                  f"({symbol}{improvement:.2%} improvement)")

        verts_after_all[name] = verts_aligned.copy()
        stats_after_all[name] = stats_a

        # --- Save corrected GLB ---
        scene = load_glb_scene(glb_path)
        T_icp = np.array(icp_info["transformation"], dtype=np.float64)
        if applied_icp:
            for geom in scene.geometry.values():
                if hasattr(geom, "vertices"):
                    v = np.asarray(geom.vertices, dtype=np.float64)
                    ones = np.ones((len(v), 1), dtype=np.float64)
                    v_h = np.hstack([v, ones])
                    v_aligned = (v_h @ T_icp.T)[:, :3]
                    geom.vertices = v_aligned.astype(np.float32)
        out_glb = output_dir / f"{name}.glb"
        scene.export(str(out_glb))

        # Copy mask and PNG
        if mask_path.exists():
            shutil.copy2(mask_path, output_dir / f"{name}.npy")
        if png_path.exists():
            shutil.copy2(png_path, output_dir / f"{name}.png")

        # --- Save info JSON ---
        new_transform = {
            "glb_path": str(out_glb),
            "translation": obj["translation"],
            "rotation": obj["rotation"],
            "scale": obj["scale"],
            "pointmap_shape": obj.get("pointmap_shape", [H, W, 3]),
            "object_name": name,
            "icp_transformation": icp_info["transformation"],
        }
        new_transforms.append(new_transform)

        info = {**new_transform, "icp_info": icp_info}
        with open(output_dir / f"{name}_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        # --- Per-object comparison image ---
        compare_path = output_dir / f"{name}_compare.png"
        make_comparison_image(
            png_path if png_path.exists() else None,
            mask, verts, verts_aligned, depth, fx, fy, cx, cy, H, W,
            name, stats_b, stats_a, compare_path,
        )
        print(f"  Saved: {compare_path.name}")

        # --- Per-object render (after only) ---
        render_img = render_projected_vertices(verts_aligned, fx, fy, cx, cy, H, W, depth)
        if Image:
            Image.fromarray(render_img).save(str(output_dir / f"{name}_render.png"))

        print()

    # --- Save new transforms ---
    with open(output_dir / "object_transforms.json", "w", encoding="utf-8") as f:
        json.dump(new_transforms, f, indent=2)

    # --- Save results JSON ---
    results = {"before": stats_before_all, "after": stats_after_all}
    with open(output_dir / "depth_alignment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # --- Dashboard ---
    print("Generating dashboard...")
    make_dashboard(objects, stats_before_all, stats_after_all, output_dir)

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Object':35s}  {'Before':>8s}  {'After':>8s}  {'Change':>8s}")
    print("-" * 65)
    for name in stats_before_all:
        b = stats_before_all[name].get("depth_error_rel_mean", 0) * 100
        a = stats_after_all.get(name, {}).get("depth_error_rel_mean", 0) * 100
        delta = b - a
        print(f"  {name:33s}  {b:7.1f}%  {a:7.1f}%  {delta:+7.1f}pp")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
