#!/usr/bin/env python
"""Re-optimize SAM3D object poses with mask-filtered depth loss.

Loads existing GLBs (vertices in PyTorch3D camera space), re-optimizes
a translation + scale correction per object using:
  - Silhouette loss: projected vertices vs GT mask
  - Depth loss: projected vertex Z vs MoGe depth (within mask only)

Outputs new GLBs, transforms JSON, per-object comparison PNGs, and
diagnostic scatter/dashboard figures to a new results folder.

Usage:
    python reoptimize_depth.py [--data-dir output/sam3d_dining] [--output-dir output/sam3d_dining_v2]

Requires: numpy, trimesh, scipy, matplotlib, Pillow  (the `agent` conda env)
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
    from scipy.optimize import minimize
    from scipy.ndimage import binary_dilation
except ImportError:
    sys.exit("scipy not installed.  pip install scipy")

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
    w, h = int(data["image_width"]), int(data["image_height"])
    return {
        "K": K, "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
        "depth": depth, "width": w, "height": h,
        "points": data["points"].astype(np.float32),
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
# Mask-filtered depth error (fixes issue #1)
# ---------------------------------------------------------------------------

def compute_masked_depth_error(verts_pt3d, mask, depth, fx, fy, cx, cy, H, W):
    """Compute depth error only for vertices projecting within the object mask."""
    verts_cv = pt3d_to_opencv(verts_pt3d)
    uv, z_v = project(verts_cv, fx, fy, cx, cy)

    u_int = np.round(uv[:, 0]).astype(np.int32)
    v_int = np.round(uv[:, 1]).astype(np.int32)

    # Filter: in bounds AND within mask
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
# Ray-cast mesh to get per-pixel depth
# ---------------------------------------------------------------------------

def raycast_mesh_depth(
    glb_path: Path,
    mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    max_rays: int = 10000,
    max_faces: int = 8000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cast rays from camera through mask pixels, intersect with mesh.

    The mesh is in PyTorch3D camera space.  We convert to OpenCV for
    ray generation, then cast rays through the mesh in OpenCV space.

    High-poly meshes are simplified to max_faces triangles before
    ray-casting to avoid OOM in trimesh's ray-triangle intersection.

    Returns:
        pixel_rows:  (N,) row indices of mask pixels that hit the mesh
        pixel_cols:  (N,) col indices
        hit_depths:  (N,) depth (Z) at each hit in OpenCV space
    """
    mask_bool = mask > 127 if mask.dtype == np.uint8 else mask > 0.5

    # Get mask pixel coordinates
    rows, cols = np.where(mask_bool)
    n_mask = len(rows)
    if n_mask == 0:
        return np.array([]), np.array([]), np.array([])

    # Subsample if too many pixels
    if n_mask > max_rays:
        idx = np.random.default_rng(42).choice(n_mask, max_rays, replace=False)
        rows, cols = rows[idx], cols[idx]

    # Build rays in OpenCV camera space: origin at (0,0,0), direction through each pixel
    dx = (cols.astype(np.float64) - cx) / fx
    dy = (rows.astype(np.float64) - cy) / fy
    dz = np.ones_like(dx)
    directions = np.stack([dx, dy, dz], axis=-1)
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions /= norms

    origins = np.zeros_like(directions)

    # Load mesh and convert to OpenCV space for ray intersection
    scene = trimesh.load(str(glb_path), force="scene")
    all_verts = []
    all_faces = []
    face_offset = 0
    for geom in scene.geometry.values():
        if hasattr(geom, "vertices") and hasattr(geom, "faces"):
            v = np.asarray(geom.vertices, dtype=np.float64)
            f = np.asarray(geom.faces, dtype=np.int64) + face_offset
            all_verts.append(v)
            all_faces.append(f)
            face_offset += len(v)

    if not all_verts:
        return np.array([]), np.array([]), np.array([])

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)

    # PT3D -> OpenCV
    verts_cv = verts.copy()
    verts_cv[:, 0] = -verts[:, 0]
    verts_cv[:, 1] = -verts[:, 1]

    mesh_cv = trimesh.Trimesh(vertices=verts_cv, faces=faces, process=False)

    # Simplify high-poly meshes to avoid OOM in ray-triangle intersection
    # trimesh's ray_triangle allocates (n_faces * n_rays, 3) which can be huge
    n_faces = len(mesh_cv.faces)
    if n_faces > max_faces:
        print(f"    Simplifying mesh: {n_faces} -> {max_faces} faces")
        try:
            import fast_simplification
            pts_s, tri_s = fast_simplification.simplify(
                np.asarray(mesh_cv.vertices, dtype=np.float64),
                np.asarray(mesh_cv.faces, dtype=np.int32),
                target_count=max_faces,
            )
            mesh_cv = trimesh.Trimesh(vertices=pts_s, faces=tri_s, process=False)
            print(f"    After simplification: {len(mesh_cv.faces)} faces")
        except Exception as e:
            # Fallback: reduce rays so n_faces * n_rays < 100M
            print(f"    Simplification failed ({e}), reducing rays instead")
            safe_rays = min(len(rows), max(500, 100_000_000 // n_faces))
            if safe_rays < len(rows):
                idx = np.random.default_rng(42).choice(len(rows), safe_rays, replace=False)
                rows, cols = rows[idx], cols[idx]
                dx = (cols.astype(np.float64) - cx) / fx
                dy = (rows.astype(np.float64) - cy) / fy
                dz = np.ones_like(dx)
                directions = np.stack([dx, dy, dz], axis=-1)
                norms = np.linalg.norm(directions, axis=-1, keepdims=True)
                directions /= norms
                origins = np.zeros_like(directions)
                print(f"    Reduced rays to {safe_rays}")

    # Ray-cast
    locations, index_ray, index_tri = mesh_cv.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=False
    )

    if len(locations) == 0:
        return np.array([]), np.array([]), np.array([])

    hit_depths = locations[:, 2]  # Z in OpenCV = depth
    hit_rows = rows[index_ray]
    hit_cols = cols[index_ray]

    return hit_rows, hit_cols, hit_depths


# ---------------------------------------------------------------------------
# Re-optimization using per-pixel depth (closed-form least squares)
# ---------------------------------------------------------------------------

def optimize_object_pose(
    verts_pt3d: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    centroid_pt3d: np.ndarray,
    glb_path: Path = None,
    lambda_depth: float = 1.0,
    lambda_mask: float = 0.5,
    lambda_reg: float = 0.01,
    max_ds: float = 0.25,
    skip_threshold: float = 0.03,
) -> dict:
    """Optimize translation + scale correction using ALL mask pixel depths.

    1. Ray-cast mesh to get per-pixel depth at mask pixels
    2. Solve closed-form least squares for (dt_z, ds):
         z_corrected = (z_mesh - z_centroid) * (1 + ds) + z_centroid + dt_z
       which is linear in (dt_z, ds).

    ds is clamped to [-max_ds, +max_ds] to prevent extreme scale changes.
    Objects with rel_err < skip_threshold are skipped (already well-aligned).
    """
    if glb_path is None:
        return _optimize_vertex_only(
            verts_pt3d, mask, depth, fx, fy, cx, cy, H, W, centroid_pt3d
        )

    # Check if already well-aligned: skip optimization for good objects
    quick_stats = compute_masked_depth_error(
        verts_pt3d, mask > 127 if mask.dtype == np.uint8 else mask > 0.5,
        depth, fx, fy, cx, cy, H, W
    )
    rel_err = quick_stats.get("depth_error_rel_mean", 1.0)
    if rel_err < skip_threshold:
        print(f"    Skipping optimization (rel_err={rel_err:.2%} < {skip_threshold:.0%} threshold)")
        return {"dt": [0.0, 0.0, 0.0], "ds": 0.0, "loss": 0.0, "success": True, "niter": 0, "n_rays": 0}

    # Step 1: Ray-cast to get mesh depth at mask pixels
    hit_rows, hit_cols, z_mesh = raycast_mesh_depth(
        glb_path, mask, fx, fy, cx, cy, H, W, max_rays=15000
    )

    if len(z_mesh) < 20:
        print(f"    WARNING: Only {len(z_mesh)} ray hits, falling back to vertex-only")
        return _optimize_vertex_only(
            verts_pt3d, mask, depth, fx, fy, cx, cy, H, W, centroid_pt3d
        )

    # Step 2: Get MoGe depth at the same pixels
    z_moge = depth[hit_rows, hit_cols].astype(np.float64)
    z_mesh = z_mesh.astype(np.float64)

    z_centroid = float(centroid_pt3d[2])  # Z in PT3D = Z in OpenCV

    print(f"    Ray-cast: {len(z_mesh)} hits, "
          f"mesh_z=[{z_mesh.min():.3f},{z_mesh.max():.3f}], "
          f"moge_z=[{z_moge.min():.3f},{z_moge.max():.3f}]")

    # Step 3: Closed-form least squares
    # z_corrected = z_mesh + (z_mesh - z_centroid) * ds + dt_z
    # Minimize: ||z_corrected - z_moge||^2 + lambda_reg * (ds^2 + dt_z^2)
    a = z_mesh - z_centroid  # (N,)
    b = z_moge - z_mesh       # (N,)
    N = len(a)

    # Build regularized least squares: [A; sqrt(reg)*I] @ x = [b; 0]
    reg_w = np.sqrt(lambda_reg * N)
    A = np.column_stack([a, np.ones(N)])  # (N, 2)
    A_reg = np.vstack([A, reg_w * np.eye(2)])
    b_reg = np.concatenate([b, np.zeros(2)])

    x, residuals, rank, sv = np.linalg.lstsq(A_reg, b_reg, rcond=None)
    ds = float(np.clip(x[0], -max_ds, max_ds))
    dt_z = float(x[1])

    # If ds was clamped, re-solve for dt_z alone with ds fixed
    if abs(x[0]) > max_ds:
        print(f"    Clamped ds from {x[0]:+.4f} to {ds:+.4f}, re-solving dt_z")
        residual_after_ds = b - a * ds  # what dt_z needs to fix
        dt_z = float(np.median(residual_after_ds))  # robust median

    # Compute loss for reporting
    z_corrected = z_mesh + a * ds + dt_z
    err = z_corrected - z_moge
    loss = float(np.mean(err**2))

    return {
        "dt": [0.0, 0.0, dt_z],
        "ds": ds,
        "loss": loss,
        "success": True,
        "niter": 1,
        "n_rays": len(z_mesh),
    }


def _optimize_vertex_only(verts_pt3d, mask, depth, fx, fy, cx, cy, H, W, centroid_pt3d):
    """Fallback: vertex-only depth optimization (old approach)."""
    mask_bool = mask > 127 if mask.dtype == np.uint8 else mask > 0.5
    verts_cv = pt3d_to_opencv(verts_pt3d)
    uv, z_v = project(verts_cv, fx, fy, cx, cy)
    u_int = np.round(uv[:, 0]).astype(np.int32)
    v_int = np.round(uv[:, 1]).astype(np.int32)
    in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)
    in_mask = np.zeros(len(verts_pt3d), dtype=bool)
    in_mask[in_bounds] = mask_bool[v_int[in_bounds], u_int[in_bounds]]
    valid = in_bounds & in_mask

    if valid.sum() < 20:
        return {"dt": [0, 0, 0], "ds": 0.0, "loss": 0.0, "success": False, "niter": 0}

    z_mesh = z_v[valid].astype(np.float64)
    z_moge = depth[v_int[valid], u_int[valid]].astype(np.float64)
    z_centroid = float(centroid_pt3d[2])

    a = z_mesh - z_centroid
    b = z_moge - z_mesh
    N = len(a)
    reg_w = np.sqrt(0.01 * N)
    A = np.column_stack([a, np.ones(N)])
    A_reg = np.vstack([A, reg_w * np.eye(2)])
    b_reg = np.concatenate([b, np.zeros(2)])
    x, _, _, _ = np.linalg.lstsq(A_reg, b_reg, rcond=None)

    return {"dt": [0.0, 0.0, float(x[1])], "ds": float(x[0]), "loss": 0.0, "success": True, "niter": 1}


def apply_correction(verts_pt3d: np.ndarray, centroid: np.ndarray,
                     dt: np.ndarray, ds: float) -> np.ndarray:
    """Apply translation + scale correction to vertices."""
    return (verts_pt3d - centroid) * (1.0 + ds) + centroid + dt


# ---------------------------------------------------------------------------
# Visualization (issue #3)
# ---------------------------------------------------------------------------

def render_projected_vertices(verts_pt3d, fx, fy, cx, cy, H, W, depth_map):
    """Create a synthetic render by projecting vertices and colouring by depth."""
    verts_cv = pt3d_to_opencv(verts_pt3d)
    uv, z_v = project(verts_cv, fx, fy, cx, cy)

    u_int = np.round(uv[:, 0]).astype(np.int32)
    v_int = np.round(uv[:, 1]).astype(np.int32)
    valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H) & (z_v > 0)

    vmin, vmax = depth_map.min(), depth_map.max()
    render = np.ones((H, W, 3), dtype=np.uint8) * 240  # light gray bg

    if valid.sum() > 0:
        u_v = u_int[valid]
        v_v = v_int[valid]
        z_valid = z_v[valid]
        z_norm = np.clip((z_valid - vmin) / (vmax - vmin + 1e-6), 0, 1)

        r = np.clip((1.0 - z_norm) * 255, 0, 255).astype(np.uint8)
        g = np.clip(z_norm * 200 + 55, 0, 255).astype(np.uint8)
        b = np.clip(z_norm * 255, 0, 255).astype(np.uint8)

        # Vectorized z-buffer: sort by depth (far to near), then scatter
        # Later (nearer) writes overwrite earlier (farther) ones
        order = np.argsort(-z_valid)  # far to near
        flat_idx = v_v[order] * W + u_v[order]
        render_flat = render.reshape(-1, 3)
        colors = np.stack([r[order], g[order], b[order]], axis=-1)
        render_flat[flat_idx] = colors
        render = render_flat.reshape(H, W, 3)

    return render


def make_comparison_image(
    input_png_path: Path | None,
    mask: np.ndarray,
    verts_before: np.ndarray,
    verts_after: np.ndarray,
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    H: int, W: int,
    obj_name: str,
    stats_before: dict,
    stats_after: dict,
    output_path: Path,
):
    """Create a 3-panel comparison: input mask | before render | after render."""
    # Panel 1: Mask overlay
    if input_png_path and input_png_path.exists() and Image:
        img = Image.open(input_png_path).convert("RGB").resize((W, H))
        panel1 = np.array(img)
        # Overlay mask boundary
        mask_bool = mask > 127 if mask.dtype == np.uint8 else mask > 0.5
        from scipy.ndimage import binary_erosion
        edge = mask_bool & ~binary_erosion(mask_bool, iterations=2)
        panel1[edge] = [0, 255, 0]
    else:
        panel1 = np.zeros((H, W, 3), dtype=np.uint8)
        mask_bool = mask > 127 if mask.dtype == np.uint8 else mask > 0.5
        panel1[mask_bool] = [128, 128, 128]

    # Panel 2: Before (projected vertices)
    panel2 = render_projected_vertices(verts_before, fx, fy, cx, cy, H, W, depth)

    # Panel 3: After (projected vertices)
    panel3 = render_projected_vertices(verts_after, fx, fy, cx, cy, H, W, depth)

    # Assemble
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
        # Fallback: save with matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(canvas_w / 100, (H + 40) / 100), dpi=100)
        ax.imshow(canvas)
        ax.axis("off")
        fig.savefig(str(output_path), dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()


def make_full_scene_comparison(
    moge: dict,
    objects: list[dict],
    data_dir: Path,
    output_dir: Path,
    verts_before_all: dict,
    verts_after_all: dict,
):
    """Create full-scene overlay: all objects projected before/after."""
    H, W = moge["height"], moge["width"]
    fx, fy, cx, cy = moge["fx"], moge["fy"], moge["cx"], moge["cy"]
    depth = moge["depth"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    fig.suptitle("Full Scene: Before vs After Depth Re-optimization", fontsize=14, fontweight="bold")

    for panel_idx, (title, verts_dict) in enumerate([
        ("Before", verts_before_all), ("After", verts_after_all)
    ]):
        ax = axes[panel_idx]
        # Show MoGe depth as background
        ax.imshow(depth, cmap="gray", alpha=0.3, vmin=depth.min(), vmax=depth.max())

        colors = plt.cm.tab10(np.linspace(0, 1, len(objects)))
        for i, obj in enumerate(objects):
            name = obj["object_name"]
            if name not in verts_dict:
                continue
            v = verts_dict[name]
            v_cv = pt3d_to_opencv(v)
            uv, z = project(v_cv, fx, fy, cx, cy)

            valid = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0)
            step = max(1, valid.sum() // 2000)
            uv_s = uv[valid][::step]
            ax.scatter(uv_s[:, 0], uv_s[:, 1], s=0.3, c=[colors[i]], alpha=0.3, label=name, rasterized=True)

        ax.set_title(title)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        if panel_idx == 1:
            ax.legend(fontsize=6, loc="upper right", markerscale=10)

    plt.tight_layout()
    fig.savefig(str(output_dir / "full_scene_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def make_dashboard(objects, stats_before_all, stats_after_all, output_dir):
    """Before/after bar chart dashboard."""
    names = list(stats_before_all.keys())
    err_b = [stats_before_all[n].get("depth_error_rel_mean", 0) * 100 for n in names]
    err_a = [stats_after_all[n].get("depth_error_rel_mean", 0) * 100 for n in names]

    # Sort by before error
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
    ax.set_title("Depth Error: Before vs After Re-optimization", fontweight="bold")
    ax.legend()
    ax.axvline(5, color="green", ls="--", lw=0.8, alpha=0.4, label="5%")
    ax.axvline(15, color="orange", ls="--", lw=0.8, alpha=0.4, label="15%")

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
    parser = argparse.ArgumentParser(description="Re-optimize SAM3D poses with depth loss")
    parser.add_argument("--data-dir", type=Path, default=Path("output/sam3d_dining"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <data-dir>_v2)")
    parser.add_argument("--lambda-depth", type=float, default=1.0)
    parser.add_argument("--lambda-mask", type=float, default=0.5)
    parser.add_argument("--lambda-reg", type=float, default=0.01)
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

    # Copy MoGe npz to output dir for reference
    shutil.copy2(moge_path, output_dir / "target_moge.npz")

    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Objects: {len(objects)}")
    print(f"Image: {W}x{H}, fx={fx:.2f}")
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

        # Centroid in PT3D space
        centroid = np.array(obj["translation"], dtype=np.float32)

        # --- Before stats (mask-filtered) ---
        stats_b = compute_masked_depth_error(verts, mask_bool, depth, fx, fy, cx, cy, H, W)
        stats_before_all[name] = stats_b
        verts_before_all[name] = verts.copy()

        print(f"  BEFORE: rel_err={stats_b.get('depth_error_rel_mean', 0):.2%}  "
              f"|err|={stats_b.get('depth_error_abs_mean', 0):.4f}  "
              f"ratio={stats_b.get('scale_ratio', 0):.4f}  "
              f"valid={stats_b.get('valid_count', 0)}")

        # --- Optimize ---
        opt_result = optimize_object_pose(
            verts, mask, depth, fx, fy, cx, cy, H, W, centroid,
            glb_path=glb_path,
            lambda_depth=args.lambda_depth,
            lambda_mask=args.lambda_mask,
            lambda_reg=args.lambda_reg,
        )

        dt = np.array(opt_result["dt"], dtype=np.float32)
        ds = opt_result["ds"]
        print(f"  Optimization: dt=[{dt[0]:+.4f}, {dt[1]:+.4f}, {dt[2]:+.4f}]  "
              f"ds={ds:+.4f}  iters={opt_result['niter']}  loss={opt_result['loss']:.6f}")

        # --- Apply correction ---
        verts_corrected = apply_correction(verts, centroid, dt, ds)
        verts_after_all[name] = verts_corrected.copy()

        # --- After stats (mask-filtered) ---
        stats_a = compute_masked_depth_error(verts_corrected, mask_bool, depth, fx, fy, cx, cy, H, W)
        stats_after_all[name] = stats_a

        improvement = (stats_b.get("depth_error_rel_mean", 0) - stats_a.get("depth_error_rel_mean", 0))
        symbol = "+" if improvement > 0 else ""

        print(f"  AFTER:  rel_err={stats_a.get('depth_error_rel_mean', 0):.2%}  "
              f"|err|={stats_a.get('depth_error_abs_mean', 0):.4f}  "
              f"ratio={stats_a.get('scale_ratio', 0):.4f}  "
              f"({symbol}{improvement:.2%} improvement)")

        # --- Save corrected GLB ---
        scene = load_glb_scene(glb_path)
        for geom in scene.geometry.values():
            if hasattr(geom, "vertices"):
                v = np.asarray(geom.vertices, dtype=np.float32)
                geom.vertices = apply_correction(v, centroid, dt, ds)
        out_glb = output_dir / f"{name}.glb"
        scene.export(str(out_glb))

        # Copy mask and PNG
        if mask_path.exists():
            shutil.copy2(mask_path, output_dir / f"{name}.npy")
        if png_path.exists():
            shutil.copy2(png_path, output_dir / f"{name}.png")

        # --- Save info JSON ---
        new_centroid = centroid + dt
        new_transform = {
            "glb_path": str(out_glb),
            "translation": new_centroid.tolist(),
            "rotation": obj["rotation"],
            "scale": [float(s * (1.0 + ds)) for s in obj["scale"]],
            "pointmap_shape": obj.get("pointmap_shape", [H, W, 3]),
            "object_name": name,
        }
        new_transforms.append(new_transform)

        info = {**new_transform, "correction": {"dt": dt.tolist(), "ds": ds}}
        with open(output_dir / f"{name}_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        # --- Per-object comparison image ---
        compare_path = output_dir / f"{name}_compare.png"
        make_comparison_image(
            png_path if png_path.exists() else None,
            mask, verts, verts_corrected, depth, fx, fy, cx, cy, H, W,
            name, stats_b, stats_a, compare_path,
        )
        print(f"  Saved: {compare_path.name}")

        # --- Per-object render (after only) ---
        render_img = render_projected_vertices(verts_corrected, fx, fy, cx, cy, H, W, depth)
        if Image:
            Image.fromarray(render_img).save(str(output_dir / f"{name}_render.png"))

        print()

    # --- Save new transforms ---
    with open(output_dir / "object_transforms.json", "w", encoding="utf-8") as f:
        json.dump(new_transforms, f, indent=2)

    # --- Save results JSON ---
    results = {
        "before": stats_before_all,
        "after": stats_after_all,
    }
    with open(output_dir / "depth_alignment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # --- Full scene comparison ---
    print("Generating full scene comparison...")
    make_full_scene_comparison(moge, objects, data_dir, output_dir, verts_before_all, verts_after_all)

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
