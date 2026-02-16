#!/usr/bin/env python
"""Depth Alignment Diagnostic — Tests 1, 2, 4, 5.

Compares MoGe estimated depth against SAM3D-reconstructed GLB vertex depths
to quantify alignment errors and diagnose root causes.

Usage:
    python diagnose_depth_alignment.py [--data-dir output/sam3d_dining]

Requires: numpy, trimesh, Pillow  (the `agent` conda env)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError:
    sys.exit("trimesh not installed.  pip install trimesh")

try:
    from PIL import Image
except ImportError:
    Image = None  # optional, only for overlay vis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_moge(npz_path: Path) -> dict:
    """Load MoGe depth + intrinsics from target_moge.npz."""
    data = np.load(npz_path)
    K = data["intrinsics_px"].astype(np.float64)
    depth = data["depth"].astype(np.float32)
    w = int(data["image_width"])
    h = int(data["image_height"])
    return {
        "K": K,
        "fx": K[0, 0],
        "fy": K[1, 1],
        "cx": K[0, 2],
        "cy": K[1, 2],
        "depth": depth,
        "width": w,
        "height": h,
        "points": data["points"].astype(np.float32),
    }


def load_glb_vertices(glb_path: Path) -> np.ndarray:
    """Load GLB and return all vertices as (N, 3) float32.

    GLB vertices are already in PyTorch3D camera space (transforms baked in
    during SAM3D export).
    """
    scene = trimesh.load(str(glb_path), force="scene")
    all_verts = []
    for geom in scene.geometry.values():
        if hasattr(geom, "vertices"):
            all_verts.append(np.asarray(geom.vertices, dtype=np.float32))
    if not all_verts:
        raise ValueError(f"No mesh geometry found in {glb_path}")
    return np.concatenate(all_verts, axis=0)


def pt3d_to_opencv(verts: np.ndarray) -> np.ndarray:
    """Convert PyTorch3D camera coords to OpenCV camera coords.

    PyTorch3D: X-left, Y-up, Z-forward
    OpenCV:    X-right, Y-down, Z-forward
    """
    out = verts.copy()
    out[:, 0] = -verts[:, 0]
    out[:, 1] = -verts[:, 1]
    # Z stays the same
    return out


def project_to_pixels(
    verts_cv: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """Project OpenCV-space 3D points to pixel coords.  Returns (N, 2) [u, v]."""
    z = verts_cv[:, 2]
    u = fx * verts_cv[:, 0] / z + cx
    v = fy * verts_cv[:, 1] / z + cy
    return np.stack([u, v], axis=-1)


def sample_depth_bilinear(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample depth map at sub-pixel locations with bilinear interpolation.

    depth: (H, W)  uv: (N, 2) with u=col, v=row.
    Returns (N,) sampled depths.  Out-of-bounds pixels get NaN.
    """
    H, W = depth.shape
    u = uv[:, 0]
    v = uv[:, 1]

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    valid = (u0 >= 0) & (v0 >= 0) & (u1 < W) & (v1 < H)

    result = np.full(len(u), np.nan, dtype=np.float32)

    wu = (u - u0)[valid]
    wv = (v - v0)[valid]

    d00 = depth[v0[valid], u0[valid]]
    d01 = depth[v0[valid], u1[valid]]
    d10 = depth[v1[valid], u0[valid]]
    d11 = depth[v1[valid], u1[valid]]

    result[valid] = (
        d00 * (1 - wu) * (1 - wv)
        + d01 * wu * (1 - wv)
        + d10 * (1 - wu) * wv
        + d11 * wu * wv
    )
    return result


def load_mask(mask_path: Path) -> np.ndarray:
    """Load binary mask (H, W) as bool."""
    mask = np.load(str(mask_path))
    return mask > 127  # uint8 0/255 -> bool


# ---------------------------------------------------------------------------
# Test 1: Square Pixel Forcing
# ---------------------------------------------------------------------------

def test_square_pixel(moge: dict, objects: list[dict], data_dir: Path) -> dict:
    """Report fx-fy delta and measure pixel displacement from forcing fx=fy."""
    fx, fy, cx, cy = moge["fx"], moge["fy"], moge["cx"], moge["cy"]
    delta = fx - fy
    forced = min(fx, fy)

    print("=" * 70)
    print("TEST 1: Square Pixel Forcing")
    print("=" * 70)
    print(f"  fx       = {fx:.6f}")
    print(f"  fy       = {fy:.6f}")
    print(f"  fx - fy  = {delta:.6f}")
    print(f"  min(f)   = {forced:.6f}")

    results = {"fx": fx, "fy": fy, "delta": delta, "forced_f": forced, "objects": {}}

    if abs(delta) < 1e-4:
        print("  --> fx == fy: no square-pixel issue for this data.")
        results["issue"] = False
        return results

    results["issue"] = True
    for obj in objects:
        name = obj["object_name"]
        glb_path = data_dir / f"{name}.glb"
        if not glb_path.exists():
            continue

        verts = load_glb_vertices(glb_path)
        verts_cv = pt3d_to_opencv(verts)

        uv_orig = project_to_pixels(verts_cv, fx, fy, cx, cy)
        uv_forced = project_to_pixels(verts_cv, forced, forced, cx, cy)

        disp = np.linalg.norm(uv_orig - uv_forced, axis=-1)
        results["objects"][name] = {
            "mean_px_disp": float(np.nanmean(disp)),
            "max_px_disp": float(np.nanmax(disp)),
        }
        print(f"  {name:35s}  mean={np.nanmean(disp):.2f}px  max={np.nanmax(disp):.2f}px")

    print()
    return results


# ---------------------------------------------------------------------------
# Test 2: MoGe Depth vs GLB Vertex Depth
# ---------------------------------------------------------------------------

def test_depth_vs_vertices(moge: dict, objects: list[dict], data_dir: Path) -> dict:
    """Compare projected GLB vertex Z vs MoGe depth at the same pixels."""
    fx, fy, cx, cy = moge["fx"], moge["fy"], moge["cx"], moge["cy"]
    depth = moge["depth"]

    print("=" * 70)
    print("TEST 2: MoGe Depth vs GLB Vertex Depth")
    print("=" * 70)

    results = {}
    for obj in objects:
        name = obj["object_name"]
        glb_path = data_dir / f"{name}.glb"
        if not glb_path.exists():
            continue

        verts = load_glb_vertices(glb_path)
        verts_cv = pt3d_to_opencv(verts)

        uv = project_to_pixels(verts_cv, fx, fy, cx, cy)
        z_vertex = verts_cv[:, 2]  # depth from GLB

        # Sample MoGe depth at projected pixel locations
        z_moge = sample_depth_bilinear(depth, uv)

        valid = ~np.isnan(z_moge) & (z_vertex > 0)
        if valid.sum() == 0:
            print(f"  {name:35s}  NO valid projected vertices in image bounds")
            results[name] = {"valid_count": 0}
            continue

        z_v = z_vertex[valid]
        z_m = z_moge[valid]
        err = z_v - z_m  # positive = GLB deeper than MoGe
        abs_err = np.abs(err)
        rel_err = abs_err / np.maximum(z_m, 1e-6)

        # Subsample for percentile stats (too many verts is slow)
        n = len(z_v)
        stats = {
            "valid_count": int(valid.sum()),
            "total_verts": int(len(verts)),
            "z_vertex_mean": float(np.mean(z_v)),
            "z_vertex_median": float(np.median(z_v)),
            "z_moge_mean": float(np.mean(z_m)),
            "z_moge_median": float(np.median(z_m)),
            "depth_error_mean": float(np.mean(err)),
            "depth_error_median": float(np.median(err)),
            "depth_error_abs_mean": float(np.mean(abs_err)),
            "depth_error_abs_max": float(np.max(abs_err)),
            "depth_error_rel_mean": float(np.mean(rel_err)),
            "depth_error_rel_median": float(np.median(rel_err)),
            "scale_ratio": float(np.median(z_v) / np.median(z_m)),
        }
        results[name] = stats

        print(
            f"  {name:35s}  "
            f"vZ={stats['z_vertex_median']:.3f}  "
            f"mZ={stats['z_moge_median']:.3f}  "
            f"err={stats['depth_error_median']:+.4f}  "
            f"|err|={stats['depth_error_abs_mean']:.4f}  "
            f"rel={stats['depth_error_rel_mean']:.2%}  "
            f"ratio={stats['scale_ratio']:.4f}"
        )

    print()
    return results


# ---------------------------------------------------------------------------
# Test 4: Per-Object Depth Consistency
# ---------------------------------------------------------------------------

def test_depth_consistency(moge: dict, objects: list[dict], data_dir: Path) -> dict:
    """Check if all objects use consistent depth scale.

    Compare each object's translation Z (centroid depth) vs MoGe depth at
    the object's projected centroid.
    """
    fx, fy, cx, cy = moge["fx"], moge["fy"], moge["cx"], moge["cy"]
    depth = moge["depth"]

    print("=" * 70)
    print("TEST 4: Per-Object Depth Consistency")
    print("=" * 70)

    records = []
    for obj in objects:
        name = obj["object_name"]
        T = np.array(obj["translation"], dtype=np.float64)  # PyTorch3D space

        # Convert centroid to OpenCV
        cx_pt3d, cy_pt3d, cz_pt3d = T
        x_cv = -cx_pt3d
        y_cv = -cy_pt3d
        z_cv = cz_pt3d

        # Project centroid to pixel
        u = fx * x_cv / z_cv + cx
        v = fy * y_cv / z_cv + cy

        u_int = int(round(u))
        v_int = int(round(v))

        if 0 <= u_int < moge["width"] and 0 <= v_int < moge["height"]:
            z_moge = float(depth[v_int, u_int])
        else:
            z_moge = np.nan

        offset = z_cv - z_moge
        ratio = z_cv / z_moge if not np.isnan(z_moge) and z_moge > 0 else np.nan

        records.append({
            "name": name,
            "trans_z": float(z_cv),
            "moge_z": z_moge,
            "offset": float(offset) if not np.isnan(z_moge) else None,
            "ratio": float(ratio) if not np.isnan(ratio) else None,
            "pixel_u": u_int,
            "pixel_v": v_int,
        })

    # Compute median ratio as "expected" scale
    ratios = [r["ratio"] for r in records if r["ratio"] is not None]
    median_ratio = float(np.median(ratios)) if ratios else 1.0

    print(f"  Median depth ratio (GLB_Z / MoGe_Z): {median_ratio:.4f}")
    print(f"  {'Object':35s}  {'Trans_Z':>8s}  {'MoGe_Z':>8s}  {'Offset':>8s}  {'Ratio':>8s}  {'Anomaly':>8s}")
    print("  " + "-" * 105)

    for r in records:
        anomaly = ""
        if r["ratio"] is not None:
            dev = abs(r["ratio"] - median_ratio) / median_ratio
            if dev > 0.10:
                anomaly = f"  ** {dev:.0%} off"
        print(
            f"  {r['name']:35s}  "
            f"{r['trans_z']:8.4f}  "
            f"{r['moge_z']:8.4f}  "
            f"{r['offset']:+8.4f}  " if r["offset"] is not None else f"  {'N/A':>8s}  ",
            end=""
        )
        if r["ratio"] is not None:
            print(f"{r['ratio']:8.4f}{anomaly}")
        else:
            print("     N/A")

    print()
    return {"median_ratio": median_ratio, "objects": records}


# ---------------------------------------------------------------------------
# Test 5: Silhouette vs Depth Trade-off
# ---------------------------------------------------------------------------

def test_silhouette_vs_depth(
    moge: dict, objects: list[dict], data_dir: Path, depth_results: dict
) -> dict:
    """For each object, compare 2D projection overlap (IOU) with depth error."""
    fx, fy, cx, cy = moge["fx"], moge["fy"], moge["cx"], moge["cy"]
    H, W = moge["height"], moge["width"]

    print("=" * 70)
    print("TEST 5: Silhouette (2D) vs Depth Alignment Trade-off")
    print("=" * 70)

    results = {}
    for obj in objects:
        name = obj["object_name"]
        mask_path = data_dir / f"{name}.npy"
        glb_path = data_dir / f"{name}.glb"
        if not mask_path.exists() or not glb_path.exists():
            continue

        gt_mask = load_mask(mask_path)

        verts = load_glb_vertices(glb_path)
        verts_cv = pt3d_to_opencv(verts)
        uv = project_to_pixels(verts_cv, fx, fy, cx, cy)

        # Create projected silhouette mask from vertex scatter
        proj_mask = np.zeros((H, W), dtype=bool)
        u_int = np.round(uv[:, 0]).astype(np.int32)
        v_int = np.round(uv[:, 1]).astype(np.int32)
        valid = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
        proj_mask[v_int[valid], u_int[valid]] = True

        # Dilate projected mask slightly (vertices are sparse, not filled)
        # Use a simple box filter as dilation proxy
        from scipy.ndimage import binary_dilation
        proj_mask_dilated = binary_dilation(proj_mask, iterations=3)

        intersection = np.sum(gt_mask & proj_mask_dilated)
        union = np.sum(gt_mask | proj_mask_dilated)
        iou = float(intersection / max(union, 1))

        # Coverage: what fraction of GT mask pixels have at least one projected vertex nearby
        gt_covered = np.sum(gt_mask & proj_mask_dilated)
        coverage = float(gt_covered / max(np.sum(gt_mask), 1))

        # Get depth error from test 2
        depth_err = depth_results.get(name, {}).get("depth_error_abs_mean", None)
        depth_rel = depth_results.get(name, {}).get("depth_error_rel_mean", None)

        results[name] = {
            "iou": iou,
            "coverage": coverage,
            "depth_error_abs": depth_err,
            "depth_error_rel": depth_rel,
        }

        depth_str = f"|err|={depth_err:.4f}  rel={depth_rel:.2%}" if depth_err is not None else "N/A"
        print(f"  {name:35s}  IOU={iou:.3f}  cov={coverage:.3f}  {depth_str}")

    print()
    return results


# ---------------------------------------------------------------------------
# Test 3 (partial): Compare rendered depth if available
# ---------------------------------------------------------------------------

def test_rendered_depth(moge: dict, rendered_depth_path: Path | None, data_dir: Path) -> dict | None:
    """If a rendered depth .npy exists (from render_depth_pass.py), compare it."""
    if rendered_depth_path is None or not rendered_depth_path.exists():
        print("=" * 70)
        print("TEST 3: Rendered Depth Comparison — SKIPPED (no rendered depth file)")
        print("  Run render_depth_pass.py first, then re-run with --rendered-depth.")
        print("=" * 70)
        print()
        return None

    print("=" * 70)
    print("TEST 3: Rendered Depth vs MoGe Depth (pixel-by-pixel)")
    print("=" * 70)

    rendered = np.load(str(rendered_depth_path)).astype(np.float32)
    moge_depth = moge["depth"]

    if rendered.shape != moge_depth.shape:
        print(f"  Shape mismatch: rendered={rendered.shape} vs moge={moge_depth.shape}")
        print("  Cannot compare. Ensure render resolution matches MoGe image size.")
        return {"error": "shape_mismatch"}

    # Mask where rendered depth is valid (not background/inf/0)
    valid = (rendered > 0) & (rendered < 1e4) & (~np.isinf(rendered))
    n_valid = int(valid.sum())

    if n_valid == 0:
        print("  No valid rendered depth pixels found.")
        return {"error": "no_valid_pixels"}

    r = rendered[valid]
    m = moge_depth[valid]
    err = r - m
    abs_err = np.abs(err)
    rel_err = abs_err / np.maximum(m, 1e-6)

    results = {
        "valid_pixels": n_valid,
        "total_pixels": int(np.prod(rendered.shape)),
        "coverage_pct": float(n_valid / np.prod(rendered.shape) * 100),
        "mean_error": float(np.mean(err)),
        "median_error": float(np.median(err)),
        "abs_mean": float(np.mean(abs_err)),
        "abs_max": float(np.max(abs_err)),
        "rel_mean": float(np.mean(rel_err)),
        "rel_median": float(np.median(rel_err)),
        "scale_ratio": float(np.median(r) / np.median(m)),
    }

    print(f"  Valid pixels : {n_valid:,} / {np.prod(rendered.shape):,} ({results['coverage_pct']:.1f}%)")
    print(f"  Mean error   : {results['mean_error']:+.4f}")
    print(f"  Median error : {results['median_error']:+.4f}")
    print(f"  |err| mean   : {results['abs_mean']:.4f}")
    print(f"  |err| max    : {results['abs_max']:.4f}")
    print(f"  Rel mean     : {results['rel_mean']:.2%}")
    print(f"  Scale ratio  : {results['scale_ratio']:.4f}")
    print()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Depth Alignment Diagnostic")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("output/sam3d_dining"),
        help="Directory with target_moge.npz, GLBs, masks, transforms",
    )
    parser.add_argument(
        "--rendered-depth",
        type=Path,
        default=None,
        help="Path to rendered depth .npy from render_depth_pass.py (optional)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to save summary JSON (default: <data-dir>/depth_alignment_results.json)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    moge_path = data_dir / "target_moge.npz"
    transforms_path = data_dir / "object_transforms.json"

    if not moge_path.exists():
        sys.exit(f"MoGe file not found: {moge_path}")
    if not transforms_path.exists():
        sys.exit(f"Transforms file not found: {transforms_path}")

    # Load data
    print(f"Data directory: {data_dir}")
    print()

    moge = load_moge(moge_path)
    print(f"MoGe image: {moge['width']}x{moge['height']}")
    print(f"MoGe depth range: [{moge['depth'].min():.4f}, {moge['depth'].max():.4f}]")
    print(f"Camera: fx={moge['fx']:.4f}  fy={moge['fy']:.4f}  cx={moge['cx']:.4f}  cy={moge['cy']:.4f}")
    print()

    with open(transforms_path, encoding="utf-8") as f:
        objects = json.load(f)
    print(f"Objects: {len(objects)}")
    for obj in objects:
        print(f"  - {obj['object_name']}")
    print()

    # Run tests
    all_results = {}

    all_results["test1_square_pixel"] = test_square_pixel(moge, objects, data_dir)
    all_results["test2_depth_vs_vertices"] = test_depth_vs_vertices(moge, objects, data_dir)
    all_results["test4_depth_consistency"] = test_depth_consistency(moge, objects, data_dir)

    try:
        all_results["test5_silhouette_vs_depth"] = test_silhouette_vs_depth(
            moge, objects, data_dir, all_results["test2_depth_vs_vertices"]
        )
    except ImportError:
        print("TEST 5: SKIPPED (scipy not available for dilation)")
        all_results["test5_silhouette_vs_depth"] = {"error": "scipy_missing"}

    all_results["test3_rendered_depth"] = test_rendered_depth(
        moge, args.rendered_depth, data_dir
    )

    # Save summary JSON
    output_json = args.output_json or (data_dir / "depth_alignment_results.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {output_json}")


if __name__ == "__main__":
    main()
