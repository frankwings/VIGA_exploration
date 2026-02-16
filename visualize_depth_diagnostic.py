#!/usr/bin/env python
"""Generate visual diagnostic figures for depth alignment analysis.

Produces:
  1. Per-object scatter: GLB vertex depth vs MoGe depth
  2. Depth error heatmap: projected vertex errors on the image plane
  3. Summary dashboard: silhouette IOU vs depth error bar chart

Usage:
    python visualize_depth_diagnostic.py [--data-dir output/sam3d_dining]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch


def load_data(data_dir: Path):
    moge = np.load(data_dir / "target_moge.npz")
    depth = moge["depth"].astype(np.float32)
    K = moge["intrinsics_px"]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    H, W = int(moge["image_height"]), int(moge["image_width"])

    with open(data_dir / "object_transforms.json", encoding="utf-8") as f:
        objects = json.load(f)

    return depth, fx, fy, cx, cy, H, W, objects


def get_object_verts_cv(glb_path: Path):
    """Load GLB, return vertices in OpenCV coords."""
    scene = trimesh.load(str(glb_path), force="scene")
    verts = np.concatenate(
        [np.asarray(g.vertices, dtype=np.float32) for g in scene.geometry.values()]
    )
    v_cv = verts.copy()
    v_cv[:, 0] = -verts[:, 0]
    v_cv[:, 1] = -verts[:, 1]
    return v_cv


# -----------------------------------------------------------------------
# Figure 1: Per-object depth scatter
# -----------------------------------------------------------------------
def fig_scatter(data_dir, depth, fx, fy, cx, cy, H, W, objects):
    fig, axes = plt.subplots(3, 3, figsize=(16, 18))
    fig.suptitle("Per-Object: GLB Vertex Depth vs MoGe Depth", fontsize=15, fontweight="bold", y=0.98)

    for idx, obj in enumerate(objects):
        ax = axes[idx // 3][idx % 3]
        name = obj["object_name"]
        v_cv = get_object_verts_cv(data_dir / f"{name}.glb")

        z_v = v_cv[:, 2]
        u = fx * v_cv[:, 0] / z_v + cx
        v = fy * v_cv[:, 1] / z_v + cy

        valid = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1) & (z_v > 0)
        u_int = np.clip(np.round(u[valid]).astype(int), 0, W - 1)
        v_int = np.clip(np.round(v[valid]).astype(int), 0, H - 1)
        z_m = depth[v_int, u_int]
        z_vv = z_v[valid]

        # Subsample for plotting
        step = max(1, len(z_vv) // 4000)
        zv_s, zm_s = z_vv[::step], z_m[::step]
        err_s = zv_s - zm_s

        sc = ax.scatter(zm_s, zv_s, c=err_s, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                        s=1.5, alpha=0.4, rasterized=True)
        ax.plot([0.4, 3.5], [0.4, 3.5], "k--", lw=0.8, alpha=0.4)

        ax.set_xlim(0.5, 3.2)
        ax.set_ylim(0.5, 3.2)
        ax.set_aspect("equal")
        ax.set_xlabel("MoGe depth", fontsize=8)
        ax.set_ylabel("GLB vertex depth", fontsize=8)
        ax.tick_params(labelsize=7)

        rel = float(np.mean(np.abs(z_vv - z_m) / np.maximum(z_m, 1e-6)))
        ratio = float(np.median(z_vv) / np.median(z_m))
        color = "green" if rel < 0.05 else ("orange" if rel < 0.15 else "red")
        ax.set_title(f"{name}\nrel_err={rel:.1%}  ratio={ratio:.3f}", fontsize=9, color=color)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=mcolors.Normalize(-0.5, 0.5))
    fig.colorbar(sm, cax=cbar_ax, label="Depth error (GLB − MoGe)")

    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    out = data_dir / "depth_diagnostic_scatter.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# -----------------------------------------------------------------------
# Figure 2: Depth error heatmaps on image plane
# -----------------------------------------------------------------------
def fig_heatmaps(data_dir, depth, fx, fy, cx, cy, H, W, objects):
    fig, axes = plt.subplots(3, 3, figsize=(16, 20))
    fig.suptitle("Depth Error Projected onto Image Plane", fontsize=15, fontweight="bold", y=0.98)

    for idx, obj in enumerate(objects):
        ax = axes[idx // 3][idx % 3]
        name = obj["object_name"]
        v_cv = get_object_verts_cv(data_dir / f"{name}.glb")

        z_v = v_cv[:, 2]
        u = fx * v_cv[:, 0] / z_v + cx
        v = fy * v_cv[:, 1] / z_v + cy

        valid = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1) & (z_v > 0)
        u_int = np.clip(np.round(u[valid]).astype(int), 0, W - 1)
        v_int = np.clip(np.round(v[valid]).astype(int), 0, H - 1)
        z_m = depth[v_int, u_int]
        z_vv = z_v[valid]
        err = z_vv - z_m

        # Build error image
        err_img = np.full((H, W), np.nan, dtype=np.float32)
        err_img[v_int, u_int] = err

        # Show MoGe depth as background (faint)
        ax.imshow(depth, cmap="gray", alpha=0.3, vmin=depth.min(), vmax=depth.max())
        im = ax.imshow(err_img, cmap="RdBu_r", vmin=-0.5, vmax=0.5, alpha=0.85)

        # Load and overlay mask outline
        mask_path = data_dir / f"{name}.npy"
        if mask_path.exists():
            mask = np.load(str(mask_path)) > 127
            ax.contour(mask, levels=[0.5], colors="lime", linewidths=0.5, alpha=0.6)

        rel = float(np.mean(np.abs(err)) / np.maximum(np.mean(z_m), 1e-6))
        color = "green" if rel < 0.05 else ("orange" if rel < 0.15 else "red")
        ax.set_title(f"{name}", fontsize=9, color=color)
        ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=mcolors.Normalize(-0.5, 0.5))
    fig.colorbar(sm, cax=cbar_ax, label="Depth error (GLB − MoGe)")

    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    out = data_dir / "depth_diagnostic_heatmap.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# -----------------------------------------------------------------------
# Figure 3: Summary dashboard
# -----------------------------------------------------------------------
def fig_dashboard(data_dir, depth, fx, fy, cx, cy, H, W, objects):
    results_path = data_dir / "depth_alignment_results.json"
    if not results_path.exists():
        print("Skipping dashboard — no results JSON found.")
        return

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    t2 = results["test2_depth_vs_vertices"]
    t5 = results.get("test5_silhouette_vs_depth", {})

    names = [o["object_name"] for o in objects]
    rel_errs = [t2.get(n, {}).get("depth_error_rel_mean", 0) * 100 for n in names]
    ious = [t5.get(n, {}).get("iou", 0) for n in names]
    coverages = [t5.get(n, {}).get("coverage", 0) for n in names]
    ratios = [t2.get(n, {}).get("scale_ratio", 1.0) for n in names]

    # Sort by rel error
    order = np.argsort(rel_errs)
    names_s = [names[i] for i in order]
    rel_s = [rel_errs[i] for i in order]
    iou_s = [ious[i] for i in order]
    cov_s = [coverages[i] for i in order]
    ratio_s = [ratios[i] for i in order]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Depth Alignment Summary Dashboard", fontsize=15, fontweight="bold")

    # Top-left: Rel depth error bar
    ax = axes[0, 0]
    colors = ["green" if r < 5 else ("orange" if r < 15 else "red") for r in rel_s]
    bars = ax.barh(range(len(names_s)), rel_s, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels(names_s, fontsize=8)
    ax.set_xlabel("Relative Depth Error (%)")
    ax.set_title("Depth Error (lower = better)")
    ax.axvline(5, color="green", ls="--", lw=0.8, alpha=0.5, label="5% threshold")
    ax.axvline(15, color="orange", ls="--", lw=0.8, alpha=0.5, label="15% threshold")
    ax.legend(fontsize=7)
    for i, v in enumerate(rel_s):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=7)

    # Top-right: IOU bar
    ax = axes[0, 1]
    colors_iou = ["green" if i > 0.6 else ("orange" if i > 0.3 else "red") for i in iou_s]
    ax.barh(range(len(names_s)), iou_s, color=colors_iou, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels(names_s, fontsize=8)
    ax.set_xlabel("2D Silhouette IOU")
    ax.set_title("2D Alignment (higher = better)")
    ax.set_xlim(0, 1)
    for i, v in enumerate(iou_s):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7)

    # Bottom-left: Scatter IOU vs depth error
    ax = axes[1, 0]
    for i, name in enumerate(names):
        c = "green" if rel_errs[i] < 5 else ("orange" if rel_errs[i] < 15 else "red")
        ax.scatter(ious[i], rel_errs[i], s=80, c=c, edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(name.replace("_", " "), (ious[i], rel_errs[i]),
                    fontsize=6, ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")
    ax.set_xlabel("2D Silhouette IOU")
    ax.set_ylabel("Relative Depth Error (%)")
    ax.set_title("2D Alignment vs Depth Accuracy")
    ax.axhline(5, color="green", ls="--", lw=0.8, alpha=0.4)
    ax.axhline(15, color="orange", ls="--", lw=0.8, alpha=0.4)
    ax.set_xlim(0, 1)

    # Bottom-right: Scale ratio bar
    ax = axes[1, 1]
    colors_r = ["green" if abs(r - 1) < 0.05 else ("orange" if abs(r - 1) < 0.15 else "red") for r in ratio_s]
    ax.barh(range(len(names_s)), [r - 1.0 for r in ratio_s], left=1.0,
            color=colors_r, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels(names_s, fontsize=8)
    ax.set_xlabel("Scale Ratio (GLB / MoGe median depth)")
    ax.set_title("Depth Scale Ratio (1.0 = perfect)")
    ax.axvline(1.0, color="black", lw=1)
    ax.axvline(0.95, color="green", ls="--", lw=0.8, alpha=0.4)
    ax.axvline(1.05, color="green", ls="--", lw=0.8, alpha=0.4)
    for i, v in enumerate(ratio_s):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)

    plt.tight_layout()
    out = data_dir / "depth_diagnostic_dashboard.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# -----------------------------------------------------------------------
# Figure 4: Side-by-side MoGe depth vs rendered depth
# -----------------------------------------------------------------------
def fig_depth_comparison(data_dir, depth, H, W):
    rendered_path = data_dir / "rendered_depth.npy"
    if not rendered_path.exists():
        print("Skipping depth comparison — no rendered_depth.npy found.")
        return

    rendered = np.load(str(rendered_path)).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("MoGe Depth vs Blender Rendered Depth", fontsize=14, fontweight="bold")

    vmin, vmax = depth.min(), depth.max()

    ax = axes[0]
    im = ax.imshow(depth, cmap="turbo", vmin=vmin, vmax=vmax)
    ax.set_title("MoGe Estimated Depth")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    # Mask out background (inf / very large values)
    r_vis = rendered.copy()
    r_vis[(r_vis > 50) | (r_vis < 0)] = np.nan
    im = ax.imshow(r_vis, cmap="turbo", vmin=vmin, vmax=vmax)
    ax.set_title("Blender Rendered Depth")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    valid = (rendered > 0) & (rendered < 50)
    diff = np.full_like(depth, np.nan)
    diff[valid] = rendered[valid] - depth[valid]
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_title("Rendered − MoGe (error)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = data_dir / "depth_diagnostic_comparison.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("output/sam3d_dining"))
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    depth, fx, fy, cx, cy, H, W, objects = load_data(data_dir)

    print("Generating visualizations...")
    fig_scatter(data_dir, depth, fx, fy, cx, cy, H, W, objects)
    fig_heatmaps(data_dir, depth, fx, fy, cx, cy, H, W, objects)
    fig_dashboard(data_dir, depth, fx, fy, cx, cy, H, W, objects)
    fig_depth_comparison(data_dir, depth, H, W)
    print("Done.")


if __name__ == "__main__":
    main()
