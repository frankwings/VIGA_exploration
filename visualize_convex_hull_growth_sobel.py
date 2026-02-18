"""Visualize convex hull mask growth for greentea scene objects.

Generates the same 4-panel format as sam3d_dining_v4/*_mask_growth.png:
  - Depth + Sobel edges (with threshold overlay)
  - Original SAM mask + convex hull outline
  - Grown mask (original green, added orange)
  - Sobel gradient histogram with threshold line

Usage:
    python visualize_convex_hull_growth.py
"""

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion, sobel
from scipy.spatial import ConvexHull

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OBJECTS = [
    "ito_en_bottle",
    "green_tea_bottle",
    "alienware_keyboard",
    "headphones",
    "envelope",
]

MASKS_DIR   = PROJECT_ROOT / "output/test/greentea/sam_init"
MOGE_NPZ    = PROJECT_ROOT / "output/sam3d_rerun_fixed/target_moge.npz"
OUTPUT_DIR  = PROJECT_ROOT / "output/sam3d_convex_hull_v2/vis"

# ---------------------------------------------------------------------------
# Convex hull growth (mirrors layout_post_optimization_utils.py)
# ---------------------------------------------------------------------------

def _compute_depth_sobel(depth: np.ndarray) -> np.ndarray:
    sx = sobel(depth.astype(np.float64), axis=1)
    sy = sobel(depth.astype(np.float64), axis=0)
    return np.sqrt(sx**2 + sy**2).astype(np.float32)


def _make_convex_hull_mask(mask_bool: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask_bool)
    if len(rows) < 3:
        return mask_bool.copy()
    points_2d = np.column_stack([cols, rows])
    try:
        hull = ConvexHull(points_2d)
    except Exception:
        return mask_bool.copy()
    from matplotlib.path import Path as MplPath
    hull_pts = points_2d[hull.vertices]
    hull_path = MplPath(hull_pts)
    H, W = mask_bool.shape
    rmin, rmax = max(0, rows.min() - 2), min(H - 1, rows.max() + 2)
    cmin, cmax = max(0, cols.min() - 2), min(W - 1, cols.max() + 2)
    yy, xx = np.mgrid[rmin:rmax+1, cmin:cmax+1]
    test_pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = hull_path.contains_points(test_pts).reshape(yy.shape)
    hull_mask = np.zeros_like(mask_bool)
    hull_mask[rmin:rmax+1, cmin:cmax+1] = inside
    return hull_mask


def _grow_mask_to_convex_hull(
    mask_bool: np.ndarray,
    depth: np.ndarray,
    edge_sigma: float = 2.0,
    max_iters: int = 50,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (grown_mask, hull_mask, threshold)."""
    hull_mask = _make_convex_hull_mask(mask_bool)
    sobel_mag = _compute_depth_sobel(depth)
    sobel_in_mask = sobel_mag[mask_bool]
    if len(sobel_in_mask) == 0:
        return mask_bool.copy(), hull_mask, 0.0
    threshold = float(np.median(sobel_in_mask)) + edge_sigma * float(np.std(sobel_in_mask))
    growth_region = hull_mask & ~mask_bool
    safe_pixels = growth_region & (sobel_mag < threshold)
    struct = np.ones((3, 3), dtype=bool)
    grown = mask_bool.copy()
    for _ in range(max_iters):
        dilated = binary_dilation(grown, structure=struct)
        new_pixels = dilated & safe_pixels & ~grown
        if not new_pixels.any():
            break
        grown = grown | new_pixels
    return grown, hull_mask, threshold


# ---------------------------------------------------------------------------
# Visualization (matches sam3d_dining_v4 style exactly)
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
    H, W = mask_bool.shape
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Mask Growth: {obj_name}", fontsize=14, fontweight="bold")

    # Panel 1: depth + Sobel edges
    ax = axes[0, 0]
    ax.imshow(depth, cmap="turbo", vmin=depth.min(), vmax=depth.max())
    edge_overlay = np.zeros((H, W, 4))
    edge_overlay[sobel_mag >= threshold] = [1, 0, 0, 0.5]
    ax.imshow(edge_overlay)
    ax.set_title(f"Depth + Edges (threshold={threshold:.4f})")
    ax.axis("off")

    # Panel 2: original mask + hull
    ax = axes[0, 1]
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[hull_mask] = [60, 60, 100]
    vis[mask_bool] = [0, 200, 0]
    hull_edge = hull_mask & ~binary_erosion(hull_mask, iterations=1)
    vis[hull_edge] = [255, 255, 0]
    ax.imshow(vis)
    n_orig = mask_bool.sum()
    n_hull = hull_mask.sum()
    ax.set_title(f"Original mask ({n_orig:,}px) + Convex hull ({n_hull:,}px)")
    ax.axis("off")

    # Panel 3: grown mask
    ax = axes[1, 0]
    vis2 = np.zeros((H, W, 3), dtype=np.uint8)
    vis2[grown_mask] = [0, 150, 200]
    vis2[mask_bool] = [0, 200, 0]
    added = grown_mask & ~mask_bool
    vis2[added] = [255, 165, 0]
    ax.imshow(vis2)
    n_grown = grown_mask.sum()
    n_added = added.sum()
    ax.set_title(f"Grown mask ({n_grown:,}px, +{n_added:,} added)")
    ax.axis("off")

    # Panel 4: Sobel histogram
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
    ax.set_xlim(0, min(threshold * 5, float(sobel_mag.max())))

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load MoGe depth
    print(f"Loading depth: {MOGE_NPZ}")
    moge = np.load(str(MOGE_NPZ))
    print(f"  Keys: {list(moge.keys())}")
    # depth is stored directly as (H, W) metric depth
    depth = moge["depth"]         # (H, W) metric depth in metres
    print(f"  Depth shape={depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]")

    sobel_mag = _compute_depth_sobel(depth)

    for name in OBJECTS:
        mask_path = MASKS_DIR / f"{name}.npy"
        if not mask_path.exists():
            print(f"  SKIP {name} — mask not found: {mask_path}")
            continue

        print(f"\nProcessing: {name}")
        mask_raw = np.load(str(mask_path))
        # SAM masks saved as uint8 or bool; make bool
        mask_bool = mask_raw.astype(bool)
        if mask_bool.shape != depth.shape:
            print(f"  WARNING: mask shape {mask_bool.shape} != depth {depth.shape}")
            continue

        n_px = mask_bool.sum()
        n_total = mask_bool.size
        pct = 100.0 * n_px / n_total
        print(f"  Mask: {n_px:,} / {n_total:,} px ({pct:.1f}% visible)")

        grown_mask, hull_mask, threshold = _grow_mask_to_convex_hull(mask_bool, depth)
        n_added = (grown_mask & ~mask_bool).sum()
        print(f"  Hull: {hull_mask.sum():,}px | Grown: +{n_added:,}px | threshold={threshold:.4f}")

        out_path = OUTPUT_DIR / f"{name}_mask_growth.png"
        visualize_mask_growth(
            mask_bool, grown_mask, hull_mask, depth, sobel_mag,
            threshold, name, out_path,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
