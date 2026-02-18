"""Visualize convex hull mask growth for greentea scene objects.

Generates 4-panel diagnostic for each object:
  - Normal deviation angle map (blue=coplanar, red=different surface)
  - Original SAM mask + convex hull outline
  - Grown mask (original green, added orange)
  - Normal angle deviation histogram with threshold line

Uses local surface normal consistency instead of Sobel depth edges:
  - Compute per-pixel normals from 3D pointmap via central differences
  - Reference normal = mean of normals inside original mask
  - Allow growth where normal angle deviation < adaptive threshold (median + k*sigma)

Usage:
    python visualize_convex_hull_growth.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, gaussian_filter
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

MASKS_DIR  = PROJECT_ROOT / "output/test/greentea/sam_init"
MOGE_NPZ   = PROJECT_ROOT / "output/sam3d_rerun_fixed/target_moge.npz"
OUTPUT_DIR = PROJECT_ROOT / "output/sam3d_convex_hull_v3/vis"

# ---------------------------------------------------------------------------
# Convex hull helper
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Normal-based mask growth
# ---------------------------------------------------------------------------

def _compute_local_normals(
    pointmap: np.ndarray,   # (H, W, 3)
    smooth_sigma: float = 1.5,
) -> np.ndarray:
    """Per-pixel surface normals via central differences on the 3D point map.

    Optionally Gaussian-smooths the pointmap first to reduce noise.
    Returns (H, W, 3) unit normals; zero vector at invalid pixels.
    """
    pm = gaussian_filter(pointmap.astype(np.float32), sigma=[smooth_sigma, smooth_sigma, 0])

    # Tangents: central differences
    dx = np.zeros_like(pm)
    dy = np.zeros_like(pm)
    dx[:, 1:-1] = pm[:, 2:] - pm[:, :-2]
    dy[1:-1, :] = pm[2:, :] - pm[:-2, :]

    # Normal = cross(dx, dy)
    normals = np.cross(dx, dy)          # (H, W, 3)

    # Normalize; zero out degenerate pixels
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = norms[..., 0] > 1e-10
    normals[valid] /= norms[valid]
    normals[~valid] = 0.0
    return normals.astype(np.float32)


def _grow_mask_normal_consistency(
    mask_bool: np.ndarray,    # (H, W) bool
    pointmap: np.ndarray,     # (H, W, 3) float  — MoGe 3D points
    k_sigma: float = 2.0,
    min_threshold_deg: float = 10.0,
    max_threshold_deg: float = 60.0,
    smooth_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Grow mask toward convex hull using surface normal consistency.

    Algorithm:
    1. Compute per-pixel surface normals from the 3D pointmap (central diffs).
    2. Reference normal = mean of valid normals inside the original mask.
    3. Angle deviation map = arccos(|normals · ref|) for every pixel.
    4. Adaptive threshold = median + k_sigma * std of angles inside mask,
       clamped to [min_threshold_deg, max_threshold_deg].
    5. All hull pixels with angle < threshold AND valid normal join the mask
       (direct assignment — no connectivity requirement).

    max_threshold_deg=60 handles curved surfaces (bottles etc.) while still
    stopping at genuine surface boundaries (table, background walls ≥ 70°
    from object normals).

    Returns
    -------
    grown_mask    : (H, W) bool
    hull_mask     : (H, W) bool
    threshold_rad : adaptive threshold in radians
    normals       : (H, W, 3) computed unit normals
    angle_map     : (H, W) per-pixel angle deviation from reference (radians)
    """
    hull_mask = _make_convex_hull_mask(mask_bool)

    # --- Compute normals ---
    normals = _compute_local_normals(pointmap, smooth_sigma=smooth_sigma)

    # --- Reference normal: mean of valid normals inside original mask ---
    normal_valid_map = np.linalg.norm(normals, axis=-1) > 0.5   # (H, W)
    normals_in_mask = normals[mask_bool & normal_valid_map]

    if len(normals_in_mask) < 10:
        # Fallback: no growth
        angle_map = np.zeros(mask_bool.shape, dtype=np.float32)
        return mask_bool.copy(), hull_mask, 0.0, normals, angle_map

    ref_normal = normals_in_mask.mean(axis=0)
    ref_norm = np.linalg.norm(ref_normal)
    if ref_norm < 1e-8:
        angle_map = np.zeros(mask_bool.shape, dtype=np.float32)
        return mask_bool.copy(), hull_mask, 0.0, normals, angle_map
    ref_normal /= ref_norm

    # --- Angle deviation map ---
    # Use |dot| to handle sign ambiguity (normals can point inward or outward)
    dot = np.clip(np.abs(normals @ ref_normal), 0.0, 1.0)   # (H, W)
    angle_map = np.arccos(dot).astype(np.float32)            # (H, W) radians

    # --- Adaptive threshold from angles inside original mask ---
    angles_in_mask = angle_map[mask_bool & normal_valid_map]
    threshold_rad = float(np.median(angles_in_mask)) + k_sigma * float(np.std(angles_in_mask))
    threshold_rad = float(np.clip(
        threshold_rad,
        np.radians(min_threshold_deg),
        np.radians(max_threshold_deg),
    ))

    # --- Safe growth pixels: hull region, consistent normal, valid normal ---
    growth_region = hull_mask & ~mask_bool
    safe_pixels = growth_region & (angle_map < threshold_rad) & normal_valid_map

    # --- Direct assignment (no connectivity constraint) ---
    # Iterative connected dilation fails when depth-discontinuity pixels at the
    # mask border form a high-deviation "ring", blocking access to safe interior
    # pixels.  Direct assignment avoids this: any hull pixel with consistent
    # normals joins the mask regardless of connectivity.
    grown = mask_bool | safe_pixels

    return grown, hull_mask, threshold_rad, normals, angle_map

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_mask_growth(
    mask_bool: np.ndarray,
    grown_mask: np.ndarray,
    hull_mask: np.ndarray,
    depth: np.ndarray,
    angle_map: np.ndarray,     # (H, W) angle deviation in radians
    threshold_rad: float,
    obj_name: str,
    output_path: Path,
):
    """4-panel: normal deviation map | original+hull | grown mask | angle histogram."""
    H, W = mask_bool.shape
    threshold_deg = np.degrees(threshold_rad)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Mask Growth (normal consistency): {obj_name}",
                 fontsize=14, fontweight="bold")

    # --- Panel 1: Normal angle deviation map ---
    ax = axes[0, 0]
    # Background: depth map (grayscale)
    ax.imshow(depth, cmap="gray", vmin=depth.min(), vmax=depth.max())
    # Overlay: angle deviation colormap (masked to hull region + original mask)
    show_region = hull_mask | mask_bool
    angle_display = np.full((H, W), np.nan, dtype=np.float32)
    angle_display[show_region] = np.degrees(angle_map[show_region])
    im = ax.imshow(angle_display, cmap="RdYlGn_r", vmin=0, vmax=60, alpha=0.75)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Normal deviation (°)")
    # Red contour where angle >= threshold
    above = show_region & (angle_map >= threshold_rad)
    above_overlay = np.zeros((H, W, 4), dtype=np.float32)
    above_overlay[above] = [1, 0, 0, 0.45]
    ax.imshow(above_overlay)
    # Mask outline
    mask_edge = mask_bool & ~binary_erosion(mask_bool, iterations=1)
    edge_overlay = np.zeros((H, W, 4), dtype=np.float32)
    edge_overlay[mask_edge] = [0, 1, 0, 1.0]
    ax.imshow(edge_overlay)
    ax.set_title(f"Normal deviation (green=mask outline, red=>{threshold_deg:.1f}°)")
    ax.axis("off")

    # --- Panel 2: Original mask + convex hull ---
    ax = axes[0, 1]
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[hull_mask] = [60, 60, 100]
    vis[mask_bool] = [0, 200, 0]
    hull_edge = hull_mask & ~binary_erosion(hull_mask, iterations=1)
    vis[hull_edge] = [255, 255, 0]
    ax.imshow(vis)
    n_orig = int(mask_bool.sum())
    n_hull = int(hull_mask.sum())
    ax.set_title(f"Original mask ({n_orig:,}px) + Convex hull ({n_hull:,}px)")
    ax.axis("off")

    # --- Panel 3: Grown mask ---
    ax = axes[1, 0]
    vis2 = np.zeros((H, W, 3), dtype=np.uint8)
    vis2[grown_mask] = [0, 150, 200]
    vis2[mask_bool] = [0, 200, 0]
    added = grown_mask & ~mask_bool
    vis2[added] = [255, 165, 0]
    ax.imshow(vis2)
    n_grown = int(grown_mask.sum())
    n_added = int(added.sum())
    ax.set_title(f"Grown mask ({n_grown:,}px, +{n_added:,} added)")
    ax.axis("off")

    # --- Panel 4: Angle deviation histogram ---
    ax = axes[1, 1]
    normal_valid_map = np.linalg.norm  # placeholder — recompute from angle_map shape
    # Use angle_map directly; filter by pixels where we have valid normals (angle != 0 or mask px)
    angles_in_mask   = np.degrees(angle_map[mask_bool])
    growth_region    = hull_mask & ~mask_bool
    angles_in_growth = np.degrees(angle_map[growth_region])
    if len(angles_in_mask) > 0:
        ax.hist(angles_in_mask, bins=90, range=(0, 90), alpha=0.7, color="green",
                label="Inside mask", density=True)
    if len(angles_in_growth) > 0:
        ax.hist(angles_in_growth, bins=90, range=(0, 90), alpha=0.5, color="orange",
                label="Growth region", density=True)
    ax.axvline(threshold_deg, color="red", ls="--", lw=2,
               label=f"Threshold={threshold_deg:.1f}°")
    ax.set_xlabel("Normal deviation angle (°)")
    ax.set_ylabel("Density")
    ax.set_title("Normal angle distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(threshold_deg * 4, 90))

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading MoGe data: {MOGE_NPZ}")
    moge = np.load(str(MOGE_NPZ))
    print(f"  Keys: {list(moge.keys())}")

    pointmap = moge["points"].astype(np.float32)   # (H, W, 3) — 3D positions
    depth    = moge["depth"].astype(np.float32)    # (H, W)    — metric depth for display
    print(f"  Pointmap shape={pointmap.shape}, depth range=[{depth.min():.3f}, {depth.max():.3f}]")

    for name in OBJECTS:
        mask_path = MASKS_DIR / f"{name}.npy"
        if not mask_path.exists():
            print(f"\n  SKIP {name} — mask not found")
            continue

        print(f"\nProcessing: {name}")
        mask_bool = np.load(str(mask_path)).astype(bool)

        if mask_bool.shape != depth.shape:
            print(f"  WARNING: mask shape {mask_bool.shape} != pointmap {depth.shape}")
            continue

        n_px  = int(mask_bool.sum())
        n_tot = mask_bool.size
        print(f"  Mask: {n_px:,} / {n_tot:,} px ({100.*n_px/n_tot:.1f}%)")

        grown_mask, hull_mask, threshold_rad, normals, angle_map = \
            _grow_mask_normal_consistency(mask_bool, pointmap)

        n_added = int((grown_mask & ~mask_bool).sum())
        print(f"  Hull: {hull_mask.sum():,}px | Grown: +{n_added:,}px "
              f"| threshold={np.degrees(threshold_rad):.1f}°")

        out_path = OUTPUT_DIR / f"{name}_mask_growth.png"
        visualize_mask_growth(
            mask_bool, grown_mask, hull_mask, depth,
            angle_map, threshold_rad, name, out_path,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
