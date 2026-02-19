"""Visualize convex hull mask growth — local plane-distance consistency.

Algorithm (v4):
  1. Morphological opening (2x erosion + 2x dilation) to clean mask boundary.
  2. Compute convex hull of cleaned mask.
  3. For each hull pixel, find nearest cleaned-mask pixel via EDT.
  4. Accept pixel if |dot(normal_n, pos_candidate - pos_n)| < threshold_m
     AND the neighbor has a valid normal.
  5. Direct assignment: all accepted hull pixels join the mask at once.

4-panel output per object:
  - Plane distance map (0=coplanar/green, red=far from surface)
  - Original mask + cleaned mask + hull outline
  - Grown mask (cleaned=green, added=orange, eroded=red)
  - Plane distance histogram for growth region

Usage:
    python visualize_convex_hull_growth.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import (
    binary_erosion, binary_dilation,
    distance_transform_edt, gaussian_filter,
)
from scipy.spatial import ConvexHull

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OBJECTS = [
    "chair_cushion",
    "chair_legs",
    "newspaper",
    "placemat",
    "round_table_with_tablecloth",
    "sofa_with_patterned_cover",
    "strainer",
    "travel_pillow",
    "wooden_chair",
]

MASKS_DIR   = PROJECT_ROOT / "output/sam3d_dining_v4"
MOGE_NPZ    = PROJECT_ROOT / "output/sam3d_dining_v4/target_moge.npz"
OUTPUT_DIR  = PROJECT_ROOT / "output/sam3d_dining_plane_dist/vis"

THRESHOLD_M  = 0.03   # plane-distance acceptance threshold (meters)
SMOOTH_SIGMA = 2.0    # Gaussian sigma for normal computation (~13×13 window)

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
    hull_pts  = points_2d[hull.vertices]
    hull_path = MplPath(hull_pts)
    H, W = mask_bool.shape
    rmin, rmax = max(0, rows.min() - 2), min(H - 1, rows.max() + 2)
    cmin, cmax = max(0, cols.min() - 2), min(W - 1, cols.max() + 2)
    yy, xx = np.mgrid[rmin:rmax+1, cmin:cmax+1]
    inside = hull_path.contains_points(
        np.column_stack([xx.ravel(), yy.ravel()])
    ).reshape(yy.shape)
    hull_mask = np.zeros_like(mask_bool)
    hull_mask[rmin:rmax+1, cmin:cmax+1] = inside
    return hull_mask

# ---------------------------------------------------------------------------
# Normal computation
# ---------------------------------------------------------------------------

def _compute_local_normals(
    pointmap: np.ndarray,    # (H, W, 3)
    smooth_sigma: float = SMOOTH_SIGMA,
) -> np.ndarray:
    """Per-pixel surface normals via central differences on Gaussian-smoothed pointmap.

    Returns (H, W, 3) unit normals; zero vector at degenerate pixels.
    """
    pm = gaussian_filter(pointmap.astype(np.float32),
                         sigma=[smooth_sigma, smooth_sigma, 0])
    dx = np.zeros_like(pm)
    dy = np.zeros_like(pm)
    dx[:, 1:-1] = pm[:, 2:] - pm[:, :-2]
    dy[1:-1, :] = pm[2:, :] - pm[:-2, :]
    normals = np.cross(dx, dy)
    norms   = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid   = norms[..., 0] > 1e-10
    normals[valid]  /= norms[valid]
    normals[~valid]  = 0.0
    return normals.astype(np.float32)

# ---------------------------------------------------------------------------
# Mask growth — local plane distance
# ---------------------------------------------------------------------------

def _grow_mask_plane_distance(
    mask_bool:    np.ndarray,   # (H, W) bool  — original SAM mask
    pointmap:     np.ndarray,   # (H, W, 3)    — MoGe 3D point map
    threshold_m:  float = THRESHOLD_M,
    smooth_sigma: float = SMOOTH_SIGMA,
    erosion_iters:  int = 2,
    dilation_iters: int = 2,
) -> tuple:
    """Grow mask toward convex hull using local surface plane consistency.

    Steps
    -----
    1. Morphological opening (erosion_iters + dilation_iters) to remove
       noisy boundary protrusions.
    2. Convex hull of the cleaned mask.
    3. EDT: for every pixel find its nearest cleaned-mask pixel (r_n, c_n).
    4. Plane distance = |dot(normal_n, pos_candidate - pos_n)|.
       Accept hull pixels where plane_dist < threshold_m AND normal_n valid.
    5. Direct assignment: grown = cleaned | accepted_hull_pixels.

    Returns
    -------
    grown_mask   : (H, W) bool
    hull_mask    : (H, W) bool  — convex hull of cleaned mask
    cleaned_mask : (H, W) bool  — morphologically-opened mask
    plane_dist   : (H, W) float — plane distance map (meters)
    threshold_m  : float
    """
    # 1. Morphological opening
    cleaned = mask_bool.copy()
    if mask_bool.sum() > 50:
        eroded = binary_erosion(mask_bool, iterations=erosion_iters)
        if eroded.sum() >= 10:
            cleaned = binary_dilation(eroded, iterations=dilation_iters)
        # else: mask too small to erode — keep original

    # 2. Convex hull of cleaned mask
    hull_mask = _make_convex_hull_mask(cleaned)

    # 3. Normals
    normals      = _compute_local_normals(pointmap, smooth_sigma=smooth_sigma)
    normal_valid = np.linalg.norm(normals, axis=-1) > 0.5   # (H, W)

    # 4. EDT — nearest cleaned-mask pixel for every pixel in the image
    _, nearest_idx = distance_transform_edt(~cleaned, return_indices=True)
    nn_r, nn_c = nearest_idx[0], nearest_idx[1]             # (H, W) each

    pos_neighbor   = pointmap[nn_r, nn_c]                   # (H, W, 3)
    norm_neighbor  = normals[nn_r, nn_c]                    # (H, W, 3)
    valid_neighbor = normal_valid[nn_r, nn_c]               # (H, W)

    delta      = pointmap - pos_neighbor                    # (H, W, 3)
    plane_dist = np.abs(np.sum(norm_neighbor * delta, axis=-1)).astype(np.float32)

    # 5. Accept & assign
    growth_region = hull_mask & ~cleaned
    safe  = growth_region & (plane_dist < threshold_m) & valid_neighbor
    grown = cleaned | safe

    return grown, hull_mask, cleaned, plane_dist, threshold_m

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_mask_growth(
    mask_bool:    np.ndarray,
    grown_mask:   np.ndarray,
    hull_mask:    np.ndarray,
    cleaned_mask: np.ndarray,
    depth:        np.ndarray,
    plane_dist:   np.ndarray,   # (H, W) meters
    threshold_m:  float,
    obj_name:     str,
    output_path:  Path,
):
    """4-panel diagnostic: plane-dist map | masks+hull | grown | histogram."""
    H, W = mask_bool.shape
    growth_region = hull_mask & ~cleaned_mask
    eroded_px     = mask_bool & ~cleaned_mask

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Mask growth (plane distance): {obj_name}", fontsize=14,
                 fontweight="bold")

    # --- Panel 1: Plane distance map ---
    ax = axes[0, 0]
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    ax.imshow(depth_norm, cmap="gray")

    disp = np.full((H, W), np.nan, dtype=np.float32)
    disp[cleaned_mask]   = 0.0                         # mask itself → 0
    disp[growth_region]  = plane_dist[growth_region]   # hull gap → actual dist
    vmax = max(threshold_m * 4, 0.001)
    im = ax.imshow(disp, cmap="RdYlGn_r", vmin=0, vmax=vmax, alpha=0.75)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Plane dist (m)")

    # Mask boundary outline
    mask_edge = cleaned_mask & ~binary_erosion(cleaned_mask, iterations=1)
    edge_ov   = np.zeros((H, W, 4), dtype=np.float32)
    edge_ov[mask_edge] = [0, 1, 0, 1.0]
    ax.imshow(edge_ov)
    ax.set_title(f"Plane distance map  threshold={threshold_m*100:.1f} cm"
                 f"\n(green outline = cleaned mask)")
    ax.axis("off")

    # --- Panel 2: Original + cleaned + hull ---
    ax = axes[0, 1]
    vis1 = np.zeros((H, W, 3), dtype=np.uint8)
    vis1[hull_mask]    = [40, 40, 90]      # hull fill: dark blue
    vis1[cleaned_mask] = [0, 180, 80]      # cleaned mask: green
    vis1[eroded_px]    = [180, 40, 40]     # eroded-away pixels: red
    hull_edge = hull_mask & ~binary_erosion(hull_mask, iterations=1)
    vis1[hull_edge] = [255, 210, 0]        # hull boundary: yellow
    ax.imshow(vis1)
    ax.set_title(f"Orig {mask_bool.sum():,}px → Cleaned {cleaned_mask.sum():,}px"
                 f"  (red=eroded -{eroded_px.sum():,}px)\n"
                 f"Hull {hull_mask.sum():,}px (yellow)")
    ax.axis("off")

    # --- Panel 3: Grown mask ---
    ax = axes[1, 0]
    vis2 = np.zeros((H, W, 3), dtype=np.uint8)
    vis2[cleaned_mask] = [0, 200, 0]       # cleaned: green
    added = grown_mask & ~cleaned_mask
    vis2[added]        = [255, 165, 0]     # newly added: orange
    vis2[eroded_px]    = [140, 30, 30]     # eroded: dark red
    ax.imshow(vis2)
    ax.set_title(f"Grown: +{added.sum():,} added (orange)  "
                 f"-{eroded_px.sum():,} eroded (red)\n"
                 f"Net vs original: {int(grown_mask.sum()) - int(mask_bool.sum()):+,}px")
    ax.axis("off")

    # --- Panel 4: Plane distance histogram (growth region) ---
    ax = axes[1, 1]
    growth_dists = plane_dist[growth_region]
    if len(growth_dists) > 0:
        vmax_h = max(threshold_m * 5, 0.001)
        ax.hist(growth_dists, bins=80, range=(0, vmax_h),
                color="orange", alpha=0.8,
                label=f"Growth region ({len(growth_dists):,}px)", density=True)
    ax.axvline(threshold_m, color="red", ls="--", lw=2,
               label=f"Threshold = {threshold_m*100:.1f} cm")
    ax.set_xlabel("Plane distance (m)")
    ax.set_ylabel("Density")
    ax.set_title("Plane distance distribution — growth region")
    ax.legend(fontsize=8)

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
    moge     = np.load(str(MOGE_NPZ))
    pointmap = moge["points"].astype(np.float32)
    depth    = moge["depth"].astype(np.float32)
    print(f"  Pointmap shape={pointmap.shape}, depth [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"  threshold_m={THRESHOLD_M*100:.1f}cm  smooth_sigma={SMOOTH_SIGMA}")

    for name in OBJECTS:
        mask_path = MASKS_DIR / f"{name}.npy"
        if not mask_path.exists():
            print(f"\n  SKIP {name} — mask not found")
            continue

        print(f"\nProcessing: {name}")
        mask_bool = np.load(str(mask_path)).astype(bool)

        if mask_bool.shape != depth.shape:
            print(f"  WARNING: shape mismatch {mask_bool.shape} vs {depth.shape}")
            continue

        n_px = int(mask_bool.sum())
        print(f"  Mask: {n_px:,} / {mask_bool.size:,} px ({100.*n_px/mask_bool.size:.1f}%)")

        grown, hull_mask, cleaned, plane_dist, thr = \
            _grow_mask_plane_distance(mask_bool, pointmap)

        n_eroded = int((mask_bool & ~cleaned).sum())
        n_added  = int((grown & ~cleaned).sum())
        net      = int(grown.sum()) - int(mask_bool.sum())
        print(f"  Hull: {hull_mask.sum():,}px | "
              f"Eroded: -{n_eroded:,}px | "
              f"Added: +{n_added:,}px | "
              f"Net: {net:+,}px | "
              f"threshold={thr*100:.1f}cm")

        out_path = OUTPUT_DIR / f"{name}_mask_growth.png"
        visualize_mask_growth(
            mask_bool, grown, hull_mask, cleaned, depth,
            plane_dist, thr, name, out_path,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
