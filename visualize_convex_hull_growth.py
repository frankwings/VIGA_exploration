"""Visualize convex hull mask growth — Normal-consistency + depth-difference gate.

Algorithm (v6):
  1. Morphological opening (2x erosion + 2x dilation) to clean mask boundary.
  2. Convex hull of cleaned mask.
  3. Per-pixel surface normals from Gaussian-smoothed pointmap (sigma=2.0, ~13x13).
  4. Reference normal = mean of valid normals inside cleaned mask.
  5. Angle deviation map = arccos(|normals · ref|) per pixel.
  6. Adaptive threshold = clip(median + 2*std of angles inside mask, 10°, max_deg).
  7. EDT nearest cleaned-mask pixel → depth of nearest neighbor.
  8. Accept hull pixel if:
       angle < threshold   (normal-consistency)
       AND |depth_P - depth_neighbor| < depth_thresh_m   (depth gate)
       AND valid normal

Usage:
    python visualize_convex_hull_growth.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import (binary_erosion, binary_dilation, gaussian_filter,
                           distance_transform_edt)
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

MASKS_DIR       = PROJECT_ROOT / "output/sam3d_dining_v4"
MOGE_NPZ        = PROJECT_ROOT / "output/sam3d_dining_v4/target_moge.npz"
OUTPUT_DIR      = PROJECT_ROOT / "output/sam3d_dining_v6/vis"

SMOOTH_SIGMA    = 2.0   # Gaussian sigma for normal computation (~13x13)
MAX_ANGLE_DEG   = 60.0  # cap on adaptive normal threshold (degrees)
DEPTH_THRESH_M  = 0.05  # max depth difference to nearest mask pixel (5 cm)

# ---------------------------------------------------------------------------
# Convex hull helper
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
    rmin, rmax = max(0, rows.min()-2), min(H-1, rows.max()+2)
    cmin, cmax = max(0, cols.min()-2), min(W-1, cols.max()+2)
    yy, xx = np.mgrid[rmin:rmax+1, cmin:cmax+1]
    inside = hull_path.contains_points(
        np.column_stack([xx.ravel(), yy.ravel()])
    ).reshape(yy.shape)
    out = np.zeros_like(mask_bool)
    out[rmin:rmax+1, cmin:cmax+1] = inside
    return out

# ---------------------------------------------------------------------------
# Normal computation
# ---------------------------------------------------------------------------

def _compute_normals(pointmap: np.ndarray,
                     smooth_sigma: float = SMOOTH_SIGMA) -> tuple:
    """Returns (normals H×W×3, valid H×W bool)."""
    pm = gaussian_filter(pointmap.astype(np.float32),
                         sigma=[smooth_sigma, smooth_sigma, 0])
    dx = np.zeros_like(pm);  dy = np.zeros_like(pm)
    dx[:, 1:-1] = pm[:, 2:] - pm[:, :-2]
    dy[1:-1, :] = pm[2:,  :] - pm[:-2, :]
    n   = np.cross(dx, dy)
    nrm = np.linalg.norm(n, axis=-1, keepdims=True)
    valid = nrm[..., 0] > 1e-10
    n[valid]  /= nrm[valid]
    n[~valid]  = 0.0
    return n.astype(np.float32), valid

# ---------------------------------------------------------------------------
# Main growth function
# ---------------------------------------------------------------------------

def _grow_mask_normal_depth(
    mask_bool:      np.ndarray,
    pointmap:       np.ndarray,
    depth:          np.ndarray,
    smooth_sigma:   float = SMOOTH_SIGMA,
    erosion_iters:  int   = 2,
    dilation_iters: int   = 2,
    max_angle_deg:  float = MAX_ANGLE_DEG,
    depth_thresh_m: float = DEPTH_THRESH_M,
) -> tuple:
    """Grow mask toward convex hull using normal-consistency + depth gate.

    Returns
    -------
    grown        : (H, W) bool
    hull_mask    : (H, W) bool
    cleaned      : (H, W) bool
    angle_map    : (H, W) float32  — degrees, NaN outside hull region
    depth_diff   : (H, W) float32  — |depth_P - depth_neighbor|, NaN outside hull
    normal_ok    : (H, W) bool     — hull pixels passing normal gate
    depth_ok     : (H, W) bool     — hull pixels passing depth gate
    threshold_deg: float           — adaptive angle threshold used
    """
    H, W = mask_bool.shape

    # 1. Morphological opening
    cleaned = mask_bool.copy()
    if mask_bool.sum() > 50:
        eroded = binary_erosion(mask_bool, iterations=erosion_iters)
        if eroded.sum() >= 10:
            cleaned = binary_dilation(eroded, iterations=dilation_iters)

    hull_mask     = _make_convex_hull_mask(cleaned)
    growth_region = hull_mask & ~cleaned

    angle_map  = np.full((H, W), np.nan, dtype=np.float32)
    depth_diff = np.full((H, W), np.nan, dtype=np.float32)
    normal_ok  = np.zeros((H, W), dtype=bool)
    depth_ok   = np.zeros((H, W), dtype=bool)
    threshold_deg = max_angle_deg

    if not growth_region.any() or cleaned.sum() < 3:
        return cleaned, hull_mask, cleaned, angle_map, depth_diff, normal_ok, depth_ok, threshold_deg

    # 2. Normals
    normals, valid = _compute_normals(pointmap, smooth_sigma)

    # 3. Reference normal from cleaned mask
    inside_valid = cleaned & valid
    if inside_valid.sum() < 3:
        return cleaned, hull_mask, cleaned, angle_map, depth_diff, normal_ok, depth_ok, threshold_deg
    ref = normals[inside_valid].mean(axis=0)
    ref_nrm = np.linalg.norm(ref)
    if ref_nrm < 1e-10:
        return cleaned, hull_mask, cleaned, angle_map, depth_diff, normal_ok, depth_ok, threshold_deg
    ref /= ref_nrm

    # 4. Angle deviation map (degrees)
    dots = np.clip(np.abs((normals * ref).sum(axis=-1)), 0.0, 1.0)
    ang  = np.degrees(np.arccos(dots)).astype(np.float32)   # (H, W)

    # 5. Adaptive threshold from cleaned mask pixels
    ang_inside = ang[inside_valid]
    threshold_deg = float(np.clip(
        np.median(ang_inside) + 2.0 * ang_inside.std(),
        10.0, max_angle_deg
    ))

    # 6. EDT: nearest cleaned-mask pixel for each hull pixel
    _, nearest_idx = distance_transform_edt(~cleaned, return_indices=True)
    depth_neighbor = depth[nearest_idx[0], nearest_idx[1]]   # (H, W)

    # 7. Evaluate gates on growth_region only
    gr = growth_region
    angle_map[gr]  = ang[gr]
    dd = np.abs(depth - depth_neighbor).astype(np.float32)
    depth_diff[gr] = dd[gr]

    normal_ok[gr] = (ang[gr] < threshold_deg) & valid[gr]
    depth_ok[gr]  = dd[gr] < depth_thresh_m

    # 8. Accept if both
    accept = normal_ok & depth_ok

    grown = cleaned.copy()
    grown[accept] = True

    return grown, hull_mask, cleaned, angle_map, depth_diff, normal_ok, depth_ok, threshold_deg

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_mask_growth(
    mask_bool:     np.ndarray,
    grown:         np.ndarray,
    hull_mask:     np.ndarray,
    cleaned:       np.ndarray,
    depth:         np.ndarray,
    angle_map:     np.ndarray,
    depth_diff:    np.ndarray,
    normal_ok:     np.ndarray,
    depth_ok:      np.ndarray,
    threshold_deg: float,
    obj_name:      str,
    output_path:   Path,
):
    """4-panel: depth map | normal angle map | grown mask | angle histogram."""
    H, W = mask_bool.shape
    eroded_px     = mask_bool & ~cleaned
    growth_region = hull_mask & ~cleaned

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Mask growth (normal + depth gate): {obj_name}",
                 fontsize=14, fontweight="bold")

    # --- Panel 1: Depth map ---
    ax = axes[0, 0]
    depth_n = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    ax.imshow(depth_n, cmap="gray")
    ax.set_title("Depth map")
    ax.axis("off")

    # --- Panel 2: Normal angle map ---
    ax = axes[0, 1]
    ax.imshow(depth_n, cmap="gray")
    disp = np.full((H, W), np.nan, dtype=np.float32)
    disp[cleaned]        = 0.0
    disp[growth_region]  = angle_map[growth_region]
    im = ax.imshow(disp, cmap="RdYlGn_r", vmin=0, vmax=MAX_ANGLE_DEG, alpha=0.75)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Angle deviation (°)")
    mask_edge = cleaned & ~binary_erosion(cleaned, iterations=1)
    ov = np.zeros((H, W, 4), dtype=np.float32)
    ov[mask_edge] = [0, 1, 0, 1.0]
    ax.imshow(ov)
    ax.set_title(f"Normal angle map  (threshold = {threshold_deg:.1f}°)\n"
                 "green = within threshold, red = rejected")
    ax.axis("off")

    # --- Panel 3: Grown mask breakdown ---
    ax = axes[1, 0]
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[hull_mask]  = [40, 40, 90]
    vis[cleaned]    = [0, 180, 80]
    vis[eroded_px]  = [180, 40, 40]
    # accepted: both gates pass
    accepted = grown & ~cleaned
    vis[accepted]   = [255, 165, 0]
    # rejected by depth only (normal ok, depth failed)
    depth_only_fail = normal_ok & ~depth_ok & growth_region
    vis[depth_only_fail] = [100, 100, 255]
    ax.imshow(vis)
    net = int(grown.sum()) - int(mask_bool.sum())
    ax.set_title(
        f"Grown: +{accepted.sum():,}px (orange)  -eroded {eroded_px.sum():,}px\n"
        f"Blue = normal-ok but depth-fail ({depth_only_fail.sum():,}px)  Net: {net:+,}px"
    )
    ax.axis("off")

    # --- Panel 4: Angle histogram ---
    ax = axes[1, 1]
    gr_angles = angle_map[growth_region]
    gr_angles  = gr_angles[np.isfinite(gr_angles)]
    if len(gr_angles) > 0:
        ax.hist(gr_angles, bins=60, range=(0, MAX_ANGLE_DEG),
                color="steelblue", alpha=0.8, density=True,
                label=f"Growth region ({len(gr_angles):,}px)")
    ax.axvline(threshold_deg, color="red", ls="--", lw=2,
               label=f"Threshold {threshold_deg:.1f}°")
    ax.set_xlabel("Normal angle deviation (°)")
    ax.set_ylabel("Density")
    ax.set_title("Angle distribution — growth region")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MAX_ANGLE_DEG = MAX_ANGLE_DEG   # make accessible to visualize_mask_growth

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading MoGe data: {MOGE_NPZ}")
    moge     = np.load(str(MOGE_NPZ))
    pointmap = moge["points"].astype(np.float32)
    depth    = moge["depth"].astype(np.float32)
    print(f"  Pointmap {pointmap.shape}, depth [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"  smooth_sigma={SMOOTH_SIGMA}  max_angle={MAX_ANGLE_DEG}°  "
          f"depth_thresh={DEPTH_THRESH_M*100:.0f}cm")

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
        print(f"  Mask: {n_px:,} px")

        grown, hull_mask, cleaned, angle_map, depth_diff, normal_ok, depth_ok, threshold_deg = \
            _grow_mask_normal_depth(mask_bool, pointmap, depth)

        growth_region = hull_mask & ~cleaned
        n_eroded       = int((mask_bool & ~cleaned).sum())
        n_added        = int((grown & ~cleaned).sum())
        n_normal_only  = int((normal_ok & ~depth_ok & growth_region).sum())
        n_depth_only   = int((~normal_ok & depth_ok & growth_region).sum())
        net            = int(grown.sum()) - int(mask_bool.sum())

        print(f"  Hull: {hull_mask.sum():,}px | Eroded: -{n_eroded:,}px | "
              f"Added: +{n_added:,}px | Net: {net:+,}px")
        print(f"  Threshold: {threshold_deg:.1f} deg  |  "
              f"Normal+ok+Depth-fail: {n_normal_only:,}px  |  "
              f"Normal-fail+Depth+ok: {n_depth_only:,}px")

        out_path = OUTPUT_DIR / f"{name}_mask_growth.png"
        visualize_mask_growth(
            mask_bool, grown, hull_mask, cleaned, depth,
            angle_map, depth_diff, normal_ok, depth_ok, threshold_deg,
            name, out_path,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
