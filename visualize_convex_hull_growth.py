"""Visualize convex hull mask growth — Normal-consistency + 8-direction ray depth gate.

Algorithm (v9):
  1. Morphological opening (2x erosion + 2x dilation) to clean mask boundary.
  2. Convex hull of cleaned mask.
  3. Per-pixel surface normals from Gaussian-smoothed pointmap (sigma=2.0, ~13x13).
  4. Reference normal = mean of valid normals inside cleaned mask.
  5. Angle deviation map = arccos(|normals · ref|) per pixel.
  6. Adaptive threshold = clip(median + 2*std of angles inside mask, 10°, max_deg).
  7. Precompute ray_first_hit[H, W, 8]: for each pixel and each of 8 directions
     (N/NE/E/SE/S/SW/W/NW), the depth of the FIRST mask pixel hit along that ray.
     Cardinal directions: vectorized row/column scan. Diagonals: diagonal scan.
  8. For each hull pixel P:
       ray_depths = ray_first_hit[P.r, P.c, :]   # up to 8 values, NaN if no hit
       [dmin, dmax] = [min(ray_depths), max(ray_depths)]
  9. Accept hull pixel P if:
       angle < threshold                      (normal-consistency)
       AND dmin <= depth[P] <= dmax           (8-dir ray depth range gate)
       AND valid normal

Usage:
    python visualize_convex_hull_growth.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter
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
OUTPUT_DIR      = PROJECT_ROOT / "output/sam3d_dining_v9/vis"

SMOOTH_SIGMA    = 2.0   # Gaussian sigma for normal computation (~13x13)
MAX_ANGLE_DEG   = 60.0  # cap on adaptive normal threshold (degrees)
# v9 depth gate: 8-direction ray casting — accept P if depth[P] ∈ [min, max] of first hits

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
# 8-direction ray first-hit precomputation
# ---------------------------------------------------------------------------

def _precompute_ray_first_hit(cleaned: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """For each pixel (r,c), find depth of first mask pixel in each of 8 directions.

    Returns ray: (H, W, 8) float32. NaN = no mask pixel found in that direction.
    Direction order: N(0) NE(1) E(2) SE(3) S(4) SW(5) W(6) NW(7).

    Cardinal directions (N/E/S/W): vectorized row/column scan, O(H*W).
    Diagonal directions (NE/SE/SW/NW): diagonal scan, O(H*W).

    Recurrence for any direction d with unit step (dr, dc):
      ray[r, c, d] = depth[r, c]             if cleaned[r, c]
                   = ray[r+dr, c+dc, d]      else (propagate from the d-direction neighbor)
    Scan in reverse direction (opposite of d) so the neighbor is already computed.
    """
    H, W = cleaned.shape
    mask_dep = np.where(cleaned, depth, np.nan).astype(np.float32)
    ray = np.full((H, W, 8), np.nan, dtype=np.float32)

    # ---- N (0): dr=-1, dc=0  →  scan top→bottom, propagate downward ----
    ray[:, :, 0] = mask_dep.copy()
    for r in range(1, H):
        nan_r = np.isnan(ray[r, :, 0])
        ray[r, nan_r, 0] = ray[r - 1, nan_r, 0]

    # ---- S (4): dr=+1, dc=0  →  scan bottom→top, propagate upward ----
    ray[:, :, 4] = mask_dep.copy()
    for r in range(H - 2, -1, -1):
        nan_r = np.isnan(ray[r, :, 4])
        ray[r, nan_r, 4] = ray[r + 1, nan_r, 4]

    # ---- E (2): dr=0, dc=+1  →  scan right→left, propagate leftward ----
    ray[:, :, 2] = mask_dep.copy()
    for c in range(W - 2, -1, -1):
        nan_c = np.isnan(ray[:, c, 2])
        ray[nan_c, c, 2] = ray[nan_c, c + 1, 2]

    # ---- W (6): dr=0, dc=-1  →  scan left→right, propagate rightward ----
    ray[:, :, 6] = mask_dep.copy()
    for c in range(1, W):
        nan_c = np.isnan(ray[:, c, 6])
        ray[nan_c, c, 6] = ray[nan_c, c - 1, 6]

    # ---- NE (1): dr=-1, dc=+1  →  anti-diagonal r+c=k, scan top-right→bottom-left ----
    ray[:, :, 1] = np.full((H, W), np.nan, dtype=np.float32)
    for k in range(H + W - 1):
        c_max = min(W - 1, k)
        c_min = max(0, k - H + 1)
        for c in range(c_max, c_min - 1, -1):   # top-right to bottom-left
            r = k - c
            if 0 <= r < H:
                if cleaned[r, c]:
                    ray[r, c, 1] = depth[r, c]
                else:
                    nr, nc = r - 1, c + 1
                    if 0 <= nr < H and 0 <= nc < W:
                        ray[r, c, 1] = ray[nr, nc, 1]

    # ---- SW (5): dr=+1, dc=-1  →  anti-diagonal r+c=k, scan bottom-left→top-right ----
    ray[:, :, 5] = np.full((H, W), np.nan, dtype=np.float32)
    for k in range(H + W - 1):
        c_max = min(W - 1, k)
        c_min = max(0, k - H + 1)
        for c in range(c_min, c_max + 1):        # bottom-left to top-right
            r = k - c
            if 0 <= r < H:
                if cleaned[r, c]:
                    ray[r, c, 5] = depth[r, c]
                else:
                    nr, nc = r + 1, c - 1
                    if 0 <= nr < H and 0 <= nc < W:
                        ray[r, c, 5] = ray[nr, nc, 5]

    # ---- SE (3): dr=+1, dc=+1  →  diagonal r-c=k, scan bottom-right→top-left ----
    ray[:, :, 3] = np.full((H, W), np.nan, dtype=np.float32)
    for k in range(-(W - 1), H):
        r_min = max(0, k)
        r_max = min(H - 1, W - 1 + k)
        for r in range(r_max, r_min - 1, -1):   # bottom-right to top-left
            c = r - k
            if 0 <= c < W:
                if cleaned[r, c]:
                    ray[r, c, 3] = depth[r, c]
                else:
                    nr, nc = r + 1, c + 1
                    if 0 <= nr < H and 0 <= nc < W:
                        ray[r, c, 3] = ray[nr, nc, 3]

    # ---- NW (7): dr=-1, dc=-1  →  diagonal r-c=k, scan top-left→bottom-right ----
    ray[:, :, 7] = np.full((H, W), np.nan, dtype=np.float32)
    for k in range(-(W - 1), H):
        r_min = max(0, k)
        r_max = min(H - 1, W - 1 + k)
        for r in range(r_min, r_max + 1):        # top-left to bottom-right
            c = r - k
            if 0 <= c < W:
                if cleaned[r, c]:
                    ray[r, c, 7] = depth[r, c]
                else:
                    nr, nc = r - 1, c - 1
                    if 0 <= nr < H and 0 <= nc < W:
                        ray[r, c, 7] = ray[nr, nc, 7]

    return ray


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
) -> tuple:
    """Grow mask toward convex hull using normal-consistency + 8-dir ray depth gate.

    Depth gate (v9): cast a ray from P in each of 8 directions; accept P if
    depth[P] lies within [min, max] of the first mask pixel hit per direction.

    Returns
    -------
    grown        : (H, W) bool
    hull_mask    : (H, W) bool
    cleaned      : (H, W) bool
    angle_map    : (H, W) float32  — degrees, NaN outside hull region
    depth_diff   : (H, W) float32  — signed distance outside mask depth range
                                     (0 = inside range, >0 = out of range)
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

    # 6. Precompute 8-direction ray first-hit depths (once per object)
    ray_first_hit = _precompute_ray_first_hit(cleaned, depth)  # (H, W, 8)

    # 7. For each growth-region pixel, collect 8 ray hits → depth range [dmin, dmax]
    gr          = growth_region
    gr_rows, gr_cols = np.where(gr)
    p_depths    = depth[gr_rows, gr_cols]                       # (M,)

    angle_map[gr] = ang[gr]
    normal_ok[gr] = (ang[gr] < threshold_deg) & valid[gr]

    p_ray_hits  = ray_first_hit[gr_rows, gr_cols, :]           # (M, 8)
    dmin        = np.nanmin(p_ray_hits, axis=1)                 # (M,)
    dmax        = np.nanmax(p_ray_hits, axis=1)                 # (M,)
    has_any     = np.isfinite(dmin)                             # at least one direction hit

    # depth_diff: how far P is outside [dmin, dmax]; 0 = inside range
    dd_vals = (np.clip(dmin - p_depths, 0.0, None) +
               np.clip(p_depths - dmax, 0.0, None)).astype(np.float32)
    depth_diff[gr_rows, gr_cols] = dd_vals

    depth_ok[gr_rows, gr_cols] = has_any & (dd_vals == 0.0)

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
    """4-panel: depth+mask | mask+hull | mask+growth | angle histogram."""
    H, W = mask_bool.shape
    eroded_px     = mask_bool & ~cleaned
    growth_region = hull_mask & ~cleaned
    accepted      = grown & ~cleaned
    depth_only_fail = normal_ok & ~depth_ok & growth_region
    net           = int(grown.sum()) - int(mask_bool.sum())

    depth_n = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Mask growth (normal + depth range gate): {obj_name}",
                 fontsize=14, fontweight="bold")

    # --- Top-left: depth map + mask overlay ---
    ax = axes[0, 0]
    ax.imshow(depth_n, cmap="gray")
    ov = np.zeros((H, W, 4), dtype=np.float32)
    ov[mask_bool] = [0, 0.8, 0.3, 0.5]   # original mask: semi-transparent green
    ax.imshow(ov)
    ax.set_title(f"Depth + original mask ({mask_bool.sum():,}px)")
    ax.axis("off")
    ax.legend(handles=[
        mpatches.Patch(color=(0, 0.8, 0.3), alpha=0.5, label=f"Original mask ({mask_bool.sum():,}px)"),
    ], loc="lower right", fontsize=7, framealpha=0.7)

    # --- Top-right: mask + convex hull ---
    ax = axes[0, 1]
    vis1 = np.zeros((H, W, 3), dtype=np.uint8)
    vis1[hull_mask]  = [40, 40, 90]       # hull interior: dark blue
    vis1[cleaned]    = [0, 180, 80]       # cleaned mask: green
    vis1[eroded_px]  = [180, 40, 40]      # eroded pixels: red
    hull_edge = hull_mask & ~binary_erosion(hull_mask, iterations=1)
    vis1[hull_edge]  = [255, 210, 0]      # hull boundary: yellow
    ax.imshow(vis1)
    ax.set_title(f"Mask {cleaned.sum():,}px  Hull {hull_mask.sum():,}px  Eroded -{eroded_px.sum():,}px")
    ax.axis("off")
    ax.legend(handles=[
        mpatches.Patch(color=(0/255, 180/255, 80/255),   label=f"Cleaned mask ({cleaned.sum():,}px)"),
        mpatches.Patch(color=(40/255, 40/255, 90/255),   label=f"Hull interior ({(hull_mask & ~cleaned).sum():,}px)"),
        mpatches.Patch(color=(255/255, 210/255, 0/255),  label="Hull boundary edge"),
        mpatches.Patch(color=(180/255, 40/255, 40/255),  label=f"Eroded -{eroded_px.sum():,}px"),
    ], loc="lower right", fontsize=7, framealpha=0.7)

    # --- Bottom-left: grown mask ---
    ax = axes[1, 0]
    vis2 = np.zeros((H, W, 3), dtype=np.uint8)
    vis2[cleaned]          = [0, 200, 0]     # cleaned mask: green
    vis2[accepted]         = [255, 165, 0]   # added: orange
    vis2[eroded_px]        = [140, 30, 30]   # eroded: dark red
    vis2[depth_only_fail]  = [100, 100, 255] # normal-ok but depth-fail: blue
    ax.imshow(vis2)
    ax.set_title(
        f"Grown: +{accepted.sum():,}px  -eroded {eroded_px.sum():,}px  Net: {net:+,}px"
    )
    ax.axis("off")
    ax.legend(handles=[
        mpatches.Patch(color=(0/255, 200/255, 0/255),    label=f"Cleaned mask ({cleaned.sum():,}px)"),
        mpatches.Patch(color=(255/255, 165/255, 0/255),  label=f"New growth / accepted (+{accepted.sum():,}px)"),
        mpatches.Patch(color=(100/255, 100/255, 255/255),label=f"Normal-ok + out-of-range ({depth_only_fail.sum():,}px)"),
        mpatches.Patch(color=(140/255, 30/255, 30/255),  label=f"Eroded pixels (-{eroded_px.sum():,}px)"),
    ], loc="lower right", fontsize=7, framealpha=0.7)

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
          f"depth_gate=8-dir-ray")

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
