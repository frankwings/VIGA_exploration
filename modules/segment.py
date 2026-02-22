"""Module 1: SAM Segmentation.

Runs SAM ViT-H on a target image and outputs all detected masks with
minimal filtering (only removes tiny noise masks and obvious backgrounds).

Usage:
    python modules/segment.py --image <target.jpg> --output-dir <output/segment/>

Output:
    segment_manifest.json   — manifest listing all masks
    mask_NNN.npy            — binary mask arrays (H, W), uint8 0/255
    mask_NNN.png            — masked RGBA image (object pixels only)
    viz/all_masks_grid.png  — overview grid
    viz/mask_NNN_overlay.png — per-mask red overlay
    viz/mask_NNN_binary.png  — per-mask binary (white on black)

Conda env: sam (Python 3.10, segment_anything)
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "utils", "third_party", "sam"))


# ---------------------------------------------------------------------------
# Mask filtering
# ---------------------------------------------------------------------------

def filter_masks(raw_masks: List[Dict[str, Any]],
                 min_area_ratio: float = 0.005,
                 max_area_ratio: float = 0.60) -> List[Dict[str, Any]]:
    """Keep all masks except tiny noise and obvious background.

    Args:
        raw_masks: SAM output list of mask dicts.
        min_area_ratio: Discard masks smaller than this fraction of the image.
        max_area_ratio: Discard masks larger than this fraction of the image.

    Returns:
        Filtered list, sorted by area descending.
    """
    if not raw_masks:
        return []

    image_area = raw_masks[0]["segmentation"].size
    min_area = image_area * min_area_ratio
    max_area = image_area * max_area_ratio

    kept = [m for m in raw_masks if min_area < m["area"] < max_area]
    kept.sort(key=lambda m: m["area"], reverse=True)
    return kept


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_mask_as_png(mask_seg: np.ndarray, original_image: np.ndarray,
                     output_path: str) -> None:
    """Save a mask as an RGBA PNG (object pixels opaque, rest transparent)."""
    h, w = mask_seg.shape
    img = original_image
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask_seg, :3] = img[mask_seg]
    rgba[mask_seg, 3] = 255

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(output_path)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def make_overlay(target: np.ndarray, mask_seg: np.ndarray,
                 color: tuple = (255, 0, 0), alpha: float = 0.45) -> Image.Image:
    """Red-tinted overlay of a mask on the target image."""
    vis = target.copy()
    overlay = np.full_like(vis, color, dtype=np.uint8)
    vis[mask_seg] = (vis[mask_seg].astype(float) * (1 - alpha) +
                     overlay[mask_seg].astype(float) * alpha).astype(np.uint8)
    return Image.fromarray(vis)


def make_binary(mask_seg: np.ndarray) -> Image.Image:
    """White-on-black binary mask image."""
    return Image.fromarray((mask_seg.astype(np.uint8) * 255))


def make_grid(images: List[Image.Image], labels: List[str],
              cols: int = 4, cell_w: int = 400, cell_h: int = 300) -> Image.Image:
    """Tile images into a labelled grid."""
    n = len(images)
    rows = math.ceil(n / cols)
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)

    for idx, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, r * cell_h
        resized = img.resize((cell_w, cell_h - 20), Image.LANCZOS)
        grid.paste(resized, (x0, y0))
        draw.text((x0 + 5, y0 + cell_h - 18), label, fill=(255, 255, 255))

    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Module 1: SAM Segmentation")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--checkpoint", default=None, help="SAM ViT-H checkpoint path")
    parser.add_argument("--min-area-ratio", type=float, default=0.005,
                        help="Min mask area as fraction of image (default 0.5%%)")
    parser.add_argument("--max-area-ratio", type=float, default=0.60,
                        help="Max mask area as fraction of image (default 60%%)")
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            ROOT, "utils", "third_party", "sam", "sam_vit_h_4b8939.pth"
        )

    output_dir = os.path.abspath(args.output_dir)
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    image_path = os.path.abspath(args.image)
    print(f"[SEGMENT] Image: {image_path}")
    print(f"[SEGMENT] Output: {output_dir}")

    # Load image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"[SEGMENT] ERROR: Failed to load image: {image_path}")
        sys.exit(1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    print(f"[SEGMENT] Image size: {w}x{h}")

    # Initialize SAM
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SEGMENT] Loading SAM ViT-H on {device}...")
    t0 = time.time()
    sam = sam_model_registry["vit_h"](checkpoint=args.checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print(f"[SEGMENT] SAM loaded in {time.time() - t0:.1f}s")

    # Generate masks
    print("[SEGMENT] Generating masks...")
    t0 = time.time()
    raw_masks = mask_generator.generate(image_rgb)
    print(f"[SEGMENT] Generated {len(raw_masks)} raw masks in {time.time() - t0:.1f}s")

    # Filter
    filtered = filter_masks(raw_masks, args.min_area_ratio, args.max_area_ratio)
    print(f"[SEGMENT] Kept {len(filtered)} masks after filtering "
          f"(removed {len(raw_masks) - len(filtered)})")

    if not filtered:
        print("[SEGMENT] WARNING: No masks after filtering")
        manifest = {
            "image": image_path,
            "image_shape": [h, w],
            "num_masks": 0,
            "masks": [],
        }
        with open(os.path.join(output_dir, "segment_manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return

    # Save masks and build manifest
    mask_entries = []
    overlay_images = []
    overlay_labels = []

    # Add target image as first grid cell
    overlay_images.append(Image.fromarray(image_rgb))
    overlay_labels.append("Original")

    for idx, mask_data in enumerate(filtered):
        mask_id = f"mask_{idx:03d}"
        seg = mask_data["segmentation"]  # bool (H, W)
        area_px = int(np.count_nonzero(seg))
        area_ratio = area_px / (h * w)

        # Bounding box from SAM (XYWH format)
        bbox = mask_data.get("bbox", [0, 0, w, h])
        bbox = [int(x) for x in bbox]

        # Save NPY
        npy_path = os.path.join(output_dir, f"{mask_id}.npy")
        np.save(npy_path, (seg.astype(np.uint8) * 255))

        # Save masked RGBA PNG
        png_path = os.path.join(output_dir, f"{mask_id}.png")
        save_mask_as_png(seg, image_rgb, png_path)

        # Visualization: overlay
        overlay_img = make_overlay(image_rgb, seg)
        overlay_path = os.path.join(viz_dir, f"{mask_id}_overlay.png")
        overlay_img.save(overlay_path)

        # Visualization: binary
        binary_img = make_binary(seg)
        binary_path = os.path.join(viz_dir, f"{mask_id}_binary.png")
        binary_img.save(binary_path)

        overlay_images.append(overlay_img)
        overlay_labels.append(f"{mask_id} ({area_ratio:.1%})")

        mask_entries.append({
            "id": mask_id,
            "npy_path": npy_path,
            "png_path": png_path,
            "area_pixels": area_px,
            "area_ratio": round(area_ratio, 4),
            "bbox": bbox,
            "predicted_iou": round(float(mask_data.get("predicted_iou", 0)), 4),
        })

    # Grid visualization
    grid = make_grid(overlay_images, overlay_labels)
    grid.save(os.path.join(viz_dir, "all_masks_grid.png"))

    # Write manifest
    manifest = {
        "image": image_path,
        "image_shape": [h, w],
        "num_masks": len(mask_entries),
        "masks": mask_entries,
    }
    manifest_path = os.path.join(output_dir, "segment_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[SEGMENT] Saved {len(mask_entries)} masks to {output_dir}")
    print(f"[SEGMENT] Manifest: {manifest_path}")
    for m in mask_entries:
        print(f"  {m['id']}: {m['area_pixels']}px ({m['area_ratio']:.1%})")


if __name__ == "__main__":
    main()
