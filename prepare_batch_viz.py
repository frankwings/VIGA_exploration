"""Prepare batch run output for visualization.

Creates:
  1. Per-object masked PNG images (object on white background)
  2. object_transforms.json (combined transforms from individual info JSONs)

Usage:
    python prepare_batch_viz.py --batch-dir output/sam3d_dining_batch \
        --image data/static_scene/dining/target_resized.jpg
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-dir", required=True)
    p.add_argument("--image", required=True)
    args = p.parse_args()

    batch_dir = os.path.abspath(args.batch_dir)
    image_path = os.path.abspath(args.image)

    # Load target image
    img = np.array(Image.open(image_path).convert("RGB"))
    print(f"Target image: {img.shape}")

    # Load object names
    names_path = os.path.join(batch_dir, "all_masks_object_names.json")
    with open(names_path, "r", encoding="utf-8") as f:
        names_data = json.load(f)
    object_names = names_data.get("object_mapping", names_data.get("object_names", []))
    print(f"Objects: {object_names}")

    # Create masked PNGs
    for name in object_names:
        mask_path = os.path.join(batch_dir, f"{name}.npy")
        png_path = os.path.join(batch_dir, f"{name}.png")
        if os.path.exists(png_path):
            print(f"  {name}.png already exists, skipping")
            continue
        if not os.path.exists(mask_path):
            print(f"  WARNING: {name}.npy not found, skipping")
            continue

        mask = np.load(mask_path)
        mask = mask > 0
        if mask.ndim == 3:
            mask = mask[..., 0]

        # Apply mask to image (white background)
        masked = np.full_like(img, 255)
        masked[mask] = img[mask]
        Image.fromarray(masked).save(png_path)
        print(f"  Created {name}.png")

    # Build object_transforms.json
    transforms = []
    for name in object_names:
        info_path = os.path.join(batch_dir, f"{name}_info.json")
        if not os.path.exists(info_path):
            print(f"  WARNING: {name}_info.json not found, skipping")
            continue
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        # Fix glb_path to local path
        glb_path = os.path.join(batch_dir, f"{name}.glb")
        transforms.append({
            "object_name": name,
            "glb_path": glb_path,
            "translation": info["translation"],
            "rotation": info["rotation"],
            "scale": info["scale"],
            "iou": info.get("iou"),
        })

    transforms_path = os.path.join(batch_dir, "object_transforms.json")
    with open(transforms_path, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)
    print(f"\nWrote {len(transforms)} transforms to {transforms_path}")


if __name__ == "__main__":
    main()
