"""Module 2: Object Recognition (VLM naming).

Reads masked PNG images from Module 1 (Segment) and uses a VLM to assign
descriptive names to each object. All masks are kept, including those
identified as background.

Usage:
    python modules/recognize.py \
        --input-manifest <output/segment/segment_manifest.json> \
        --output-dir <output/recognize/> \
        --model gemini-2.5-flash

Output:
    recognize_manifest.json       — manifest listing named objects
    {name}.npy                    — renamed mask arrays
    {name}.png                    — renamed masked RGBA images
    viz/named_objects_grid.png    — grid with object name labels

Conda env: agent (Python 3.10, google-genai SDK)
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "utils"))


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    return sanitized or "object"


def _build_prompt(existing_names: Optional[List[str]] = None) -> str:
    """Build the VLM prompt text."""
    existing_str = ""
    if existing_names:
        existing_str = (
            f"\n\nAlready identified objects (do not use these names): "
            f"{', '.join(existing_names)}"
        )
    return (
        "Look at the first image showing a segmented object, and the "
        "second image showing the original image that contains this "
        "object. Identify what this object is and provide a concise, "
        "descriptive name for it (e.g., 'red_chair', 'wooden_table'). "
        "If the first image is not clear, check the second image to "
        "get the whole context.\n\n"
        "If the mask covers background elements (walls, floor, ceiling), "
        "still describe what you see (e.g., 'floor_area', 'wall_section', "
        "'ceiling_light'). Always provide a descriptive name.\n\n"
        "Use only lowercase letters, numbers, and underscores. The "
        "name should be a single word or short phrase (2-3 words max, "
        f"use underscores to separate words).{existing_str}\n\n"
        "Respond with ONLY the object name, nothing else."
    )


def _get_name_gemini(
    image_path: str,
    ori_img_path: str,
    model: str,
    existing_names: List[str],
    api_key: str,
    max_retries: int = 3,
) -> str:
    """Call Gemini via google-genai SDK with retry on rate limits."""
    from google import genai

    client = genai.Client(api_key=api_key)
    mask_img = Image.open(image_path).convert("RGB")
    ori_img = Image.open(ori_img_path).convert("RGB")
    prompt = _build_prompt(existing_names)

    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[mask_img, ori_img, prompt],
            )
            return resp.text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                wait = 15 * (attempt + 1)
                print(f"[RECOGNIZE]   Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def _get_name_openai(
    image_path: str,
    ori_img_path: str,
    model: str,
    existing_names: List[str],
) -> str:
    """Call VLM via OpenAI-compatible client (GPT-4o, etc.)."""
    from common import build_client, get_image_base64

    image_b64 = get_image_base64(image_path)
    ori_img_b64 = get_image_base64(ori_img_path)
    client = build_client(model)
    prompt = _build_prompt(existing_names)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_b64}},
                {"type": "image_url", "image_url": {"url": ori_img_b64}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content.strip()


def get_object_name_from_vlm(
    image_path: str,
    ori_img_path: str,
    model: str = "gemini-2.5-flash",
    existing_names: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> str:
    """Use a VLM to identify the object in an image and return a unique name."""
    if existing_names is None:
        existing_names = []

    try:
        if "gemini" in model.lower() and api_key:
            raw_name = _get_name_gemini(
                image_path, ori_img_path, model, existing_names, api_key
            )
        else:
            raw_name = _get_name_openai(
                image_path, ori_img_path, model, existing_names
            )

        object_name = raw_name.strip("\"'")
        object_name = re.sub(r"\s+", "_", object_name)
        object_name = sanitize_filename(object_name)

        # Ensure uniqueness
        base_name = object_name
        counter = 1
        while object_name in existing_names:
            object_name = f"{base_name}_{counter}"
            counter += 1

        return object_name

    except Exception as e:
        print(f"[RECOGNIZE] VLM naming failed: {e}, using fallback name")
        base_name = "object"
        counter = 1
        while f"{base_name}_{counter}" in existing_names:
            counter += 1
        return f"{base_name}_{counter}"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_grid(images: list, labels: list, cols: int = 4,
              cell_w: int = 400, cell_h: int = 300) -> Image.Image:
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
    parser = argparse.ArgumentParser(description="Module 2: Object Recognition")
    parser.add_argument("--input-manifest", required=True,
                        help="Path to segment_manifest.json from Module 1")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="VLM model for naming (default: gemini-2.5-flash)")
    parser.add_argument("--gemini-api-key", default=None,
                        help="Gemini API key (reads from _api_keys.py if not set)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    # Resolve Gemini API key
    api_key = args.gemini_api_key
    if api_key is None and "gemini" in args.model.lower():
        try:
            from _api_keys import GEMINI_API_KEY
            api_key = GEMINI_API_KEY
        except ImportError:
            api_key = os.environ.get("GEMINI_API_KEY")
    if "gemini" in args.model.lower() and not api_key:
        print("[RECOGNIZE] WARNING: No Gemini API key found, VLM calls will fail")

    # Read segment manifest
    with open(args.input_manifest, "r", encoding="utf-8") as f:
        seg_manifest = json.load(f)

    image_path = seg_manifest["image"]
    masks = seg_manifest["masks"]

    print(f"[RECOGNIZE] Input manifest: {args.input_manifest}")
    print(f"[RECOGNIZE] Image: {image_path}")
    print(f"[RECOGNIZE] {len(masks)} masks to identify")
    print(f"[RECOGNIZE] Model: {args.model}")

    # Process each mask
    object_names: List[str] = []
    objects = []
    grid_images = []
    grid_labels = []

    # Add target as first grid cell
    target_img = Image.open(image_path).convert("RGB")
    grid_images.append(target_img)
    grid_labels.append("Original")

    for idx, mask_entry in enumerate(masks):
        mask_id = mask_entry["id"]
        png_path = mask_entry["png_path"]
        npy_path = mask_entry["npy_path"]

        print(f"[RECOGNIZE] Identifying {mask_id} ({idx+1}/{len(masks)})...",
              flush=True)

        # Call VLM
        name = get_object_name_from_vlm(
            png_path, image_path, model=args.model,
            existing_names=object_names, api_key=api_key
        )

        object_names.append(name)
        print(f"  {mask_id} -> {name}")

        # Copy files with new names
        out_npy = os.path.join(output_dir, f"{name}.npy")
        out_png = os.path.join(output_dir, f"{name}.png")
        shutil.copy2(npy_path, out_npy)
        shutil.copy2(png_path, out_png)

        objects.append({
            "name": name,
            "mask_id": mask_id,
            "npy_path": out_npy,
            "png_path": out_png,
            "area_ratio": mask_entry.get("area_ratio", 0),
        })

        # For grid
        mask_img = Image.open(png_path).convert("RGB")
        grid_images.append(mask_img)
        grid_labels.append(name)

    # Grid visualization
    if grid_images:
        grid = make_grid(grid_images, grid_labels)
        grid.save(os.path.join(viz_dir, "named_objects_grid.png"))

    # Write manifest
    manifest = {
        "image": image_path,
        "objects": objects,
    }
    manifest_path = os.path.join(output_dir, "recognize_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[RECOGNIZE] Identified {len(objects)} objects (all masks kept)")
    print(f"[RECOGNIZE] Manifest: {manifest_path}")
    for obj in objects:
        print(f"  {obj['mask_id']} -> {obj['name']}")


if __name__ == "__main__":
    main()
