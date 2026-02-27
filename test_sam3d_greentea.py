#!/usr/bin/env python3
"""Test SAM3D with green tea bottle image."""
import os
import sys
import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'third_party', 'sam3d'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'third_party', 'sam3d', 'notebook'))

# Set environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Loading SAM3D...")
from inference import Inference, load_image

# Config path
config_path = r"D:\Projects\ProjectGenesis\GenesisVIGA\utils\third_party\sam3d\checkpoints\hf\checkpoints\pipeline.yaml"

# Image and mask
image_path = r"D:\Projects\ProjectGenesis\GenesisVIGA\green_tea_bottle.png"
mask_path = r"D:\Projects\ProjectGenesis\GenesisVIGA\green_tea_bottle.npy"

print(f"Config: {config_path}")
print(f"Image: {image_path}")
print(f"Mask: {mask_path}")

# Check files exist
if not os.path.exists(config_path):
    print(f"ERROR: Config not found: {config_path}")
    sys.exit(1)
if not os.path.exists(image_path):
    print(f"ERROR: Image not found: {image_path}")
    sys.exit(1)
if not os.path.exists(mask_path):
    print(f"ERROR: Mask not found: {mask_path}")
    sys.exit(1)

print("Initializing inference...")
inference = Inference(config_path, compile=False)

print("Loading image...")
image = load_image(image_path)

print("Loading mask...")
mask = np.load(mask_path)
mask = mask > 0

print(f"Image shape: {image.size if hasattr(image, 'size') else 'unknown'}")
print(f"Mask shape: {mask.shape}")

print("Running inference...")
output = inference(image, mask, seed=42)

print("Success!")
print(f"Output keys: {output.keys()}")

# Save GLB
output_dir = r"D:\Projects\ProjectGenesis\GenesisVIGA\output\sam3d_test"
os.makedirs(output_dir, exist_ok=True)
glb_path = os.path.join(output_dir, "green_tea_bottle.glb")

mesh = output["glb"]
mesh.export(glb_path)
print(f"Saved GLB to: {glb_path}")
