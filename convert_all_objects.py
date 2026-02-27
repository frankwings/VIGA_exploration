#!/usr/bin/env python3
"""Convert all segmented objects to 3D using SAM3D."""
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, r'D:\Projects\ProjectGenesis\GenesisVIGA\utils\third_party\sam3d')
sys.path.insert(0, r'D:\Projects\ProjectGenesis\GenesisVIGA\utils\third_party\sam3d\notebook')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from inference import Inference, load_image
from PIL import Image

# Input folder with segmented objects
sam_init_dir = Path(r'D:\Projects\ProjectGenesis\GenesisVIGA\output\dynamic_scene\20260130_154003\greentea\sam_init')
output_dir = Path(r'D:\Projects\ProjectGenesis\GenesisVIGA\output\sam3d_objects')
output_dir.mkdir(exist_ok=True)

config_path = r'D:\Projects\ProjectGenesis\GenesisVIGA\utils\third_party\sam3d\checkpoints\hf\checkpoints\pipeline.yaml'

# Objects to convert (the actual object images, not the inverted ones)
objects_to_convert = [
    'green_tea_bottle_1.png',  # Bottle body
    'bottle_cap.png',          # Cap
    'bottle_neck.png',         # Neck
    'bottle_wrapper.png',      # Label/wrapper
]

print("Initializing SAM3D...")
inference = Inference(config_path, compile=False)

for obj_file in objects_to_convert:
    obj_path = sam_init_dir / obj_file
    if not obj_path.exists():
        print(f"Skipping {obj_file} - not found")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing: {obj_file}")
    print(f"{'='*60}")
    
    # Load image
    img = Image.open(obj_path).convert('RGBA')
    img_arr = np.array(img)
    
    # Extract mask from alpha channel
    mask = img_arr[..., 3] > 0
    rgb = img_arr[..., :3]
    
    print(f"Image size: {img.size}")
    print(f"Mask coverage: {100*np.sum(mask)/mask.size:.1f}%")
    
    if np.sum(mask) < 1000:
        print(f"Skipping - mask too small")
        continue
    
    # Run inference
    print("Running SAM3D inference...")
    try:
        output = inference(rgb, mask, seed=42)
        
        # Save GLB
        obj_name = obj_file.replace('.png', '')
        glb_path = output_dir / f"{obj_name}.glb"
        output['glb'].export(str(glb_path))
        print(f"Saved: {glb_path}")
    except Exception as e:
        print(f"Error: {e}")
        continue

print("\n" + "="*60)
print("DONE!")
print(f"Output directory: {output_dir}")
