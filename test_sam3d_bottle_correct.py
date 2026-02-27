#!/usr/bin/env python3
"""Test SAM3D with CORRECTLY INVERTED mask."""
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'third_party', 'sam3d'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'third_party', 'sam3d', 'notebook'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import numpy as np
    print("Loading SAM3D...")
    from inference import Inference, load_image

    config_path = r"D:\Projects\ProjectGenesis\GenesisVIGA\utils\third_party\sam3d\checkpoints\hf\checkpoints\pipeline.yaml"
    image_path = r"D:\Projects\ProjectGenesis\GenesisVIGA\test_results_images\01_greentea_input.jpg"
    
    # Load masks - INVERT because mask stores 255=background, 0=object
    masks = np.load(r"D:\Projects\ProjectGenesis\GenesisVIGA\test_masks.npy")
    mask = masks[0] == 0  # INVERTED: select where mask is 0 (the bottle)
    
    print(f"Mask coverage (bottle): {100*np.sum(mask)/mask.size:.1f}%")

    print("Initializing inference...")
    inference = Inference(config_path, compile=False)

    print("Loading image...")
    image = load_image(image_path)

    print("Running inference with CORRECTED mask...")
    output = inference(image, mask, seed=42)
    
    print("SUCCESS!")
    
    # Save GLB
    output_dir = r"D:\Projects\ProjectGenesis\GenesisVIGA\output\sam3d_test"
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, "green_tea_bottle_CORRECT.glb")
    output["glb"].export(glb_path)
    print(f"Saved: {glb_path}")
    
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nDone!")
