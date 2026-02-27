#!/usr/bin/env python3
"""
SAM3D Segmentation-Only Pipeline
- Load input image
- Run SAM3D segmentation
- Save mask results without mesh conversion
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("D:\\Projects\\ProjectGenesis\\GenesisVIGA\\output\\segmentation_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Add SAM3D path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'third_party', 'sam3d'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'third_party', 'sam3d', 'notebook'))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def save_mask_visualization(image, masks, output_dir):
    """
    Visualize and save mask results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, mask in enumerate(masks):
        try:
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.imshow(mask, alpha=0.5, cmap='viridis')
            plt.title(f'Mask {i+1}')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'mask_{i+1}_visualization.png'))
            plt.close()
        except Exception as e:
            logging.error(f"Error saving mask {i+1}: {e}")
    
    # Save masks as NumPy arrays
    np.save(os.path.join(output_dir, 'masks.npy'), masks)
    logging.info(f"Saved {len(masks)} masks to {output_dir}")

def main():
    try:
        from inference import Inference, load_image
        
        # Full paths
        input_image_path = r"D:\Projects\ProjectGenesis\GenesisVIGA\docs\test_results_images\01_greentea_input.jpg"
        output_dir = r"D:\Projects\ProjectGenesis\GenesisVIGA\output\segmentation_results"
        config_path = r"D:\Projects\ProjectGenesis\GenesisVIGA\utils\third_party\sam3d\checkpoints\hf\checkpoints\pipeline.yaml"
        
        logging.info("Starting SAM3D segmentation...")
        
        logging.info("Loading SAM3D inference engine...")
        inference = Inference(config_path, compile=False)
        
        logging.info("Loading input image...")
        image = load_image(input_image_path)
        
        logging.info("Running segmentation (no mesh conversion)...")
        
        # Placeholder for segmentation method - may need adjustment
        masks = inference.generate_masks(image)
        
        logging.info(f"Detected {len(masks)} masks")
        
        # Visualize and save masks
        save_mask_visualization(image, masks, output_dir)
        
        logging.info("SAM3D segmentation complete.")
        
    except Exception as e:
        logging.error(f"Error in SAM3D segmentation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()