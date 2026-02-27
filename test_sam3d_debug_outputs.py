#!/usr/bin/env python3
"""Debug SAM3D outputs."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'third_party', 'sam3d'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'third_party', 'sam3d', 'notebook'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from inference import Inference, load_image

config_path = r"D:\Projects\ProjectGenesis\GenesisVIGA\utils\third_party\sam3d\checkpoints\hf\checkpoints\pipeline.yaml"
image_path = r"D:\Projects\ProjectGenesis\GenesisVIGA\test_results_images\01_greentea_input.jpg"

# Load masks
masks = np.load(r"D:\Projects\ProjectGenesis\GenesisVIGA\test_masks.npy")
mask = masks[0] > 0

print(f"Mask coverage: {100*np.sum(mask)/mask.size:.1f}%")

# Initialize
inference = Inference(config_path, compile=False)
image = load_image(image_path)

# Check the merged RGBA
rgba = inference.merge_mask_to_rgba(image, mask)
print(f"RGBA shape: {rgba.shape}")
print(f"Alpha channel - min: {rgba[..., 3].min()}, max: {rgba[..., 3].max()}")
print(f"Alpha=255 pixels: {np.sum(rgba[..., 3] == 255)} ({100*np.sum(rgba[..., 3] == 255)/rgba[..., 3].size:.1f}%)")

# Save RGBA for inspection
from PIL import Image as PILImage
PILImage.fromarray(rgba).save(r"D:\Projects\ProjectGenesis\GenesisVIGA\output\sam3d_test\debug_rgba.png")
print("Saved debug_rgba.png")

# Run inference
print("Running inference...")
output = inference(image, mask, seed=42)

# Check mesh
mesh = output.get("mesh")
if mesh:
    print(f"Mesh vertices: {mesh[0].vertices.shape}")
    print(f"Mesh faces: {mesh[0].faces.shape}")
    
    # Save raw mesh
    import trimesh
    verts = mesh[0].vertices.cpu().numpy()
    faces = mesh[0].faces.cpu().numpy()
    colors = mesh[0].vertex_attrs[:, :3].cpu().numpy()
    
    print(f"Vertex positions - min: {verts.min(axis=0)}, max: {verts.max(axis=0)}")
    print(f"Vertex colors - min: {colors.min()}, max: {colors.max()}")
    
    raw_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=(colors * 255).astype(np.uint8))
    raw_mesh.export(r"D:\Projects\ProjectGenesis\GenesisVIGA\output\sam3d_test\debug_raw_mesh.glb")
    print("Saved debug_raw_mesh.glb")

print("Done!")
