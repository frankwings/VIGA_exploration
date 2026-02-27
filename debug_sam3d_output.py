"""Debug SAM3D output structure."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'third_party', 'sam3d', 'notebook'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils', 'third_party', 'sam3d'))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Loading SAM3D...")
from inference import Inference, load_image

config_path = r"D:\Projects\ProjectGenesis\GenesisVIGA\utils\third_party\sam3d\checkpoints\hf\checkpoints\pipeline.yaml"
image_path = r"D:\Projects\ProjectGenesis\GenesisVIGA\output\test_sam\green_tea_bottle.png"
mask_path = r"D:\Projects\ProjectGenesis\GenesisVIGA\output\test_sam\green_tea_bottle.npy"

print("Initializing inference...")
inference = Inference(config_path, compile=False)

print("Loading image and mask...")
image = load_image(image_path)
mask = np.load(mask_path)
mask = mask > 0

print(f"Image type: {type(image)}")
print(f"Image size: {image.size if hasattr(image, 'size') else 'N/A'}")
print(f"Mask shape: {mask.shape}")
print(f"Mask True pixels: {mask.sum()}")

print("\nRunning inference...")
output = inference(image, mask, seed=42)

print("\n=== OUTPUT ANALYSIS ===")
print(f"Output keys: {list(output.keys())}")

# Check mesh
if 'mesh' in output:
    mesh = output['mesh']
    print(f"\n[mesh] type: {type(mesh)}")
    if hasattr(mesh, 'vertices'):
        verts = mesh.vertices
        print(f"  vertices: {verts.shape}")
        print(f"  X range: {verts[:,0].min():.4f} to {verts[:,0].max():.4f}")
        print(f"  Y range: {verts[:,1].min():.4f} to {verts[:,1].max():.4f}")
        print(f"  Z range: {verts[:,2].min():.4f} to {verts[:,2].max():.4f}")
    if hasattr(mesh, 'faces'):
        print(f"  faces: {mesh.faces.shape}")

# Check GLB
if 'glb' in output:
    glb = output['glb']
    print(f"\n[glb] type: {type(glb)}")
    if hasattr(glb, 'vertices'):
        verts = glb.vertices
        print(f"  vertices: {verts.shape}")
        print(f"  X range: {verts[:,0].min():.4f} to {verts[:,0].max():.4f}")
        print(f"  Y range: {verts[:,1].min():.4f} to {verts[:,1].max():.4f}")
        print(f"  Z range: {verts[:,2].min():.4f} to {verts[:,2].max():.4f}")

# Check Gaussian
if 'gaussian' in output:
    gs = output['gaussian']
    print(f"\n[gaussian] type: {type(gs)}")
    if hasattr(gs, 'get_xyz'):
        xyz = gs.get_xyz
        if callable(xyz):
            xyz = xyz()
        print(f"  xyz shape: {xyz.shape if hasattr(xyz, 'shape') else 'N/A'}")

# Check transforms
for key in ['rotation', 'translation', 'scale']:
    if key in output:
        val = output[key]
        print(f"\n[{key}]")
        if hasattr(val, 'shape'):
            print(f"  shape: {val.shape}")
            print(f"  value: {val}")
        else:
            print(f"  value: {val}")

# Check pointmap
if 'pointmap' in output:
    pm = output['pointmap']
    print(f"\n[pointmap] type: {type(pm)}")
    if hasattr(pm, 'shape'):
        print(f"  shape: {pm.shape}")

print("\n=== DEBUG COMPLETE ===")
