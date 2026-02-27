"""Inspect GLB file structure."""
import trimesh
import numpy as np
import sys

glb_path = sys.argv[1] if len(sys.argv) > 1 else "output/viga_test/green_tea_bottle_viga.glb"

print(f"Loading: {glb_path}")
mesh = trimesh.load(glb_path)

print(f"\nType: {type(mesh)}")

if hasattr(mesh, 'geometry'):
    print(f"Geometries: {list(mesh.geometry.keys())}")
    for name, geom in mesh.geometry.items():
        print(f"\n=== {name} ===")
        print(f"  Vertices: {len(geom.vertices)}")
        print(f"  Faces: {len(geom.faces)}")
        print(f"  Bounds: {geom.bounds}")
        print(f"  Extents: {geom.extents}")
        if hasattr(geom, 'visual'):
            print(f"  Visual: {type(geom.visual)}")
else:
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Bounds:\n{mesh.bounds}")
    print(f"Extents: {mesh.extents}")
    
    # Check vertex distribution
    verts = mesh.vertices
    print(f"\nVertex stats:")
    print(f"  X range: {verts[:,0].min():.3f} to {verts[:,0].max():.3f}")
    print(f"  Y range: {verts[:,1].min():.3f} to {verts[:,1].max():.3f}")
    print(f"  Z range: {verts[:,2].min():.3f} to {verts[:,2].max():.3f}")
    
    # Check if it's flat
    for axis, name in enumerate(['X', 'Y', 'Z']):
        axis_range = verts[:,axis].max() - verts[:,axis].min()
        if axis_range < 0.01:
            print(f"  WARNING: {name} axis is nearly flat (range={axis_range:.6f})")
