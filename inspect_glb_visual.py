"""Inspect visual format of GLB files to debug vertex color issues."""
import sys
import os
import trimesh

paths = sys.argv[1:]
if not paths:
    paths = [
        "output/modular_dining_v2/3d_reconstruction/sofa_with_blanket_canonical.glb",
        "output/modular_dining_v2/2d3d_registration/sofa_with_blanket.glb",
    ]

for path in paths:
    if not os.path.exists(path):
        print(f"{path}: NOT FOUND")
        continue
    scene = trimesh.load(path)
    print(f"\n=== {os.path.basename(path)} ===")
    print(f"Type: {type(scene).__name__}")

    geoms = {}
    if isinstance(scene, trimesh.Scene):
        geoms = dict(scene.geometry)
    else:
        geoms = {"mesh": scene}

    for name, geom in geoms.items():
        print(f"  Geometry: {name}")
        print(f"    Vertices: {geom.vertices.shape}")
        print(f"    Faces: {geom.faces.shape}")
        v = geom.visual
        vtype = type(v).__name__
        print(f"    Visual type: {vtype}")

        if hasattr(v, "vertex_colors"):
            vc = v.vertex_colors
            if vc is not None and hasattr(vc, "shape"):
                print(f"    vertex_colors: {vc.shape} {vc.dtype}")
                if len(vc) > 0:
                    print(f"    sample: {vc[:3].tolist()}")
                    unique = len(set(map(tuple, vc[:100].tolist())))
                    print(f"    unique in first 100: {unique}")
            else:
                print(f"    vertex_colors: None")

        if hasattr(v, "material"):
            mat = v.material
            print(f"    Material: {type(mat).__name__}")
            if hasattr(mat, "main_color"):
                print(f"    main_color: {mat.main_color}")

        if vtype == "TextureVisuals":
            if hasattr(v, "uv") and v.uv is not None:
                print(f"    UV coords: {v.uv.shape}")
            if hasattr(v, "material") and v.material is not None:
                mat = v.material
                if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                    tex = mat.baseColorTexture
                    print(f"    Texture: {tex.size if hasattr(tex, 'size') else 'present'}")
                if hasattr(mat, "image") and mat.image is not None:
                    print(f"    Image: {mat.image.size}")
