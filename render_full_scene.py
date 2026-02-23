"""Render all reconstructed GLBs in a single Blender scene with MoGe camera.

Usage:
    blender -b -P render_full_scene.py -- <glb_dir> <moge_npz> <output_png>

Where:
    glb_dir:    Directory containing *.glb files and object_transforms.json
    moge_npz:   MoGe intrinsics .npz from the full target image
    output_png: Output render path
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Euler, Vector


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        print("[ERROR] Usage: blender -b -P render_full_scene.py -- <glb_dir> <moge_npz> <output_png>")
        sys.exit(1)
    args = argv[argv.index("--") + 1:]
    if len(args) < 3:
        print("[ERROR] Need: glb_dir, moge_npz, output_png")
        sys.exit(1)
    return {
        "glb_dir": os.path.abspath(args[0]),
        "moge_npz": os.path.abspath(args[1]),
        "output_png": os.path.abspath(args[2]),
    }


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def setup_camera_opencv(fx, fy, cx, cy, img_w, img_h):
    """Camera at origin, looking along -Y_blender (= +Z_opencv forward).

    Uses Rx(-90) rotation. The resulting image is flipped vertically in
    post-processing to correct for OpenCV Y-down convention.
    """
    bpy.ops.object.camera_add()
    cam_obj = bpy.context.active_object
    cam_obj.name = "SceneCam"
    cam = cam_obj.data

    cam_obj.location = (0, 0, 0)
    cam_obj.rotation_euler = Euler((math.radians(-90), 0, 0), 'XYZ')

    cam.sensor_fit = 'HORIZONTAL'
    cam.sensor_width = float(img_w)
    cam.lens = fx

    cam.shift_x = (cx - img_w / 2.0) / img_w
    cam.shift_y = -(cy - img_h / 2.0) / img_h

    cam.clip_start = 0.001
    cam.clip_end = 1000.0

    bpy.context.scene.camera = cam_obj
    print(f"[INFO] Camera: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print(f"[INFO] Image: {img_w}x{img_h}, lens={cam.lens:.1f}mm")
    return cam_obj


def setup_lighting():
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is None:
        bg = world.node_tree.nodes.new(type='ShaderNodeBackground')
    bg.inputs['Strength'].default_value = 0.5
    bg.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    out = world.node_tree.nodes.get("World Output")
    if out:
        world.node_tree.links.new(bg.outputs['Background'], out.inputs['Surface'])

    # Key light — moderate to not wash out textures
    bpy.ops.object.light_add(type='SUN')
    sun = bpy.context.active_object
    sun.data.energy = 2.0
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))

    # Fill light — soft fill from opposite side
    bpy.ops.object.light_add(type='AREA', location=(2, 2, 2))
    area = bpy.context.active_object
    area.data.energy = 30.0
    area.data.size = 3.0


def import_glb(glb_path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=glb_path)
    after = set(bpy.data.objects)
    imported = list(after - before)
    return imported


def setup_render(width, height):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = True
    try:
        scene.eevee.taa_render_samples = 64
    except Exception:
        pass


def flip_image(path):
    """Flip rendered image vertically and horizontally.

    Vertical flip: OpenCV Y-down vs Blender camera Y-up.
    Horizontal flip: PyTorch3D X-left vs OpenCV/Blender X-right.
    """
    img = bpy.data.images.load(path)
    w, h = img.size
    pixels = list(img.pixels)
    px = 4  # RGBA channels per pixel
    stride = w * px
    flipped = []
    for row in range(h - 1, -1, -1):
        row_data = pixels[row * stride:(row + 1) * stride]
        reversed_row = []
        for col in range(w - 1, -1, -1):
            reversed_row.extend(row_data[col * px:(col + 1) * px])
        flipped.extend(reversed_row)
    img.pixels = flipped
    img.save_render(path)
    bpy.data.images.remove(img)


def main():
    args = parse_args()

    # Load MoGe intrinsics
    data = np.load(args["moge_npz"])
    intrinsics_px = data["intrinsics_px"]
    img_w = int(data["image_width"])
    img_h = int(data["image_height"])

    fx = float(intrinsics_px[0, 0])
    fy = float(intrinsics_px[1, 1])
    cx = float(intrinsics_px[0, 2])
    cy = float(intrinsics_px[1, 2])

    # Load object transforms
    transforms_path = os.path.join(args["glb_dir"], "object_transforms.json")
    with open(transforms_path, "r", encoding="utf-8") as f:
        transforms = json.load(f)

    print(f"[INFO] Loading {len(transforms)} objects from {args['glb_dir']}")

    clear_scene()
    setup_render(img_w, img_h)
    setup_lighting()
    setup_camera_opencv(fx, fy, cx, cy, img_w, img_h)

    # Import all GLBs
    # Support both list format [{glb_path:...}] and dict format {name: {glb_path:...}}
    if isinstance(transforms, dict):
        items = [(k, v) for k, v in transforms.items()
                 if isinstance(v, dict) and "glb_path" in v]
    else:
        items = [(t.get("object_name", str(i)), t) for i, t in enumerate(transforms)]

    for obj_name, t in items:
        glb_path = t["glb_path"]
        if not os.path.isabs(glb_path):
            glb_path = os.path.join(args["glb_dir"], os.path.basename(glb_path))

        if not os.path.exists(glb_path):
            print(f"[WARN] GLB not found: {glb_path}, skipping {obj_name}")
            continue

        imported = import_glb(glb_path)
        print(f"[INFO] Imported {obj_name}: {len(imported)} objects")

        # Print bounds for each mesh
        for obj in imported:
            if obj.type == 'MESH':
                verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
                if verts:
                    xs = [v.x for v in verts]
                    ys = [v.y for v in verts]
                    zs = [v.z for v in verts]
                    print(f"  {obj.name}: X=[{min(xs):.3f},{max(xs):.3f}] "
                          f"Y=[{min(ys):.3f},{max(ys):.3f}] Z=[{min(zs):.3f},{max(zs):.3f}]")

    # Render
    os.makedirs(os.path.dirname(args["output_png"]), exist_ok=True)
    bpy.context.scene.render.filepath = args["output_png"]
    bpy.ops.render.render(write_still=True)

    # Flip vertically for OpenCV convention
    flip_image(args["output_png"])
    print(f"[INFO] Full scene rendered: {args['output_png']}")


if __name__ == "__main__":
    main()
