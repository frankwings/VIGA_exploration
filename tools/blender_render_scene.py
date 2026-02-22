"""Render all aligned GLBs in a composed Blender scene with MoGe camera intrinsics.

Reads intrinsics directly from object_transforms.json (normalized format).
The aligned GLBs are already in camera space (Y-up PyTorch3D convention).

Usage:
    blender -b -P tools/blender_render_scene.py -- <data_dir> <output_png> [--width W] [--height H]

Where:
    data_dir:    Directory containing *_pbr_aligned.glb and object_transforms.json
    output_png:  Output render path
    --width W:   Override render width (default: pointmap_shape width)
    --height H:  Override render height (default: pointmap_shape height)
"""
import json
import math
import os
import sys

import bpy
from mathutils import Euler


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        print("[ERROR] Usage: blender -b -P tools/blender_render_scene.py -- <data_dir> <output_png> [--width W] [--height H]")
        sys.exit(1)
    args = argv[argv.index("--") + 1:]
    if len(args) < 2:
        print("[ERROR] Need: data_dir output_png")
        sys.exit(1)

    result = {
        "data_dir": os.path.abspath(args[0]),
        "output_png": os.path.abspath(args[1]),
        "width": None,
        "height": None,
    }

    i = 2
    while i < len(args):
        if args[i] == "--width" and i + 1 < len(args):
            result["width"] = int(args[i + 1])
            i += 2
        elif args[i] == "--height" and i + 1 < len(args):
            result["height"] = int(args[i + 1])
            i += 2
        else:
            i += 1

    return result


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def setup_camera_opencv(fx, fy, cx, cy, img_w, img_h):
    """Camera at origin, looking along -Y_blender (= +Z_opencv forward).

    Uses Rx(90) + Rz(180) rotation so that:
      - Camera forward  = -Y_blender  (= +Z in PyTorch3D camera space)
      - Camera up       = +Z_blender  (= +Y in PyTorch3D camera space)
      - Camera right    = -X_blender  (= -X_pt3d = physical right)

    No post-render flip is needed with this rotation.
    """
    bpy.ops.object.camera_add()
    cam_obj = bpy.context.active_object
    cam_obj.name = "SceneCam"
    cam = cam_obj.data

    cam_obj.location = (0, 0, 0)
    cam_obj.rotation_euler = Euler((math.radians(90), 0, math.radians(180)), 'XYZ')

    cam.sensor_fit = 'HORIZONTAL'
    cam.sensor_width = float(img_w)
    cam.lens = fx

    cam.shift_x = -(cx - img_w / 2.0) / img_w
    cam.shift_y = (cy - img_h / 2.0) / img_h

    cam.clip_start = 0.001
    cam.clip_end = 1000.0

    bpy.context.scene.camera = cam_obj
    print(f"[SCENE] Camera: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print(f"[SCENE] Image: {img_w}x{img_h}, lens={cam.lens:.1f}mm")
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
    bg.inputs['Strength'].default_value = 1.0
    bg.inputs['Color'].default_value = (0.85, 0.85, 0.85, 1.0)
    out = world.node_tree.nodes.get("World Output")
    if out:
        world.node_tree.links.new(bg.outputs['Background'], out.inputs['Surface'])

    # Key light
    bpy.ops.object.light_add(type='SUN')
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(2, 2, 2))
    area = bpy.context.active_object
    area.data.energy = 50.0
    area.data.size = 3.0


def import_glb(glb_path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=glb_path)
    after = set(bpy.data.objects)
    return list(after - before)


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


def main():
    args = parse_args()

    # Load object transforms
    transforms_path = os.path.join(args["data_dir"], "object_transforms.json")
    with open(transforms_path, "r", encoding="utf-8") as f:
        transforms = json.load(f)

    if not transforms:
        print("[ERROR] No objects in object_transforms.json")
        sys.exit(1)

    # Extract camera intrinsics from first object (all objects share same camera)
    K = transforms[0]["intrinsics"]       # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    pm_shape = transforms[0]["pointmap_shape"]  # [H, W]
    pm_h, pm_w = pm_shape

    # Convert normalized intrinsics to pixel values
    fx_px = K[0][0] * pm_w
    fy_px = K[1][1] * pm_h
    cx_px = K[0][2] * pm_w
    cy_px = K[1][2] * pm_h

    # Render resolution: default to pointmap shape, allow override
    render_w = args["width"] or pm_w
    render_h = args["height"] or pm_h

    # Scale intrinsics if render resolution differs from pointmap shape
    if render_w != pm_w or render_h != pm_h:
        scale_x = render_w / pm_w
        scale_y = render_h / pm_h
        fx_px *= scale_x
        fy_px *= scale_y
        cx_px *= scale_x
        cy_px *= scale_y

    print(f"[SCENE] Loading {len(transforms)} objects from {args['data_dir']}")
    print(f"[SCENE] Pointmap: {pm_w}x{pm_h}, Render: {render_w}x{render_h}")

    clear_scene()
    setup_render(render_w, render_h)
    setup_lighting()
    setup_camera_opencv(fx_px, fy_px, cx_px, cy_px, render_w, render_h)

    # Import all aligned GLBs
    data_dir = args["data_dir"]
    for t in transforms:
        name = t.get("object_name", "unknown")

        # Prefer PBR-aligned > plain aligned
        candidates = [
            os.path.join(data_dir, f"{name}_pbr_aligned.glb"),
            os.path.join(data_dir, f"{name}.glb"),
        ]
        glb_path = None
        for c in candidates:
            if os.path.exists(c):
                glb_path = c
                break

        if glb_path is None:
            # Try glb_path from transforms (may be absolute VM path)
            fallback = t.get("glb_path", "")
            if os.path.exists(fallback):
                glb_path = fallback

        if glb_path is None:
            print(f"[WARN] No GLB for {name}, skipping")
            continue

        imported = import_glb(glb_path)
        print(f"[SCENE] Imported {name}: {len(imported)} objects from {os.path.basename(glb_path)}")

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
    os.makedirs(os.path.dirname(args["output_png"]) or ".", exist_ok=True)
    bpy.context.scene.render.filepath = args["output_png"]
    bpy.ops.render.render(write_still=True)

    print(f"[SCENE] Composed scene rendered: {args['output_png']}")


if __name__ == "__main__":
    main()
