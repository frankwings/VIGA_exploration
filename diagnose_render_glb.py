"""Render a corrected GLB using MoGe camera intrinsics in Blender.

Usage:
    blender -b -P diagnose_render_glb.py -- <glb> <npz> <output_png>

The corrected GLB has vertices in OpenCV camera space:
  X_opencv = right, Y_opencv = down, Z_opencv = forward (+)

After Blender's glTF import (Y-up to Z-up conversion):
  X_b = X_opencv
  Y_b = -Z_opencv   (objects in front of camera -> negative Y_b)
  Z_b = Y_opencv

Camera setup:
  - Camera at origin
  - Rotate -90 degrees around X so local -Z (Blender forward) maps to world -Y
  - This makes the camera look along +Z_opencv (forward)
  - Camera up (local +Y) maps to world -Z_b = -Y_opencv = image up (correct for OpenCV)
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Euler, Vector


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        print("[ERROR] Usage: blender -b -P diagnose_render_glb.py -- <glb> <npz> <output_png>")
        sys.exit(1)
    args = argv[argv.index("--") + 1:]
    if len(args) < 3:
        print("[ERROR] Need: glb_path, npz_path, output_png")
        sys.exit(1)
    return {
        "glb_path": os.path.abspath(args[0]),
        "npz_path": os.path.abspath(args[1]),
        "output_png": os.path.abspath(args[2]),
    }


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def get_mesh_bounds():
    """Get bounding box of all mesh objects in Blender world space."""
    min_co = Vector((float('inf'),)*3)
    max_co = Vector((float('-inf'),)*3)
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for v in obj.data.vertices:
                wco = obj.matrix_world @ v.co
                for i in range(3):
                    min_co[i] = min(min_co[i], wco[i])
                    max_co[i] = max(max_co[i], wco[i])
    return min_co, max_co


def setup_camera_opencv(fx, fy, cx, cy, img_w, img_h):
    """Set up camera at origin matching OpenCV convention.

    After glTF import, OpenCV Z_forward maps to Blender -Y.
    Camera at origin rotated -90deg around X looks along -Y_b = +Z_opencv.
    """
    bpy.ops.object.camera_add()
    cam_obj = bpy.context.active_object
    cam_obj.name = "DiagCam"
    cam = cam_obj.data

    cam_obj.location = (0, 0, 0)
    cam_obj.rotation_euler = Euler((math.radians(-90), 0, 0), 'XYZ')
    bpy.context.view_layer.update()

    # Verify camera direction
    local_fwd = Vector((0, 0, -1))
    world_fwd = cam_obj.matrix_world.to_3x3() @ local_fwd
    local_up = Vector((0, 1, 0))
    world_up = cam_obj.matrix_world.to_3x3() @ local_up
    local_right = Vector((1, 0, 0))
    world_right = cam_obj.matrix_world.to_3x3() @ local_right
    print(f"[INFO] Camera forward (world): ({world_fwd[0]:.3f}, {world_fwd[1]:.3f}, {world_fwd[2]:.3f})")
    print(f"[INFO] Camera up     (world): ({world_up[0]:.3f}, {world_up[1]:.3f}, {world_up[2]:.3f})")
    print(f"[INFO] Camera right  (world): ({world_right[0]:.3f}, {world_right[1]:.3f}, {world_right[2]:.3f})")

    # Blender focal length: lens_mm = fx_pixels * sensor_width_mm / image_width_pixels
    # Set sensor_width = image_width so lens = fx
    cam.sensor_fit = 'HORIZONTAL'
    cam.sensor_width = float(img_w)
    cam.lens = fx

    # Principal point shift
    # Blender shift_x = (cx - img_w/2) / img_w  (fraction of image width)
    # Blender shift_y = -(cy - img_h/2) / img_h  (negative because Blender Y is up, OpenCV Y is down)
    cam.shift_x = (cx - img_w / 2.0) / img_w
    cam.shift_y = -(cy - img_h / 2.0) / img_h

    cam.clip_start = 0.001
    cam.clip_end = 1000.0

    bpy.context.scene.camera = cam_obj
    print(f"[INFO] Camera: fx={fx:.1f} fy={fy:.1f} sensor={cam.sensor_width:.0f}mm lens={cam.lens:.1f}mm")
    print(f"[INFO] Camera shift: x={cam.shift_x:.4f} y={cam.shift_y:.4f}")
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
    output_node = world.node_tree.nodes.get("World Output")
    if output_node:
        world.node_tree.links.new(bg.outputs['Background'], output_node.inputs['Surface'])

    bpy.ops.object.light_add(type='SUN')
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(40), 0, math.radians(20))


def import_glb(glb_path):
    if not os.path.exists(glb_path):
        print(f"[ERROR] GLB not found: {glb_path}")
        return None
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=glb_path)
    imported = bpy.context.selected_objects
    if not imported:
        print(f"[WARN] No objects from {glb_path}")
        return None

    for obj in imported:
        if obj.type == 'MESH':
            verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
            if verts:
                xs = [v.x for v in verts]
                ys = [v.y for v in verts]
                zs = [v.z for v in verts]
                print(f"[INFO] Mesh '{obj.name}' Blender bounds:")
                print(f"[INFO]   X: [{min(xs):.3f}, {max(xs):.3f}]")
                print(f"[INFO]   Y: [{min(ys):.3f}, {max(ys):.3f}]")
                print(f"[INFO]   Z: [{min(zs):.3f}, {max(zs):.3f}]")
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


def main():
    args = parse_args()

    data = np.load(args["npz_path"])
    intrinsics_px = data["intrinsics_px"]
    img_w = int(data["image_width"])
    img_h = int(data["image_height"])

    fx = float(intrinsics_px[0, 0])
    fy = float(intrinsics_px[1, 1])
    cx = float(intrinsics_px[0, 2])
    cy = float(intrinsics_px[1, 2])

    print(f"[INFO] Image: {img_w}x{img_h}")
    print(f"[INFO] Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    clear_scene()
    setup_render(img_w, img_h)
    setup_lighting()
    import_glb(args["glb_path"])

    min_co, max_co = get_mesh_bounds()
    center = (min_co + max_co) / 2
    print(f"[INFO] Mesh center (Blender): ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print(f"[INFO] Mesh bounds X: [{min_co[0]:.3f}, {max_co[0]:.3f}]")
    print(f"[INFO] Mesh bounds Y: [{min_co[1]:.3f}, {max_co[1]:.3f}]")
    print(f"[INFO] Mesh bounds Z: [{min_co[2]:.3f}, {max_co[2]:.3f}]")

    setup_camera_opencv(fx, fy, cx, cy, img_w, img_h)

    os.makedirs(os.path.dirname(args["output_png"]), exist_ok=True)
    bpy.context.scene.render.filepath = args["output_png"]
    bpy.ops.render.render(write_still=True)

    # Flip vertically: OpenCV Y-down vs Blender camera Y-up causes vertical inversion
    flip_image(args["output_png"])
    print(f"[INFO] Rendered (flipped): {args['output_png']}")


def flip_image(path):
    """Flip rendered image vertically and horizontally.

    Vertical flip: OpenCV Y-down vs Blender camera Y-up.
    Horizontal flip: PyTorch3D X-left vs OpenCV/Blender X-right.
    """
    img = bpy.data.images.load(path)
    w, h = img.size
    pixels = list(img.pixels)  # flat RGBA list
    px = 4  # RGBA channels per pixel
    stride = w * px
    flipped = []
    for row in range(h - 1, -1, -1):
        row_data = pixels[row * stride:(row + 1) * stride]
        # Reverse pixel order within the row (horizontal flip)
        reversed_row = []
        for col in range(w - 1, -1, -1):
            reversed_row.extend(row_data[col * px:(col + 1) * px])
        flipped.extend(reversed_row)
    img.pixels = flipped
    img.save_render(path)
    bpy.data.images.remove(img)


if __name__ == "__main__":
    main()
