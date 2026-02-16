"""Render SAM3D objects with corrected transforms.

The Transform3d in sam3d_worker.py has a bug: transform_points uses
`points_h @ self._matrix.T` (transpose), which reverses the application order
from the intended scale->rotate->translate to translate->rotate->scale.

This script corrects each object's vertices in-place:
  Buggy:   v_buggy   = S * R^T @ (v_pre + T)
  Correct: v_correct = S * R @ v_pre + T

Correction: v_correct = R^2 @ v_buggy + (T - S * R @ T)

After correction, all objects are in standard TRELLIS camera space:
  - Camera at origin (0, 0, 0)
  - Looking along +Z
  - +X = left, +Y = up (or vice versa - TBD)
"""

import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector


def parse_args():
    argv = sys.argv
    idx = argv.index("--")
    args = argv[idx + 1:]
    if len(args) < 2:
        print("[ERROR] Need transforms JSON and output PNG")
        sys.exit(1)

    result = {
        "transforms_path": os.path.abspath(args[0]),
        "output_png": os.path.abspath(args[1]),
        "lens": 35.0,
        "blend_path": None,
    }

    i = 2
    while i < len(args):
        if args[i] == "--lens" and i + 1 < len(args):
            result["lens"] = float(args[i + 1])
            i += 2
        elif args[i] == "--blend" and i + 1 < len(args):
            result["blend_path"] = os.path.abspath(args[i + 1])
            i += 2
        else:
            i += 1
    return result


def quaternion_to_matrix_np(q):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    two_s = 2.0 / (w*w + x*x + y*y + z*z)
    R = np.array([
        [1 - two_s * (y*y + z*z), two_s * (x*y - z*w),     two_s * (x*z + y*w)],
        [two_s * (x*y + z*w),     1 - two_s * (x*x + z*z), two_s * (y*z - x*w)],
        [two_s * (x*z - y*w),     two_s * (y*z + x*w),     1 - two_s * (x*x + y*y)],
    ])
    return R


def correct_object_vertices(obj, R_np, T_np, S_np):
    """Apply correction to undo Transform3d bug and apply correct transform.

    Buggy:   v_buggy = S * R^T @ (v_pre + T)
    Correct: v_correct = S * R @ v_pre + T
    Derivation: v_correct = R^2 @ v_buggy + (T - S * R @ T)

    Additionally, TRELLIS uses -X = screen-right (mirrored). We negate X
    so that +X = screen-right (standard convention).

    Final: v_final = mirror_x @ (R^2 @ v_buggy + (T - S * R @ T))
    """
    R2 = R_np @ R_np  # R squared
    S = S_np[0]  # Uniform scale
    offset = T_np - S * (R_np @ T_np)

    mesh = obj.data
    world_mat = np.array(obj.matrix_world)[:3, :3]
    world_loc = np.array(obj.matrix_world.translation)

    for v in mesh.vertices:
        # Get world-space coordinate
        co = np.array(obj.matrix_world @ v.co)
        # Apply correction (no X-mirror; camera (0,π,0) already maps screen-right to -X)
        co_corrected = R2 @ co + offset
        # Convert back to local space
        local_co = np.linalg.solve(world_mat, co_corrected - world_loc)
        v.co = local_co.tolist()

    mesh.update()


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def import_glb(glb_path, name_prefix=""):
    if not os.path.exists(glb_path):
        print(f"[WARN] Not found: {glb_path}")
        return None, []

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=glb_path)

    imported = bpy.context.selected_objects
    if not imported:
        return None, []

    root = None
    for obj in imported:
        if obj.parent not in imported:
            root = obj
            break
    if not root:
        root = imported[0]
    if name_prefix:
        root.name = name_prefix

    return root, imported


def setup_camera(lens_mm):
    """Camera at origin looking along +Z (TRELLIS camera convention).

    TRELLIS uses Y-down (OpenCV-like) convention:
      +Z = depth (into scene)
      +X = right in image
      +Y = down in image

    Blender camera: local -Z = forward, local +Y = up.
    We need camera -Z → world +Z (look forward) and camera +Y → world -Y
    (so screen-up = world -Y, matching TRELLIS Y-down).

    rotation_euler = (pi, 0, 0):
      Rotate pi around X: flips both Z and Y.
      Camera -Z → +Z (look into scene). Camera +Y → -Y (screen up = world -Y).
    """
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "Camera"
    camera.location = Vector((0.0, 0.0, 0.0))

    # Rotate pi around Y: look along +Z with +Y = up (OpenGL convention)
    camera.rotation_euler = (0.0, math.pi, 0.0)

    camera.data.lens = lens_mm
    camera.data.sensor_width = 36.0
    camera.data.clip_start = 0.01
    camera.data.clip_end = 100.0

    bpy.context.scene.camera = camera
    print(f"[INFO] Camera at origin, looking +Z, lens={lens_mm}mm")
    return camera


def setup_lighting():
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is None:
        bg = world.node_tree.nodes.new(type='ShaderNodeBackground')
    bg.inputs['Strength'].default_value = 0.8
    bg.inputs['Color'].default_value = (0.85, 0.85, 0.85, 1.0)
    wo = world.node_tree.nodes.get("World Output")
    if wo:
        world.node_tree.links.new(bg.outputs['Background'], wo.inputs['Surface'])

    bpy.ops.object.light_add(type='SUN')
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(45), 0, math.radians(30))

    bpy.ops.object.light_add(type='AREA')
    area = bpy.context.active_object
    area.name = "Fill"
    area.location = Vector((0, -1, 2))
    area.data.energy = 30.0
    area.data.size = 3.0


def setup_render(resolution=512):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.file_format = 'PNG'
    try:
        scene.eevee.taa_render_samples = 64
    except Exception:
        pass


def main():
    args = parse_args()

    with open(args["transforms_path"], 'r') as f:
        objects_data = json.load(f)

    clear_scene()
    setup_render()
    setup_lighting()

    transforms_dir = os.path.dirname(args["transforms_path"])

    for obj_data in objects_data:
        glb_path = obj_data.get("glb_path") or obj_data.get("glb")
        if not glb_path:
            continue
        if not os.path.isabs(glb_path):
            candidate = os.path.join(transforms_dir, os.path.basename(glb_path))
            if os.path.exists(candidate):
                glb_path = candidate

        name = os.path.splitext(os.path.basename(glb_path))[0]
        root, imported = import_glb(glb_path, name)
        if not root:
            continue

        # Get TRELLIS transform parameters
        T_np = np.array(obj_data["translation"])
        R_quat = np.array(obj_data["rotation"])  # (w, x, y, z)
        S_np = np.array(obj_data["scale"])
        R_np = quaternion_to_matrix_np(R_quat)

        # Apply correction to all mesh objects
        for obj in imported:
            if obj.type == 'MESH':
                correct_object_vertices(obj, R_np, T_np, S_np)
                print(f"[INFO] Corrected {obj.name}")

    # After correction, compute scene bounds
    min_co = Vector((float('inf'),) * 3)
    max_co = Vector((float('-inf'),) * 3)
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for v in obj.data.vertices:
                wc = obj.matrix_world @ v.co
                for i in range(3):
                    min_co[i] = min(min_co[i], wc[i])
                    max_co[i] = max(max_co[i], wc[i])

    center = (min_co + max_co) / 2
    print(f"[INFO] Corrected scene bounds: min=({min_co.x:.2f}, {min_co.y:.2f}, {min_co.z:.2f})")
    print(f"[INFO]                         max=({max_co.x:.2f}, {max_co.y:.2f}, {max_co.z:.2f})")
    print(f"[INFO] Center: ({center.x:.2f}, {center.y:.2f}, {center.z:.2f})")

    setup_camera(args["lens"])

    if args["blend_path"]:
        os.makedirs(os.path.dirname(args["blend_path"]), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=args["blend_path"])
        print(f"[INFO] Saved blend: {args['blend_path']}")

    os.makedirs(os.path.dirname(args["output_png"]), exist_ok=True)
    bpy.context.scene.render.filepath = args["output_png"]
    bpy.ops.render.render(write_still=True)
    print(f"[INFO] Rendered: {args['output_png']}")


if __name__ == "__main__":
    main()
