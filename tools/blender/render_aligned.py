"""Render SAM3D-reconstructed objects aligned to the original camera viewpoint.

The GLB files from SAM3D have vertices already transformed into camera space
by transform_mesh_vertices() in sam3d_worker.py. Based on actual vertex analysis,
the scene coordinate frame after those transforms is:
  - +Y = depth (into the scene, away from camera)
  - +X/Z = lateral spread

This script imports all GLBs and automatically determines camera placement
by analyzing the actual vertex positions.

Usage:
    blender -b -P render_aligned.py -- <transforms.json> <output.png> [options]

Options:
    --lens N          Camera focal length in mm (default: auto-fit)
    --blend path      Save .blend file
    --cam-back N      Camera distance behind nearest object (default: 0.5)
"""

import json
import math
import os
import sys

import bpy
from mathutils import Euler, Vector


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        print("[ERROR] Usage: blender -b -P render_aligned.py -- transforms.json output.png [--lens N] [--blend out.blend]")
        sys.exit(1)
    idx = argv.index("--")
    args = argv[idx + 1:]

    if len(args) < 2:
        print("[ERROR] Need transforms JSON and output PNG path")
        sys.exit(1)

    result = {
        "transforms_path": os.path.abspath(args[0]),
        "output_png": os.path.abspath(args[1]),
        "lens": None,  # None = auto-fit
        "blend_path": None,
        "cam_back": 0.5,
    }

    i = 2
    while i < len(args):
        if args[i] == "--lens" and i + 1 < len(args):
            result["lens"] = float(args[i + 1])
            i += 2
        elif args[i] == "--blend" and i + 1 < len(args):
            result["blend_path"] = os.path.abspath(args[i + 1])
            i += 2
        elif args[i] == "--cam-back" and i + 1 < len(args):
            result["cam_back"] = float(args[i + 1])
            i += 2
        else:
            i += 1

    return result


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def get_scene_bounds():
    """Compute bounding box of all mesh objects in the scene."""
    min_co = Vector((float('inf'), float('inf'), float('inf')))
    max_co = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for v in obj.data.vertices:
                world_co = obj.matrix_world @ v.co
                for i in range(3):
                    min_co[i] = min(min_co[i], world_co[i])
                    max_co[i] = max(max_co[i], world_co[i])

    return min_co, max_co


def setup_camera_auto(min_co, max_co, cam_back=0.5, lens_mm=None):
    """Auto-place camera to frame all objects.

    Analyzes the scene bounding box to determine the primary depth axis
    and places the camera accordingly.
    """
    center = (min_co + max_co) / 2
    size = max_co - min_co

    # Determine primary depth axis (the one with largest spread)
    extents = [size.x, size.y, size.z]
    depth_axis = extents.index(max(extents))

    print(f"[INFO] Scene bounds: min=({min_co.x:.2f}, {min_co.y:.2f}, {min_co.z:.2f})")
    print(f"[INFO]               max=({max_co.x:.2f}, {max_co.y:.2f}, {max_co.z:.2f})")
    print(f"[INFO] Scene center: ({center.x:.2f}, {center.y:.2f}, {center.z:.2f})")
    print(f"[INFO] Scene size:   ({size.x:.2f}, {size.y:.2f}, {size.z:.2f})")
    print(f"[INFO] Depth axis: {'XYZ'[depth_axis]} (spread={extents[depth_axis]:.2f})")

    # Camera at min of depth axis, looking toward max
    cam_pos = Vector(center)
    cam_pos[depth_axis] = min_co[depth_axis] - cam_back

    # The two lateral axes determine the field of view needed
    lateral_axes = [i for i in range(3) if i != depth_axis]
    lateral_extent = max(size[lateral_axes[0]], size[lateral_axes[1]])

    # Distance from camera to scene center along depth
    depth_to_center = center[depth_axis] - cam_pos[depth_axis]

    # Auto-fit lens to frame the scene with some margin
    if lens_mm is None:
        sensor_width = 36.0
        # Half-angle needed: atan(half_extent / depth_to_nearest)
        depth_to_nearest = min_co[depth_axis] - cam_pos[depth_axis]
        if depth_to_nearest < 0.1:
            depth_to_nearest = 0.1
        half_angle = math.atan(lateral_extent * 0.65 / depth_to_nearest)
        lens_mm = (sensor_width / 2) / math.tan(half_angle)
        lens_mm = max(10, min(lens_mm, 200))  # Clamp to reasonable range
        print(f"[INFO] Auto-fit lens: {lens_mm:.1f}mm (half_angle={math.degrees(half_angle):.1f}°)")

    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "Camera"
    camera.location = cam_pos

    # Point camera along the depth axis (from min toward max)
    look_dir = Vector((0, 0, 0))
    look_dir[depth_axis] = 1.0  # Look toward positive depth

    # Determine up vector (use Z-up if depth isn't Z, else use Y-up)
    if depth_axis == 2:  # Z is depth
        up_hint = 'Y'
    elif depth_axis == 1:  # Y is depth
        up_hint = 'Z'
    else:  # X is depth
        up_hint = 'Z'

    camera.rotation_euler = look_dir.to_track_quat('-Z', up_hint).to_euler('XYZ')

    camera.data.lens = lens_mm
    camera.data.sensor_width = 36.0
    camera.data.clip_start = 0.01
    camera.data.clip_end = 100.0

    bpy.context.scene.camera = camera
    print(f"[INFO] Camera at ({cam_pos.x:.2f}, {cam_pos.y:.2f}, {cam_pos.z:.2f})")
    print(f"[INFO] Looking along {'XYZ'[depth_axis]}+, up={up_hint}, lens={lens_mm:.1f}mm")
    return camera


def setup_lighting(scene_center):
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is None:
        bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
    bg_node.inputs['Strength'].default_value = 0.8
    bg_node.inputs['Color'].default_value = (0.9, 0.9, 0.9, 1.0)

    output_node = world.node_tree.nodes.get("World Output")
    if output_node:
        world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    # Key light - sun from above and slightly behind camera
    bpy.ops.object.light_add(type='SUN')
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 3.0
    sun.rotation_euler = Euler((math.radians(50), 0, math.radians(30)))

    # Area light near scene center for fill
    bpy.ops.object.light_add(type='AREA')
    area = bpy.context.active_object
    area.name = "Fill"
    area.location = scene_center + Vector((0, -1.5, 2.0))
    area.data.energy = 50.0
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


def import_glb(glb_path, name_prefix=""):
    """Import GLB without modifying vertex positions or origins."""
    if not os.path.exists(glb_path):
        print(f"[WARN] Not found: {glb_path}")
        return None

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=glb_path)

    imported = bpy.context.selected_objects
    if not imported:
        print(f"[WARN] No objects from {glb_path}")
        return None

    root = None
    for obj in imported:
        if obj.parent not in imported:
            root = obj
            break
    if not root:
        root = imported[0]

    if name_prefix:
        root.name = name_prefix

    print(f"[INFO] Imported {name_prefix}: {len(imported)} objects")
    return root


def main():
    args = parse_args()

    with open(args["transforms_path"], 'r') as f:
        objects_data = json.load(f)

    print(f"[INFO] Loading {len(objects_data)} objects")

    clear_scene()
    setup_render()

    # Import all GLBs first
    transforms_dir = os.path.dirname(args["transforms_path"])

    for obj_data in objects_data:
        glb_path = obj_data.get("glb_path") or obj_data.get("glb")
        if not glb_path:
            continue

        if not os.path.isabs(glb_path):
            candidate = os.path.join(transforms_dir, os.path.basename(glb_path))
            if os.path.exists(candidate):
                glb_path = candidate
            else:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                candidate = os.path.join(project_root, glb_path)
                if os.path.exists(candidate):
                    glb_path = candidate

        name = os.path.splitext(os.path.basename(glb_path))[0]
        import_glb(glb_path, name)

    # Analyze actual vertex positions and auto-place camera
    min_co, max_co = get_scene_bounds()
    center = (min_co + max_co) / 2

    setup_lighting(center)
    setup_camera_auto(min_co, max_co, cam_back=args["cam_back"], lens_mm=args["lens"])

    # Save blend if requested
    if args["blend_path"]:
        os.makedirs(os.path.dirname(args["blend_path"]), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=args["blend_path"])
        print(f"[INFO] Saved blend: {args['blend_path']}")

    # Render
    os.makedirs(os.path.dirname(args["output_png"]), exist_ok=True)
    bpy.context.scene.render.filepath = args["output_png"]
    bpy.ops.render.render(write_still=True)
    print(f"[INFO] Rendered: {args['output_png']}")


if __name__ == "__main__":
    main()
