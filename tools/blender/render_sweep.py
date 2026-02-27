"""Sweep camera orientations to find the correct view matching target image."""

import json
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv
idx = argv.index("--")
transforms_path = os.path.abspath(argv[idx + 1])
output_dir = os.path.abspath(argv[idx + 2])

# Clear and import
bpy.ops.wm.read_factory_settings(use_empty=True)
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

with open(transforms_path, 'r') as f:
    objects_data = json.load(f)

transforms_dir = os.path.dirname(transforms_path)
for obj_data in objects_data:
    glb_path = obj_data.get("glb_path") or obj_data.get("glb")
    if not glb_path:
        continue
    if not os.path.isabs(glb_path):
        candidate = os.path.join(transforms_dir, os.path.basename(glb_path))
        if os.path.exists(candidate):
            glb_path = candidate
    if not os.path.exists(glb_path):
        continue
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=glb_path)

# Get bounds
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
size = max_co - min_co

# World
world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
bg.inputs['Strength'].default_value = 1.0
bg.inputs['Color'].default_value = (0.9, 0.9, 0.9, 1.0)
wo = world.node_tree.nodes.get("World Output")
if wo:
    world.node_tree.links.new(bg.outputs['Background'], wo.inputs['Surface'])
bpy.ops.object.light_add(type='SUN')
bpy.context.active_object.data.energy = 3.0

# Render settings
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.image_settings.color_mode = 'RGB'
os.makedirs(output_dir, exist_ok=True)

# The green_tea_bottle (foreground) is at Y=2.909, headphones (background) at Y=0.430
# Camera should be at high Y looking -Y. Try various up vectors and tilts.

# Distance from scene max_y
cam_dist = 3.0

views = []

# Looking from high Y toward -Y with different up vectors
for up_label, up_vec in [("Xup", 'X'), ("nXup", '-X'), ("Zup", 'Z'), ("nZup", '-Z')]:
    # Camera position beyond max Y
    cam_pos = Vector((center.x, max_co.y + cam_dist, center.z))
    look_at = center
    look_dir = look_at - cam_pos

    # Handle the up vector mapping for to_track_quat
    # '-Z' is camera forward, up_vec is the up hint
    # But to_track_quat only accepts 'X','Y','Z' not negatives
    # For negative, we'll compute manually
    if up_vec.startswith('-'):
        # Use the positive axis but flip the camera
        actual_up = up_vec[1]
    else:
        actual_up = up_vec

    views.append((f"highY_{up_label}", cam_pos, look_dir, actual_up, up_vec.startswith('-')))

# Also try looking from high Y with slight downward tilt
for angle_deg in [0, 15, 30]:
    cam_pos = Vector((center.x, max_co.y + cam_dist, center.z))
    # Tilt: move camera up in X (if X is "up") and adjust look point down
    tilt_offset = cam_dist * math.tan(math.radians(angle_deg))
    for up_label, up_axis in [("Xup", 0), ("nXup", 0), ("Zup", 2), ("nZup", 2)]:
        cam_tilted = Vector(cam_pos)
        if up_label.startswith('n'):
            cam_tilted[up_axis] -= tilt_offset
        else:
            cam_tilted[up_axis] += tilt_offset
        look_dir = center - cam_tilted
        is_neg = up_label.startswith('n')
        actual_up = up_label[-3] if not is_neg else up_label[-3]
        # Just use Z for tilt tests
        if angle_deg > 0:
            views.append((f"highY_{up_label}_tilt{angle_deg}", cam_tilted, look_dir, actual_up, is_neg))

for label, cam_pos, look_dir, up_hint, flip_up in views:
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "Camera"
    camera.location = cam_pos
    camera.rotation_euler = look_dir.to_track_quat('-Z', up_hint).to_euler('XYZ')

    # If flip_up, rotate 180 around look direction
    if flip_up:
        import mathutils
        from mathutils import Quaternion
        q = look_dir.to_track_quat('-Z', up_hint)
        flip_q = Quaternion(look_dir.normalized(), math.pi)
        q = flip_q @ q
        camera.rotation_euler = q.to_euler('XYZ')

    camera.data.lens = 25
    camera.data.sensor_width = 36.0
    camera.data.clip_start = 0.01
    camera.data.clip_end = 100.0
    scene.camera = camera

    out_path = os.path.join(output_dir, f"{label}.png")
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print(f"[DONE] {label}")

print("[ALL DONE]")
