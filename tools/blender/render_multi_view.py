"""Render scene from 6 cardinal directions to find the correct camera orientation."""

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
    if not glb_path or os.path.isabs(glb_path):
        pass
    else:
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
max_dim = max(size.x, size.y, size.z)

# Setup world
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
sun = bpy.context.active_object
sun.data.energy = 3.0

# Render settings
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.image_settings.color_mode = 'RGB'

os.makedirs(output_dir, exist_ok=True)

# 6 viewing directions + 2 diagonal
views = [
    ("+X", Vector((1, 0, 0)), 'Z'),   # Looking from +X toward -X
    ("-X", Vector((-1, 0, 0)), 'Z'),   # Looking from -X toward +X
    ("+Y", Vector((0, 1, 0)), 'Z'),    # Looking from +Y toward -Y
    ("-Y", Vector((0, -1, 0)), 'Z'),   # Looking from -Y toward +Y
    ("+Z", Vector((0, 0, 1)), 'Y'),    # Looking from +Z toward -Z
    ("-Z", Vector((0, 0, -1)), 'Y'),   # Looking from -Z toward +Z
    # Also try +Y with X-up (rotated 90)
    ("+Y_Xup", Vector((0, 1, 0)), 'X'),
    ("-Y_Xup", Vector((0, -1, 0)), 'X'),
]

for label, direction, up in views:
    # Camera position: center + offset along direction
    cam_pos = center + direction * (max_dim * 1.2)

    # Remove old camera
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "Camera"
    camera.location = cam_pos

    # Look toward center
    look_dir = center - cam_pos
    camera.rotation_euler = look_dir.to_track_quat('-Z', up).to_euler('XYZ')

    camera.data.lens = 25
    camera.data.sensor_width = 36.0
    camera.data.clip_start = 0.01
    camera.data.clip_end = 100.0
    scene.camera = camera

    # Render
    out_path = os.path.join(output_dir, f"view_{label}.png")
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print(f"[DONE] {label} -> {out_path}")

print("[ALL DONE]")
