"""Blender script to render GLB file."""
import bpy
import mathutils
import sys
import os
import math

# Get arguments after --
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

glb_path = argv[0] if len(argv) > 0 else "output/sam3d_test/green_tea_bottle.glb"
output_path = argv[1] if len(argv) > 1 else "docs/test_results_images/green_tea_bottle_render.png"

print(f"GLB: {glb_path}")
print(f"Output: {output_path}")

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLB
bpy.ops.import_scene.gltf(filepath=glb_path)

# Get imported objects
imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
print(f"Imported {len(imported_objects)} mesh objects")

# Calculate bounding box for all objects
min_co = [float('inf')] * 3
max_co = [float('-inf')] * 3

for obj in imported_objects:
    for v in obj.bound_box:
        world_v = obj.matrix_world @ bpy.data.objects[obj.name].data.vertices[0].co if obj.data.vertices else obj.matrix_world @ bpy.mathutils.Vector(v)
        for i in range(3):
            min_co[i] = min(min_co[i], v[i])
            max_co[i] = max(max_co[i], v[i])

# Center point
center = [(min_co[i] + max_co[i]) / 2 for i in range(3)]
size = max([max_co[i] - min_co[i] for i in range(3)])

print(f"Object center: {center}, size: {size}")

# Add camera
bpy.ops.object.camera_add(location=(center[0] + size * 1.5, center[1] - size * 1.5, center[2] + size * 0.8))
camera = bpy.context.object
camera.name = "RenderCamera"

# Point camera at center
direction = mathutils.Vector(center) - camera.location
rot_quat = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rot_quat.to_euler()

bpy.context.scene.camera = camera

# Add lights
# Key light
bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
key_light = bpy.context.object
key_light.data.energy = 3

# Fill light
bpy.ops.object.light_add(type='SUN', location=(-5, -3, 5))
fill_light = bpy.context.object
fill_light.data.energy = 1.5

# Back light
bpy.ops.object.light_add(type='SUN', location=(0, 5, 8))
back_light = bpy.context.object
back_light.data.energy = 1

# Set render settings
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.cycles.samples = 128
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.render.film_transparent = True
scene.render.filepath = output_path
scene.render.image_settings.file_format = 'PNG'

# Enable GPU
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.get_devices()
for device in prefs.devices:
    device.use = True

print("Rendering...")
bpy.ops.render.render(write_still=True)
print(f"Saved to: {output_path}")
