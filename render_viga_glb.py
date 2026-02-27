"""Blender script to render VIGA GLB with transforms."""
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

glb_path = argv[0] if len(argv) > 0 else "output/viga_test/green_tea_bottle_viga.glb"
output_path = argv[1] if len(argv) > 1 else "docs/test_results_images/green_tea_bottle_viga_render.png"

# VIGA output transforms
translation = [0.06601577252149582, -0.4401596486568451, 1.4577733278274536]
rotation_quat = [6.686314736725762e-05, -0.018951283767819405, -0.9978983402252197, -0.06196640804409981]
scale_val = 2.312847852706909

print(f"GLB: {glb_path}")
print(f"Output: {output_path}")
print(f"Translation: {translation}")
print(f"Rotation: {rotation_quat}")
print(f"Scale: {scale_val}")

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import GLB
bpy.ops.import_scene.gltf(filepath=glb_path)

# Get imported mesh objects
imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
print(f"Imported {len(imported_objects)} mesh objects")

# Apply transforms to imported objects
for obj in imported_objects:
    # Note: VIGA already applied transforms to vertices, so we just need proper camera setup
    pass

# Calculate bounding box
min_co = [float('inf')] * 3
max_co = [float('-inf')] * 3
for obj in imported_objects:
    for v in obj.bound_box:
        for i in range(3):
            min_co[i] = min(min_co[i], v[i])
            max_co[i] = max(max_co[i], v[i])

center = [(min_co[i] + max_co[i]) / 2 for i in range(3)]
size = max([max_co[i] - min_co[i] for i in range(3)])
print(f"Bounding box center: {center}, size: {size}")

# Add camera - position it to see the object well
cam_distance = size * 2.5
bpy.ops.object.camera_add(
    location=(center[0] + cam_distance * 0.7, center[1] - cam_distance * 0.7, center[2] + cam_distance * 0.5)
)
camera = bpy.context.object
camera.name = "RenderCamera"

# Point camera at center
direction = mathutils.Vector(center) - camera.location
rot_quat = direction.to_track_quat('-Z', 'Y')
camera.rotation_euler = rot_quat.to_euler()

bpy.context.scene.camera = camera

# Add lights
# Key light (main)
bpy.ops.object.light_add(type='AREA', location=(center[0] + size*2, center[1] - size*2, center[2] + size*3))
key_light = bpy.context.object
key_light.data.energy = 500
key_light.data.size = size * 2

# Fill light
bpy.ops.object.light_add(type='AREA', location=(center[0] - size*2, center[1] - size, center[2] + size*2))
fill_light = bpy.context.object
fill_light.data.energy = 200
fill_light.data.size = size * 2

# Back/rim light
bpy.ops.object.light_add(type='AREA', location=(center[0], center[1] + size*2, center[2] + size*2))
back_light = bpy.context.object
back_light.data.energy = 300
back_light.data.size = size

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
