import bpy
import mathutils
import math
from pathlib import Path

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def load_glb(glb_path):
    bpy.ops.import_scene.gltf(filepath=str(glb_path))

def setup_render():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.cycles.samples = 64

def render_single_frame(output_path):
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)

# Render keyboard
clear_scene()
load_glb("output/sam3d_reconstruction_batch/alienware_keyboard.glb")
setup_render()

# Add lighting
bpy.ops.object.light_add(type='SUN', location=(2, 2, 5))
bpy.ops.object.light_add(type='AREA', location=(-2, -2, 3))

# Add camera
bpy.ops.object.camera_add(location=(3, -3, 2))
camera = bpy.context.object
bpy.context.scene.camera = camera

# Point camera at center
camera.rotation_euler = (math.radians(60), 0, math.radians(45))

# Render
render_single_frame("output/sam3d_reconstruction_batch/keyboard_render.png")

print("Keyboard render complete!")