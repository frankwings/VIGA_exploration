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
    scene.cycles.samples = 32  # Lower quality for speed

def render_single_frame(output_path):
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    
# Test render
clear_scene()
load_glb("output/sam3d_reconstruction_batch/green_tea_bottle.glb")
setup_render()

# Add simple lighting
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))

# Add camera
bpy.ops.object.camera_add(location=(0, -5, 2))
camera = bpy.context.object
bpy.context.scene.camera = camera

# Point camera at center
camera.rotation_euler = (math.radians(70), 0, 0)

# Render single test frame
render_single_frame("output/sam3d_reconstruction_batch/test_render.png")

print("Test render complete!")