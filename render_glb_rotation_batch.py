"""Blender script: render rotation GIFs for all GLBs in a directory.

Usage:
    blender --background --python render_glb_rotation_batch.py -- \
        --input-dir <dir_with_glbs> --output-dir <dir_for_outputs> \
        [--num-frames 36] [--resolution 512]

Renders each GLB as a 360-degree camera orbit, saves PNGs to subfolders.
After Blender finishes, run the GIF assembly separately with PIL.
"""
import bpy
import os
import sys
import math
import glob
import argparse
from mathutils import Vector


def parse_args():
    # Blender passes everything after '--' to sys.argv
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing .glb files")
    parser.add_argument("--output-dir", required=True, help="Output directory for frames")
    parser.add_argument("--num-frames", type=int, default=36)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    for m in list(bpy.data.lights):
        bpy.data.lights.remove(m)
    for m in list(bpy.data.cameras):
        bpy.data.cameras.remove(m)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img)


def get_all_mesh_bounds(objs):
    min_v = Vector((float('inf'), float('inf'), float('inf')))
    max_v = Vector((float('-inf'), float('-inf'), float('-inf')))
    found = False
    for obj in objs:
        if obj.type != 'MESH':
            continue
        for vert in obj.data.vertices:
            wc = obj.matrix_world @ vert.co
            min_v.x = min(min_v.x, wc.x)
            min_v.y = min(min_v.y, wc.y)
            min_v.z = min(min_v.z, wc.z)
            max_v.x = max(max_v.x, wc.x)
            max_v.y = max(max_v.y, wc.y)
            max_v.z = max(max_v.z, wc.z)
            found = True
    if not found:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    return min_v, max_v


def setup_lighting():
    light_data = bpy.data.lights.new('Key', type='AREA')
    light_data.energy = 300
    light_data.size = 3.0
    light_obj = bpy.data.objects.new('Key', light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (-2, -3, 4)
    light_obj.rotation_euler = (math.radians(50), 0, math.radians(-30))

    light_data2 = bpy.data.lights.new('Fill', type='AREA')
    light_data2.energy = 150
    light_data2.size = 2.0
    light_obj2 = bpy.data.objects.new('Fill', light_data2)
    bpy.context.collection.objects.link(light_obj2)
    light_obj2.location = (3, -2, 3)
    light_obj2.rotation_euler = (math.radians(50), 0, math.radians(30))

    world = bpy.data.worlds.get('World')
    if not world:
        world = bpy.data.worlds.new('World')
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Color'].default_value = (0.15, 0.15, 0.17, 1.0)
        bg.inputs['Strength'].default_value = 0.3


def add_ground(center_z_min):
    bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, center_z_min))
    plane = bpy.context.active_object
    mat = bpy.data.materials.new('Ground')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.35, 0.33, 0.30, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.85
    plane.data.materials.append(mat)


def render_rotation(cam_obj, center, dist, elevation_rad, frames_dir, num_frames):
    for i in range(num_frames):
        angle = 2.0 * math.pi * i / num_frames
        cam_obj.location = (
            center.x + dist * math.sin(angle) * math.cos(elevation_rad),
            center.y - dist * math.cos(angle) * math.cos(elevation_rad),
            center.z + dist * math.sin(elevation_rad)
        )
        direction = center - cam_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()

        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        bpy.context.scene.render.filepath = frame_path
        bpy.ops.render.render(write_still=True)
    print(f"  Rendered {num_frames} frames to {frames_dir}")


args = parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
num_frames = args.num_frames
resolution = args.resolution

os.makedirs(output_dir, exist_ok=True)

# Render settings (EEVEE for speed)
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
bpy.context.scene.render.resolution_x = resolution
bpy.context.scene.render.resolution_y = resolution
bpy.context.scene.render.image_settings.file_format = 'PNG'

# Find all GLBs
glb_files = sorted(glob.glob(os.path.join(input_dir, "*.glb")))
print(f"\nFound {len(glb_files)} GLB files in {input_dir}")

for glb_path in glb_files:
    name = os.path.splitext(os.path.basename(glb_path))[0]
    frames_dir = os.path.join(output_dir, f"{name}_frames")

    # Skip if already rendered
    if os.path.exists(frames_dir) and len(glob.glob(os.path.join(frames_dir, "*.png"))) >= num_frames:
        print(f"SKIP (already rendered): {name}")
        continue

    clear_scene()

    # Import GLB
    pre = set(o.name for o in bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=glb_path)
    post = set(o.name for o in bpy.data.objects)
    new_objs = [bpy.data.objects[n] for n in (post - pre)]

    bpy.context.view_layer.update()
    min_v, max_v = get_all_mesh_bounds(new_objs)
    dims = max_v - min_v
    max_dim = max(dims.x, dims.y, dims.z)

    print(f"\n{'='*60}")
    print(f"GLB: {name} (max_dim={max_dim:.4f})")
    print(f"{'='*60}")

    center = (min_v + max_v) / 2.0
    dist = max(max_dim * 2.5, 0.5)
    elevation_rad = math.radians(25)

    add_ground(min_v.z)
    setup_lighting()

    cam_data = bpy.data.cameras.new('Camera')
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    os.makedirs(frames_dir, exist_ok=True)
    render_rotation(cam_obj, center, dist, elevation_rad, frames_dir, num_frames)

print("\nDONE — all rotation frames rendered.")
