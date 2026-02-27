"""Blender script to render SAM3D GLB files as 360° rotation GIFs.
Run with: blender --background --python render_sam3d_glbs.py

Renders 36 frames per GLB (10° increments), saves PNGs to subfolders.
GIF assembly is done separately (see bottom of script for PIL code).
"""
import bpy
import os
import math
from mathutils import Vector

SAM3D_DIR = "D:/Projects/ProjectGenesis/GenesisVIGA/output/dynamic_scene/20260210_194152/artist/sam_init"
OUTPUT_DIR = "D:/Projects/ProjectGenesis/GenesisVIGA/docs/test_results_images/dynamic_artist_run1/sam3d_renders"
NUM_FRAMES = 36
RESOLUTION = 384

GLBS = [
    "ceramic_jug.glb",
    "green_pears.glb",
    "orange_pears.glb",
    "plate_with_fruits.glb",
    "orange_pear.glb",
    "green_apple.glb",
    "green_pear.glb",
    "green_apple_1.glb",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def get_all_mesh_bounds(objs):
    """Get world-space bounding box from mesh vertices directly."""
    min_v = Vector((float('inf'), float('inf'), float('inf')))
    max_v = Vector((float('-inf'), float('-inf'), float('-inf')))
    found = False
    for obj in objs:
        if obj.type != 'MESH':
            continue
        mesh = obj.data
        for vert in mesh.vertices:
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

    # Soft world ambient
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
    """Add a ground plane at the bottom of the object."""
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
    """Orbit camera around center and render each frame."""
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


# --- Render settings (EEVEE for speed) ---
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
bpy.context.scene.render.resolution_x = RESOLUTION
bpy.context.scene.render.resolution_y = RESOLUTION
bpy.context.scene.render.image_settings.file_format = 'PNG'

# --- Render rotation for each GLB ---
for glb_name in GLBS:
    path = os.path.join(SAM3D_DIR, glb_name)
    if not os.path.exists(path):
        print(f"SKIP: {path}")
        continue

    clear_scene()

    # Import
    pre = set(o.name for o in bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    post = set(o.name for o in bpy.data.objects)
    new_names = list(post - pre)
    new_objs = [bpy.data.objects[n] for n in new_names]

    bpy.context.view_layer.update()
    min_v, max_v = get_all_mesh_bounds(new_objs)
    dims = max_v - min_v
    max_dim = max(dims.x, dims.y, dims.z)

    name = glb_name.replace('.glb', '')
    print(f"\n{'='*60}")
    print(f"GLB: {glb_name} (max_dim={max_dim:.4f})")
    print(f"{'='*60}")

    center = (min_v + max_v) / 2.0
    dist = max(max_dim * 2.5, 0.5)
    elevation_rad = math.radians(25)

    # Setup scene
    add_ground(min_v.z)
    setup_lighting()

    # Create camera
    cam_data = bpy.data.cameras.new('Camera')
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Render rotation frames
    frames_dir = os.path.join(OUTPUT_DIR, f"{name}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    render_rotation(cam_obj, center, dist, elevation_rad, frames_dir, NUM_FRAMES)

print("\nDONE — all rotation frames rendered. Run GIF assembly next.")
