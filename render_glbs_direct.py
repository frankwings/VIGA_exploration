"""Direct GLB loader — no VIGA, no modifications. Just import and render each one individually."""
import bpy
import os
import math
from mathutils import Vector

ASSETS_DIR = "D:/Projects/ProjectGenesis/GenesisVIGA/data/static_scene/greentea/assets"
OUTPUT_DIR = "D:/Projects/ProjectGenesis/GenesisVIGA/output/glb_direct_render"

GLBS = [
    "green_tea_bottle.glb",
    "alienware_keyboard.glb",
    "headphones.glb",
    "envelope.glb",
    "ito_en_bottle.glb",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "individual"), exist_ok=True)

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
        return Vector((0,0,0)), Vector((0,0,0))
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

def setup_camera_for_bounds(min_v, max_v):
    """Place camera to frame the object nicely."""
    center = (min_v + max_v) / 2.0
    dims = max_v - min_v
    max_dim = max(dims.x, dims.y, dims.z)

    # Distance = 2x the max dimension
    dist = max(max_dim * 2.5, 0.5)

    cam_data = bpy.data.cameras.new('Camera')
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    bpy.context.collection.objects.link(cam_obj)

    # Camera at 30 deg above, front-left
    angle_h = math.radians(-30)
    angle_v = math.radians(25)
    cam_obj.location = (
        center.x + dist * math.sin(angle_h) * math.cos(angle_v),
        center.y - dist * math.cos(angle_h) * math.cos(angle_v),
        center.z + dist * math.sin(angle_v)
    )

    direction = center - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()
    bpy.context.scene.camera = cam_obj
    return cam_obj

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

# --- Render settings ---
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 1024
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.cycles.samples = 256

# Enable GPU
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.preferences.addons['cycles'].preferences.get_devices()
for device in bpy.context.preferences.addons['cycles'].preferences.devices:
    if device.type in ('CUDA', 'OPTIX'):
        device.use = True
bpy.context.scene.cycles.device = 'GPU'

# --- Render each GLB individually ---
for glb_name in GLBS:
    path = os.path.join(ASSETS_DIR, glb_name)
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

    print(f"\n{'='*60}")
    print(f"GLB: {glb_name}")
    print(f"  Objects imported: {len(new_objs)} -> {new_names}")
    print(f"  Bounding box min: ({min_v.x:.4f}, {min_v.y:.4f}, {min_v.z:.4f})")
    print(f"  Bounding box max: ({max_v.x:.4f}, {max_v.y:.4f}, {max_v.z:.4f})")
    print(f"  Dimensions: ({dims.x:.4f}, {dims.y:.4f}, {dims.z:.4f})")
    print(f"  Max dimension: {max(dims.x, dims.y, dims.z):.4f}")
    for o in new_objs:
        print(f"    - {o.name} type={o.type} loc={o.location[:]} scale={o.scale[:]}")
    print(f"{'='*60}")

    # Setup scene
    add_ground(min_v.z)
    setup_lighting()
    setup_camera_for_bounds(min_v, max_v)

    # Render
    name = glb_name.replace('.glb', '')
    render_path = os.path.join(OUTPUT_DIR, "individual", f"{name}.png")
    bpy.context.scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {render_path}")

    # Save .blend for this asset
    blend_path = os.path.join(OUTPUT_DIR, "individual", f"{name}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"Saved: {blend_path}")

# --- Now render all together ---
print("\n\nRendering all GLBs together...")
clear_scene()

all_objs_list = []
spacing_x = 0  # Will be computed based on actual sizes

group_data = []  # (name, objs, min_v, max_v)
for glb_name in GLBS:
    path = os.path.join(ASSETS_DIR, glb_name)
    if not os.path.exists(path):
        continue
    pre = set(o.name for o in bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    post = set(o.name for o in bpy.data.objects)
    new_names = list(post - pre)
    new_objs = [bpy.data.objects[n] for n in new_names]
    bpy.context.view_layer.update()
    min_v, max_v = get_all_mesh_bounds(new_objs)
    dims = max_v - min_v
    max_dim = max(dims.x, dims.y, dims.z)

    # Normalize each to 1.0 max dim for the group render
    if max_dim > 0:
        scale_factor = 1.0 / max_dim
        for o in new_objs:
            o.scale = [s * scale_factor for s in o.scale]
        bpy.context.view_layer.update()
        min_v, max_v = get_all_mesh_bounds(new_objs)

    group_data.append((glb_name, new_objs, min_v, max_v))
    all_objs_list.extend(new_objs)

# Space them out
offset_x = 0
gap = 0.3
for i, (name, objs, min_v, max_v) in enumerate(group_data):
    dims = max_v - min_v
    center = (min_v + max_v) / 2.0
    # Shift so this group starts at offset_x
    target_x = offset_x + dims.x / 2.0
    delta = Vector((target_x - center.x, -center.y, -min_v.z))
    for o in objs:
        o.location = o.location + delta
    offset_x += dims.x + gap

bpy.context.view_layer.update()

# Get overall bounds
overall_min, overall_max = get_all_mesh_bounds(all_objs_list)
add_ground(overall_min.z)
setup_lighting()
setup_camera_for_bounds(overall_min, overall_max)

# Render group CYCLES
render_path = os.path.join(OUTPUT_DIR, "render_all_cycles.png")
bpy.context.scene.render.filepath = render_path
bpy.ops.render.render(write_still=True)
print(f"Rendered group: {render_path}")

# Save group .blend
blend_path = os.path.join(OUTPUT_DIR, "glb_direct_all.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print(f"Saved: {blend_path}")

print("\nDONE — all renders complete.")
