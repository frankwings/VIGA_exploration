"""
Blender script to render 360° rotation animations
- Rotate the OBJECT, not the camera
- Camera stays fixed, object rotates around Y and X axes

Usage:
    blender --background --python blender_render_rotation.py -- input.glb output_dir [--frames 30]
"""
import bpy
import mathutils
import math
import sys
from pathlib import Path

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    args = {
        'input': argv[0] if len(argv) > 0 else None,
        'output_dir': argv[1] if len(argv) > 1 else 'rotation_frames',
        'frames': 30,
        'resolution': 512,
    }

    i = 2
    while i < len(argv):
        if argv[i] == '--frames' and i + 1 < len(argv):
            args['frames'] = int(argv[i + 1])
            i += 2
        elif argv[i] == '--resolution' and i + 1 < len(argv):
            args['resolution'] = int(argv[i + 1])
            i += 2
        else:
            i += 1
    
    return args

def setup_scene(resolution=512):
    """Clear scene and setup rendering"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for collection in bpy.data.collections:
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj)

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 64
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    
    world = bpy.data.worlds.new("DarkWorld")
    world.use_nodes = True
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs[0].default_value = (0.08, 0.08, 0.08, 1)
    scene.world = world

def import_glb(filepath):
    """Import GLB and return center, size, and mesh objects"""
    bpy.ops.import_scene.gltf(filepath=filepath)
    
    imported = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    print(f"Imported {len(imported)} mesh objects")
    
    # Calculate bounds
    min_co = [float('inf')] * 3
    max_co = [float('-inf')] * 3
    
    for obj in imported:
        bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
        for corner in bbox_corners:
            for i in range(3):
                min_co[i] = min(min_co[i], corner[i])
                max_co[i] = max(max_co[i], corner[i])
    
    center = mathutils.Vector([(min_co[i] + max_co[i]) / 2 for i in range(3)])
    size = max([max_co[i] - min_co[i] for i in range(3)])
    
    return center, size, imported

def create_pivot_parent(objects, center):
    """Create an empty at center and parent all objects to it"""
    # Create empty at object center
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=center)
    pivot = bpy.context.object
    pivot.name = "ObjectPivot"
    
    # Parent all mesh objects to pivot
    for obj in objects:
        obj.parent = pivot
        # Keep transform (don't move objects when parenting)
        obj.matrix_parent_inverse = pivot.matrix_world.inverted()
    
    return pivot

def add_lighting(center, size):
    """Add lighting setup"""
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 8))
    sun = bpy.context.object
    sun.data.energy = 3
    sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))
    
    bpy.ops.object.light_add(type='AREA', location=(-3, -3, 4))
    fill = bpy.context.object
    fill.data.energy = 150
    fill.data.size = 3
    
    bpy.ops.object.light_add(type='AREA', location=(0, 5, 3))
    rim = bpy.context.object
    rim.data.energy = 100
    rim.data.size = 2

def setup_camera(center, size):
    """Create camera looking at center"""
    radius = size * 2.2
    
    # Camera position: in front, slightly elevated
    cam_location = (center.x, center.y - radius, center.z + size * 0.3)
    
    bpy.ops.object.camera_add(location=cam_location)
    camera = bpy.context.object
    camera.name = "RenderCamera"
    bpy.context.scene.camera = camera
    
    # Point camera at center
    direction = center - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    return camera

def render_y_rotation(pivot, n_frames, output_dir, basename):
    """
    Y-axis rotation: rotate object around Z axis (Blender's up)
    This gives a turntable effect
    """
    print(f"📷 Rendering Y-axis rotation ({n_frames} frames)...")
    
    pivot.rotation_mode = 'QUATERNION'
    
    for i in range(n_frames):
        angle = (2 * math.pi * i) / n_frames
        
        # Rotate around Z axis (vertical in Blender)
        quat = mathutils.Quaternion((0, 0, 1), angle)
        pivot.rotation_quaternion = quat
        
        output_path = output_dir / f"{basename}_y_{i:02d}.png"
        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        
        print(f"   ✓ Y-axis {i+1}/{n_frames} ({int(360*i/n_frames)}°)")
    
    # Reset rotation
    pivot.rotation_quaternion = mathutils.Quaternion()

def render_x_rotation(pivot, n_frames, output_dir, basename):
    """
    X-axis rotation: rotate object around X axis
    This gives a tumbling/tilting effect
    """
    print(f"📷 Rendering X-axis rotation ({n_frames} frames)...")
    
    pivot.rotation_mode = 'QUATERNION'
    
    for i in range(n_frames):
        angle = (2 * math.pi * i) / n_frames
        
        # Rotate around X axis
        quat = mathutils.Quaternion((1, 0, 0), angle)
        pivot.rotation_quaternion = quat
        
        output_path = output_dir / f"{basename}_x_{i:02d}.png"
        bpy.context.scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        
        print(f"   ✓ X-axis {i+1}/{n_frames} ({int(360*i/n_frames)}°)")
    
    # Reset rotation
    pivot.rotation_quaternion = mathutils.Quaternion()

def main():
    args = parse_args()
    
    if not args['input']:
        print("Usage: blender --background --python blender_render_rotation.py -- input.glb output_dir [--frames 30]")
        return
    
    input_path = Path(args['input'])
    output_dir = Path(args['output_dir'])
    n_frames = args['frames']
    
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = input_path.stem
    
    print(f"📂 Input: {input_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🎬 Frames per axis: {n_frames}")
    
    # Setup
    setup_scene(resolution=args['resolution'])
    center, size, objects = import_glb(str(input_path))
    print(f"   Center: [{center.x:.2f}, {center.y:.2f}, {center.z:.2f}], Size: {size:.2f}")
    
    # Create pivot at center and parent objects to it
    pivot = create_pivot_parent(objects, center)
    
    add_lighting(center, size)
    setup_camera(center, size)
    
    # Render rotations (rotate object, not camera)
    render_y_rotation(pivot, n_frames, output_dir, basename)
    render_x_rotation(pivot, n_frames, output_dir, basename)
    
    print(f"✅ Done! {n_frames * 2} frames saved to: {output_dir}")

if __name__ == "__main__":
    main()
