"""Debug script to print bounding boxes of imported GLBs."""

import json
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv
idx = argv.index("--")
transforms_path = os.path.abspath(argv[idx + 1])

# Clear scene
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

    name = os.path.splitext(os.path.basename(glb_path))[0]

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=glb_path)

    imported = bpy.context.selected_objects

    # Get bounding box across all meshes
    min_co = Vector((float('inf'), float('inf'), float('inf')))
    max_co = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in imported:
        if obj.type == 'MESH':
            mesh = obj.data
            for v in mesh.vertices:
                world_co = obj.matrix_world @ v.co
                for i in range(3):
                    if world_co[i] < min_co[i]:
                        min_co[i] = world_co[i]
                    if world_co[i] > max_co[i]:
                        max_co[i] = world_co[i]

    center = (min_co + max_co) / 2
    size = max_co - min_co
    print(f"[OBJ] {name:25s}  center=({center.x:+.3f}, {center.y:+.3f}, {center.z:+.3f})  size=({size.x:.3f}, {size.y:.3f}, {size.z:.3f})")

# Overall bounds
min_all = Vector((float('inf'), float('inf'), float('inf')))
max_all = Vector((float('-inf'), float('-inf'), float('-inf')))

for obj in bpy.data.objects:
    if obj.type == 'MESH':
        mesh = obj.data
        for v in mesh.vertices:
            world_co = obj.matrix_world @ v.co
            for i in range(3):
                if world_co[i] < min_all[i]:
                    min_all[i] = world_co[i]
                if world_co[i] > max_all[i]:
                    max_all[i] = world_co[i]

center_all = (min_all + max_all) / 2
size_all = max_all - min_all
print(f"\n[ALL] Scene bounds:")
print(f"      min=({min_all.x:+.3f}, {min_all.y:+.3f}, {min_all.z:+.3f})")
print(f"      max=({max_all.x:+.3f}, {max_all.y:+.3f}, {max_all.z:+.3f})")
print(f"      center=({center_all.x:+.3f}, {center_all.y:+.3f}, {center_all.z:+.3f})")
print(f"      size=({size_all.x:.3f}, {size_all.y:.3f}, {size_all.z:.3f})")
