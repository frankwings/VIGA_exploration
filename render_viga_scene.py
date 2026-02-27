#!/usr/bin/env python3
"""
Blender script to render VIGA generated scene
Usage: blender --background scene.blend --python render_viga_scene.py
"""
import bpy
import mathutils
import math
import os
from pathlib import Path

def setup_camera_and_lighting():
    """Add camera and lighting if they don't exist."""
    # Clear existing cameras and lights
    bpy.ops.object.select_all(action='DESELECT')
    objects_to_remove = []
    for obj in bpy.context.scene.objects:
        if obj.type in ('CAMERA', 'LIGHT'):
            objects_to_remove.append(obj)
    
    for obj in objects_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # Add camera
    bpy.ops.object.camera_add(location=(3, -3, 2))
    camera = bpy.context.active_object
    camera.name = "RenderCamera"
    
    # Point camera at scene center
    target = mathutils.Vector((0, 0, 0))
    camera.location = mathutils.Vector((3, -3, 2))
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    # Add key light
    bpy.ops.object.light_add(type='SUN', location=(2, 2, 4))
    light = bpy.context.active_object
    light.name = "KeyLight"
    light.data.energy = 3
    
    # Add fill light
    bpy.ops.object.light_add(type='AREA', location=(-2, 1, 2))
    fill_light = bpy.context.active_object
    fill_light.name = "FillLight"
    fill_light.data.energy = 1

def render_static_image(output_path):
    """Render a static image."""
    # Set render settings
    scene = bpy.context.scene
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_path
    
    # Render
    bpy.ops.render.render(write_still=True)
    print(f"✅ Static render saved: {output_path}")

def render_rotation_sequence(output_dir, axis, frames=20):
    """Render rotation animation around specified axis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all mesh objects to rotate
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if not mesh_objects:
        print("❌ No mesh objects found to rotate")
        return
    
    # Store original rotations
    original_rotations = {}
    for obj in mesh_objects:
        original_rotations[obj.name] = obj.rotation_euler.copy()
    
    frame_paths = []
    
    for frame in range(frames):
        angle = (frame / frames) * 2 * math.pi
        
        # Rotate all mesh objects
        for obj in mesh_objects:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            
            if axis == 'y':
                obj.rotation_euler = (
                    original_rotations[obj.name][0],
                    original_rotations[obj.name][1] + angle,
                    original_rotations[obj.name][2]
                )
            elif axis == 'x':
                obj.rotation_euler = (
                    original_rotations[obj.name][0] + angle,
                    original_rotations[obj.name][1],
                    original_rotations[obj.name][2]
                )
        
        # Update scene
        bpy.context.view_layer.update()
        
        # Render frame
        frame_path = output_dir / f"frame_{frame:04d}.png"
        bpy.context.scene.render.filepath = str(frame_path)
        bpy.ops.render.render(write_still=True)
        frame_paths.append(str(frame_path))
        print(f"📸 Rendered frame {frame+1}/{frames}: {axis}-axis")
    
    # Create GIF using PIL
    try:
        from PIL import Image
        images = [Image.open(path) for path in frame_paths]
        gif_path = output_dir / f"rotation_{axis}_axis.gif"
        
        # Create ping-pong animation
        images_ping_pong = images + images[-2:0:-1]
        images_ping_pong[0].save(
            gif_path,
            save_all=True,
            append_images=images_ping_pong[1:],
            duration=100,  # ms per frame
            loop=0
        )
        print(f"🎬 GIF created: {gif_path}")
        
        # Clean up frame images
        for path in frame_paths:
            try:
                os.remove(path)
            except:
                pass
                
    except ImportError:
        print(f"❌ PIL not available, frames saved to: {output_dir}")
    
    # Restore original rotations
    for obj in mesh_objects:
        if obj.name in original_rotations:
            obj.rotation_euler = original_rotations[obj.name]

def main():
    """Main rendering function."""
    print("🎬 Starting VIGA scene rendering...")
    
    # Setup scene
    setup_camera_and_lighting()
    
    # Create output directory
    output_dir = Path("output/viga_renders")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Render static image
    static_path = output_dir / "greentea_static.png"
    render_static_image(str(static_path))
    
    # Render rotation GIFs
    rotation_dir = output_dir / "rotations"
    render_rotation_sequence(str(rotation_dir), 'y', frames=20)
    render_rotation_sequence(str(rotation_dir), 'x', frames=20)
    
    print("🎉 All renders completed!")

if __name__ == "__main__":
    main()