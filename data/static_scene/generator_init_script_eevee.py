"""Blender script for static scene initialization using EEVEE (Camera1 only)."""
import bpy
import os
import sys

if __name__ == "__main__":

    code_fpath = sys.argv[6]  # Path to the code file
    if len(sys.argv) > 7:
        rendering_dir = sys.argv[7] # Path to save the rendering from camera1
    else:
        rendering_dir = None
    if len(sys.argv) > 8:
        save_blend = sys.argv[8] # Path to save the blend file
    else:
        save_blend = None

    # Use EEVEE for fast iteration rendering
    if bpy.app.version >= (4, 0, 0):
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    # Setting up rendering resolution
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    # Set color mode to RGB
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

    # Read and execute the code from the specified file
    with open(code_fpath, "r") as f:
        code = f.read()
    try:
        exec(code)
    except:
        raise ValueError

    # Render from camera1
    if 'Camera1' in bpy.data.objects and rendering_dir:
        bpy.context.scene.camera = bpy.data.objects['Camera1']
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, 'render1.png')
        bpy.ops.render.render(write_still=True)

    # Save the blend file
    if save_blend:
        # Set the save version to 0
        bpy.context.preferences.filepaths.save_version = 0
        # Save the blend file
        bpy.ops.wm.save_as_mainfile(filepath=save_blend)
