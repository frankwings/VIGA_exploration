"""Render rotation GIF from Run 7 final scene (round 9)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BLENDER_CMD = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
BLEND_FILE = Path(r"D:\Projects\ProjectGenesis\GenesisVIGA\output\static_scene\20260208_164519\greentea\renders\9\state.blend")
OUTPUT_DIR = Path(r"D:\Projects\ProjectGenesis\GenesisVIGA\output\static_scene\20260208_164519\greentea\rotation_gif")
NUM_FRAMES = 36
RESOLUTION = 512

BLENDER_SCRIPT = r'''
import bpy
import math
import sys
import os
import statistics
from mathutils import Vector

argv = sys.argv
args = argv[argv.index("--") + 1:]
output_dir = args[0]
num_frames = int(args[1])
res = int(args[2])

os.makedirs(output_dir, exist_ok=True)

scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT' if bpy.app.version >= (4, 0, 0) else 'BLENDER_EEVEE'
scene.render.resolution_x = res
scene.render.resolution_y = res
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = False

mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
if not mesh_objects:
    print("No mesh objects found!")
    sys.exit(1)

camera = scene.camera
if camera is None:
    cam_data = bpy.data.cameras.new("RotationCam")
    camera = bpy.data.objects.new("RotationCam", cam_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

print(f"Camera lens: {camera.data.lens}mm")
scene.frame_set(1)

# Compute scene center from all mesh bounding boxes
all_corners = []
for obj in mesh_objects:
    if obj.name == 'Ground':
        continue
    if hasattr(obj, 'bound_box') and obj.bound_box:
        all_corners.extend(obj.matrix_world @ Vector(c) for c in obj.bound_box)

if all_corners:
    anchor = sum(all_corners, Vector()) / len(all_corners)
else:
    anchor = Vector((0, 0, 0.3))

print(f"Anchor point: {anchor}")

cam_pos = camera.matrix_world.translation.copy()
offset = cam_pos - anchor
radius = offset.length
radius = max(radius, 0.5)
radius = min(radius, 15.0)

horiz_dist = math.sqrt(offset.x**2 + offset.y**2)
elevation = math.atan2(offset.z, horiz_dist)
start_angle = math.atan2(offset.y, offset.x)

print(f"Orbit radius: {radius:.2f}, elevation: {math.degrees(elevation):.1f} deg")

for i in range(num_frames):
    angle = start_angle + (2 * math.pi * i) / num_frames
    x = anchor.x + radius * math.cos(elevation) * math.cos(angle)
    y = anchor.y + radius * math.cos(elevation) * math.sin(angle)
    z = anchor.z + radius * math.sin(elevation)
    camera.location = Vector((x, y, z))
    direction = anchor - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    scene.render.filepath = os.path.join(output_dir, f"frame_{i:03d}.png")
    bpy.ops.render.render(write_still=True)
    print(f"Rendered frame {i+1}/{num_frames}")

print("Rotation rendering complete!")
'''


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_dir = OUTPUT_DIR / "frames"
    gif_path = OUTPUT_DIR / "rotation.gif"

    # Write blender script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(BLENDER_SCRIPT)
        script_path = f.name

    try:
        cmd = [
            BLENDER_CMD,
            "--background", str(BLEND_FILE),
            "--python", script_path,
            "--", str(frames_dir), str(NUM_FRAMES), str(RESOLUTION)
        ]
        cmd_str = ' '.join(f'"{c}"' if ' ' in c else c for c in cmd)

        env = os.environ.copy()
        env['AL_LIB_LOGLEVEL'] = '0'

        print(f"Running Blender rotation render ({NUM_FRAMES} frames)...")
        result = subprocess.run(cmd_str, shell=True, stdin=subprocess.DEVNULL, env=env, timeout=600)

        if result.returncode != 0:
            print("Blender render failed!")
            return
    except Exception as e:
        print(f"Error: {e}")
        return
    finally:
        os.unlink(script_path)

    # Create GIF
    try:
        from PIL import Image
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if not frame_files:
            print("No frames found!")
            return

        frames = []
        for fp in frame_files:
            img = Image.open(fp)
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (40, 40, 40))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((384, 384), Image.LANCZOS)
            frames.append(img)

        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:],
            duration=100, loop=0, optimize=True
        )
        print(f"\nGIF saved: {gif_path} ({len(frames)} frames)")
    except Exception as e:
        print(f"GIF creation failed: {e}")

    # Also render all rounds
    rounds = [1, 3, 6, 9]
    for r in rounds:
        blend = Path(f"D:/Projects/ProjectGenesis/GenesisVIGA/output/static_scene/20260208_164519/greentea/renders/{r}/state.blend")
        if not blend.exists():
            continue
        r_frames_dir = OUTPUT_DIR / f"round_{r}_frames"
        r_gif_path = OUTPUT_DIR / f"round_{r}.gif"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(BLENDER_SCRIPT)
            script_path = f.name

        try:
            cmd = [
                BLENDER_CMD,
                "--background", str(blend),
                "--python", script_path,
                "--", str(r_frames_dir), str(NUM_FRAMES), str(RESOLUTION)
            ]
            cmd_str = ' '.join(f'"{c}"' if ' ' in c else c for c in cmd)
            print(f"\nRendering round {r} rotation...")
            subprocess.run(cmd_str, shell=True, stdin=subprocess.DEVNULL, env=env, timeout=600)
        finally:
            os.unlink(script_path)

        try:
            from PIL import Image
            frame_files = sorted(r_frames_dir.glob("frame_*.png"))
            if frame_files:
                frames = []
                for fp in frame_files:
                    img = Image.open(fp)
                    if img.mode == 'RGBA':
                        bg = Image.new('RGB', img.size, (40, 40, 40))
                        bg.paste(img, mask=img.split()[-1])
                        img = bg
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((384, 384), Image.LANCZOS)
                    frames.append(img)
                frames[0].save(
                    r_gif_path, save_all=True, append_images=frames[1:],
                    duration=100, loop=0, optimize=True
                )
                print(f"Round {r} GIF saved: {r_gif_path}")
        except Exception as e:
            print(f"Round {r} GIF failed: {e}")


if __name__ == "__main__":
    main()
