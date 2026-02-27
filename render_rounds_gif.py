"""Render each round's state.blend as a rotation GIF and compile results into MD."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Configuration
BLENDER_CMD = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
BASE_DIR = Path(r"D:\Projects\ProjectGenesis\GenesisVIGA\output\dynamic_scene\20260206_011742\artist\renders")
OUTPUT_DIR = Path(r"D:\Projects\ProjectGenesis\GenesisVIGA\output\dynamic_scene\20260206_011742\artist\gifs")
TARGET_IMAGE = Path(r"D:\Projects\ProjectGenesis\GenesisVIGA\data\dynamic_scene\artist\target.png")
NUM_FRAMES = 36  # 36 frames = 10 degrees per frame
RESOLUTION_X = 512
RESOLUTION_Y = 512

# Blender script template for rotation rendering
BLENDER_SCRIPT = '''
import bpy
import math
import sys
import os
import statistics
from mathutils import Vector

# Parse args after "--"
argv = sys.argv
args = argv[argv.index("--") + 1:]
output_dir = args[0]
num_frames = int(args[1])
res_x = int(args[2])
res_y = int(args[3])

os.makedirs(output_dir, exist_ok=True)

# Set render settings
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT' if bpy.app.version >= (4, 0, 0) else 'BLENDER_EEVEE'
scene.render.resolution_x = res_x
scene.render.resolution_y = res_y
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = False

# Check for mesh objects
mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
if not mesh_objects:
    print("No mesh objects found!")
    sys.exit(1)

# --- Get existing camera ---
camera = scene.camera
if camera is None:
    cam_data = bpy.data.cameras.new("RotationCam")
    camera = bpy.data.objects.new("RotationCam", cam_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera

# Keep original camera lens (don't override GPT-set focal length)
print(f"Camera lens: {camera.data.lens}mm")

# Set to animation frame 1 so objects are in their initial positions
scene.frame_set(1)

# --- Ray-casting anchor point from original camera ---
depsgraph = bpy.context.evaluated_depsgraph_get()

cam_pos = camera.matrix_world.translation.copy()
cam_forward = (camera.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
cam_right = (camera.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized()
cam_up = (camera.matrix_world.to_3x3() @ Vector((0, 1, 0))).normalized()

print(f"Camera pos: {cam_pos}")
print(f"Camera forward: {cam_forward}")

# Cast a grid of rays from the camera position
spread_angles_deg = [0, -10, 10, -20, 20, -5, 5, -15, 15]
hit_depths = []

for h_deg in spread_angles_deg:
    for v_deg in spread_angles_deg:
        h_rad = math.radians(h_deg)
        v_rad = math.radians(v_deg)
        ray_dir = (cam_forward
                   + math.tan(h_rad) * cam_right
                   + math.tan(v_rad) * cam_up).normalized()
        result, location, normal, index, obj, matrix = scene.ray_cast(
            depsgraph, cam_pos, ray_dir, distance=1000.0
        )
        if result:
            depth = (location - cam_pos).length
            hit_depths.append(depth)
            print(f"  Ray ({h_deg:+3d},{v_deg:+3d}) hit '{obj.name}' at depth {depth:.2f}")

if hit_depths:
    median_depth = statistics.median(hit_depths)
    print(f"Hit count: {len(hit_depths)}, median depth: {median_depth:.2f}")
else:
    print("No ray hits! Falling back to bounding box center.")
    all_corners = []
    for obj in mesh_objects:
        if hasattr(obj, 'bound_box') and obj.bound_box:
            all_corners.extend(obj.matrix_world @ Vector(c) for c in obj.bound_box)
    if all_corners:
        bb_center = sum(all_corners, Vector()) / len(all_corners)
        median_depth = (bb_center - cam_pos).length
    else:
        median_depth = 5.0

# Anchor = camera forward * median_depth
anchor = cam_pos + cam_forward * median_depth

# Compute orbit params from original camera position relative to anchor
offset = cam_pos - anchor
radius = offset.length
radius = max(radius, 1.0)
radius = min(radius, 30.0)

# Derive start_angle and elevation from original camera position
horiz_dist = math.sqrt(offset.x**2 + offset.y**2)
elevation = math.atan2(offset.z, horiz_dist)
start_angle = math.atan2(offset.y, offset.x)

print(f"Anchor point: {anchor}")
print(f"Orbit radius: {radius:.2f}, elevation: {math.degrees(elevation):.1f} deg")
print(f"Start angle: {math.degrees(start_angle):.1f} deg")

# Render rotation frames: frame 0 starts from original camera position
for i in range(num_frames):
    angle = start_angle + (2 * math.pi * i) / num_frames

    x = anchor.x + radius * math.cos(elevation) * math.cos(angle)
    y = anchor.y + radius * math.cos(elevation) * math.sin(angle)
    z = anchor.z + radius * math.sin(elevation)

    camera.location = Vector((x, y, z))

    # Point camera at anchor
    direction = anchor - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    scene.render.filepath = os.path.join(output_dir, f"frame_{i:03d}.png")
    bpy.ops.render.render(write_still=True)
    print(f"Rendered frame {i+1}/{num_frames}")

print("Rotation rendering complete!")
'''


def render_blend_rotation(blend_path: Path, output_dir: Path) -> bool:
    """Render a .blend file as rotation frames."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the blender script to a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(BLENDER_SCRIPT)
        script_path = f.name

    try:
        cmd = [
            BLENDER_CMD,
            "--background", str(blend_path),
            "--python", script_path,
            "--", str(output_dir), str(NUM_FRAMES), str(RESOLUTION_X), str(RESOLUTION_Y)
        ]

        # Windows quoting
        cmd_str = ' '.join(f'"{c}"' if ' ' in c else c for c in cmd)

        env = os.environ.copy()
        env['AL_LIB_LOGLEVEL'] = '0'

        result = subprocess.run(
            cmd_str, shell=True,
            stdin=subprocess.DEVNULL,
            capture_output=False,
            env=env, timeout=600
        )

        return result.returncode == 0
    except Exception as e:
        print(f"  Error: {e}")
        return False
    finally:
        os.unlink(script_path)


def frames_to_gif(frames_dir: Path, gif_path: Path, duration_ms: int = 100) -> bool:
    """Convert rendered frames to GIF."""
    try:
        from PIL import Image

        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if not frame_files:
            print(f"  No frames found in {frames_dir}")
            return False

        frames = []
        for fp in frame_files:
            img = Image.open(fp)
            # Convert to RGB if needed (GIF doesn't support RGBA well)
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize for reasonable GIF size
            img = img.resize((384, 384), Image.LANCZOS)
            frames.append(img)

        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True
        )

        print(f"  GIF saved: {gif_path} ({len(frames)} frames)")
        return True
    except Exception as e:
        print(f"  GIF creation failed: {e}")
        return False


def create_markdown(gif_dir: Path, md_path: Path, target_img: Path, rounds_info: dict):
    """Create markdown document with embedded GIF results."""
    lines = []
    lines.append("# VIGA Dynamic Scene - Artist Task Results")
    lines.append("")
    lines.append("**Task:** Reconstruct a Cezanne still life painting as a 3D Blender scene")
    lines.append(f"**Date:** 2026-02-06")
    lines.append(f"**Model:** GPT-5")
    lines.append(f"**Total Rounds:** {len(rounds_info)}")
    lines.append("")

    lines.append("## Target Image")
    lines.append("")
    # Use relative path from the MD file location
    target_rel = os.path.relpath(target_img, md_path.parent)
    lines.append(f"![Target]({target_rel.replace(os.sep, '/')})")
    lines.append("")

    lines.append("## Iteration Results")
    lines.append("")

    for round_num in sorted(rounds_info.keys()):
        info = rounds_info[round_num]
        lines.append(f"### Round {round_num}")
        lines.append("")

        if info.get("gif_path"):
            gif_rel = os.path.relpath(info["gif_path"], md_path.parent)
            lines.append(f"![Round {round_num} - 360° Rotation]({gif_rel.replace(os.sep, '/')})")

        if info.get("keyframes"):
            lines.append("")
            lines.append("**Keyframes:**")
            lines.append("")
            lines.append("| Frame 1 | Frame 2 | Frame 3 |")
            lines.append("|---------|---------|---------|")
            kf_cells = []
            for kf in info["keyframes"][:3]:
                kf_rel = os.path.relpath(kf, md_path.parent)
                kf_cells.append(f"![kf]({kf_rel.replace(os.sep, '/')})")
            while len(kf_cells) < 3:
                kf_cells.append("(none)")
            lines.append(f"| {' | '.join(kf_cells)} |")

        if info.get("error"):
            lines.append(f"*{info['error']}*")

        lines.append("")

    lines.append("---")
    lines.append("*Generated by VIGA (Vision-as-Inverse-Graphics Agent)*")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Markdown saved: {md_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rounds_info = {}

    # Process each round
    for round_num in range(1, 13):
        round_dir = BASE_DIR / str(round_num)
        blend_file = round_dir / "state.blend"

        print(f"\n{'='*60}")
        print(f"Round {round_num}")
        print(f"{'='*60}")

        if not blend_file.exists():
            print(f"  No state.blend found, skipping")
            rounds_info[round_num] = {"error": "No state.blend found"}
            continue

        # Collect existing keyframe renders
        keyframes = sorted([str(p) for p in round_dir.glob("Camera_*.png")])

        # Render rotation frames
        frames_dir = OUTPUT_DIR / f"round_{round_num}_frames"
        gif_path = OUTPUT_DIR / f"round_{round_num}.gif"

        print(f"  Rendering rotation from {blend_file}...")
        success = render_blend_rotation(blend_file, frames_dir)

        if success:
            print(f"  Creating GIF...")
            gif_ok = frames_to_gif(frames_dir, gif_path)
            rounds_info[round_num] = {
                "gif_path": str(gif_path) if gif_ok else None,
                "keyframes": keyframes,
                "error": None if gif_ok else "GIF creation failed"
            }
        else:
            rounds_info[round_num] = {
                "gif_path": None,
                "keyframes": keyframes,
                "error": "Blender render failed"
            }

    # Create markdown
    md_path = OUTPUT_DIR.parent / "RESULTS.md"
    create_markdown(OUTPUT_DIR, md_path, TARGET_IMAGE, rounds_info)

    print(f"\nDone! Results at: {md_path}")


if __name__ == "__main__":
    main()
