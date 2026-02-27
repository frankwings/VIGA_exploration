"""
Render rotation GIFs for SAM3D convex-hull GLBs using Blender + PIL.
Run: python render_new_glbs.py
"""
import subprocess
import sys
from pathlib import Path
from PIL import Image

BLENDER = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
SCRIPT   = str(Path(__file__).parent / "tools" / "blender_render_rotation.py")

GLB_DIR  = Path("output/sam3d_convex_hull_v2")
OUT_DIR  = Path("output/sam3d_convex_hull_v2/gifs")
OUT_DIR.mkdir(exist_ok=True)

OBJECTS    = ["green_tea_bottle", "ito_en_bottle", "alienware_keyboard", "headphones", "envelope"]
NUM_FRAMES = 24


def frames_to_gif(frames_dir: Path, gif_path: Path, duration=80):
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        print(f"  No frames in {frames_dir}")
        return False
    imgs = [Image.open(f).convert("RGB").resize((384, 384), Image.LANCZOS) for f in frames]
    loop = imgs + imgs[-2:0:-1]
    loop[0].save(gif_path, save_all=True, append_images=loop[1:], duration=duration, loop=0)
    print(f"  Saved: {gif_path}")
    return True


for obj in OBJECTS:
    glb = (GLB_DIR / f"{obj}.glb").resolve()
    if not glb.exists():
        print(f"SKIP {obj}: GLB not found at {glb}")
        continue

    frames_dir = OUT_DIR / f"{obj}_frames"
    frames_dir.mkdir(exist_ok=True)

    print(f"\nRendering {obj} ({NUM_FRAMES} frames)...")
    result = subprocess.run(
        [BLENDER, "-b", "-P", SCRIPT, "--", str(glb), str(frames_dir.resolve()), "--frames", str(NUM_FRAMES)],
        timeout=300,
    )
    if result.returncode != 0:
        print(f"  Blender failed (exit {result.returncode})")
        continue

    frames_to_gif(frames_dir, OUT_DIR / f"{obj}.gif")

print("\nAll done. GIFs in:", OUT_DIR.resolve())
