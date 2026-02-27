"""Assemble rotation frame PNGs into GIFs.

Usage:
    python assemble_rotation_gifs.py <frames_parent_dir> [--output-dir <gif_dir>] [--size 384]

Looks for subdirectories named *_frames/ containing frame_000.png, frame_001.png, ...
Creates {name}_rotation.gif for each.
"""
import argparse
import glob
import os
from pathlib import Path
from PIL import Image


def assemble_gif(frames_dir: str, output_path: str, size: int = 384):
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frame_paths:
        print(f"  SKIP (no frames): {frames_dir}")
        return

    frames = []
    for fp in frame_paths:
        img = Image.open(fp).convert("RGB")
        if size and (img.width != size or img.height != size):
            img = img.resize((size, size), Image.LANCZOS)
        frames.append(img)

    # Ping-pong loop: forward + reverse (minus endpoints to avoid freeze)
    if len(frames) > 2:
        sequence = list(frames) + list(reversed(frames[1:-1]))
    else:
        sequence = list(frames)

    sequence[0].save(
        output_path,
        save_all=True,
        append_images=sequence[1:],
        duration=80,
        loop=0,
    )
    print(f"  GIF: {output_path} ({len(sequence)} frames, {size}x{size})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames_parent_dir", help="Directory containing *_frames/ subdirs")
    parser.add_argument("--output-dir", default=None, help="Output dir for GIFs (default: same as input)")
    parser.add_argument("--size", type=int, default=384, help="GIF frame size (px)")
    args = parser.parse_args()

    parent = args.frames_parent_dir
    out_dir = args.output_dir or parent
    os.makedirs(out_dir, exist_ok=True)

    frame_dirs = sorted(glob.glob(os.path.join(parent, "*_frames")))
    if not frame_dirs:
        print(f"No *_frames/ subdirectories found in {parent}")
        return

    print(f"Found {len(frame_dirs)} frame directories")
    for fd in frame_dirs:
        name = os.path.basename(fd).replace("_frames", "")
        gif_path = os.path.join(out_dir, f"{name}_rotation.gif")
        assemble_gif(fd, gif_path, size=args.size)

    print("Done.")


if __name__ == "__main__":
    main()
