from PIL import Image
from pathlib import Path
import sys

def create_pingpong_gif(frame_dir, start_frame, end_frame, output_path, duration=80):
    """Create ping-pong GIF from frame_XXXX.png files"""
    frames = []
    for i in range(start_frame, end_frame + 1):
        frame_path = Path(frame_dir) / f"frame_{i:04d}.png"
        if frame_path.exists():
            frames.append(frame_path)
    
    if not frames:
        print(f"No frames found in range {start_frame}-{end_frame}")
        return
    
    print(f"Found {len(frames)} frames")
    
    images = [Image.open(f) for f in frames]
    
    # Ping-pong: forward + reverse (excluding first and last to avoid duplicate)
    pingpong = images + images[-2:0:-1]
    
    print(f"Ping-pong sequence: {len(pingpong)} frames")
    
    pingpong[0].save(
        str(output_path), 
        save_all=True, 
        append_images=pingpong[1:], 
        duration=duration, 
        loop=0
    )
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    frame_dir = r'C:\output\viga_renders\rotations'
    output_dir = Path(r'C:\output\viga_renders')
    
    # Y-axis rotation GIF (frames 0-19 are Y rotation based on the rendering log)
    print("Creating Y-axis ping-pong GIF...")
    create_pingpong_gif(frame_dir, 0, 19, output_dir / 'viga_y_rotation.gif')
    
    print("\nDone!")