#!/usr/bin/env python3
"""
Render point cloud rotation animation
"""
import argparse
import open3d as o3d
import numpy as np
from pathlib import Path
import math

def render_rotation(input_path: str, output_dir: str, n_frames: int = 12, 
                    point_size: float = 3.0, elevation: float = 0.3):
    """Render point cloud rotating around Z axis"""
    
    output_d = Path(output_dir)
    output_d.mkdir(parents=True, exist_ok=True)
    
    # Load point cloud
    print(f"📂 Loading: {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)
    n_points = len(pcd.points)
    print(f"   Points: {n_points:,}")
    print(f"   Rendering {n_frames} frames...")
    
    input_p = Path(input_path)
    
    for i in range(n_frames):
        # Calculate angle (full 360 rotation)
        angle = (2 * math.pi * i) / n_frames
        
        # Camera position on circle
        front = [math.sin(angle), -math.cos(angle), elevation]
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=512, height=512)
        vis.add_geometry(pcd)
        
        # Set render options
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.array([0.1, 0.1, 0.1])
        
        # Set camera
        ctr = vis.get_view_control()
        ctr.set_zoom(0.7)
        ctr.set_front(front)
        ctr.set_lookat(pcd.get_center())
        ctr.set_up([0, 0, 1])
        
        # Render and save
        vis.poll_events()
        vis.update_renderer()
        
        output_path = output_d / f"{input_p.stem}_{i:03d}.png"
        vis.capture_screen_image(str(output_path), do_render=True)
        
        vis.destroy_window()
        
        print(f"   Frame {i+1}/{n_frames}: {output_path.name}")
    
    print(f"✅ Done! Frames saved to: {output_d}")
    return output_d

def create_gif(frame_dir: str, output_path: str, duration: int = 100):
    """Create GIF from frames"""
    from PIL import Image
    import glob
    
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("*.png"))
    
    if not frames:
        print("No frames found!")
        return
    
    print(f"📦 Creating GIF from {len(frames)} frames...")
    
    images = [Image.open(f) for f in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"💾 GIF saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Render point cloud rotation")
    parser.add_argument("input", help="Input PLY file")
    parser.add_argument("-o", "--output-dir", default="rotation_frames", help="Output directory")
    parser.add_argument("-n", "--n-frames", type=int, default=12, help="Number of frames")
    parser.add_argument("--point-size", type=float, default=3.0, help="Point size")
    parser.add_argument("--elevation", type=float, default=0.3, help="Camera elevation")
    parser.add_argument("--gif", help="Also create GIF with this filename")
    parser.add_argument("--gif-speed", type=int, default=100, help="GIF frame duration in ms")
    
    args = parser.parse_args()
    
    frame_dir = render_rotation(
        args.input, 
        args.output_dir, 
        args.n_frames,
        args.point_size,
        args.elevation
    )
    
    if args.gif:
        create_gif(str(frame_dir), args.gif, args.gif_speed)

if __name__ == "__main__":
    main()
