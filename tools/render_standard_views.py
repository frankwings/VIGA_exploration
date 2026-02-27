#!/usr/bin/env python3
"""
Standard multi-view renderer for SAM3D outputs
- N views rotating around Y axis (left-right)
- N views rotating around X axis (up-down)
- Combine into GIF
"""
import argparse
import open3d as o3d
import numpy as np
from pathlib import Path
import math

def load_pointcloud(input_path: str):
    """Load point cloud from PLY or GLB"""
    input_p = Path(input_path)
    
    if input_p.suffix.lower() == '.glb':
        import trimesh
        mesh = trimesh.load(str(input_path))
        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
        if mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors[:, :3] / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd = o3d.io.read_point_cloud(str(input_path))
    
    return pcd

def render_view(pcd, front: list, up: list, output_path: str, point_size: float = 3.0):
    """Render a single view"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=512, height=512)
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.65)
    ctr.set_front(front)
    ctr.set_lookat(pcd.get_center())
    ctr.set_up(up)
    
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path, do_render=True)
    vis.destroy_window()

def render_axis_rotation(pcd, axis: str, n_frames: int, output_dir: Path, 
                         basename: str, point_size: float = 3.0):
    """Render 360° rotation around an axis"""
    
    results = []
    
    for i in range(n_frames):
        # Full 360° rotation
        angle = (2 * math.pi * i) / n_frames
        
        if axis == 'y':
            # Rotate around Y (vertical) axis - horizontal rotation
            front = [math.sin(angle), -math.cos(angle), 0.25]
            up = [0, 0, 1]
            label = f"Y-axis {i+1}/{n_frames} ({int(360*i/n_frames)}°)"
        else:  # axis == 'x'
            # Rotate around X (horizontal) axis - vertical rotation
            # Camera orbits in the Y-Z plane
            front = [0, -math.cos(angle), math.sin(angle)]
            # Adjust up vector to stay perpendicular
            up = [0, math.sin(angle), math.cos(angle)]
            label = f"X-axis {i+1}/{n_frames} ({int(360*i/n_frames)}°)"
        
        output_path = output_dir / f"{basename}_{axis}_{i:02d}.png"
        render_view(pcd, front, up, str(output_path), point_size)
        results.append(str(output_path))
        print(f"   ✓ {label}")
    
    return results

def create_gif(image_paths: list, output_path: str, duration: int = 100):
    """Create GIF from images (360° rotation loops naturally)"""
    from PIL import Image
    
    images = [Image.open(p) for p in image_paths]
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )
    print(f"💾 GIF saved: {output_path}")

def render_standard_views(input_path: str, output_dir: str = None, 
                          n_per_axis: int = 10, point_size: float = 3.0,
                          gif_path: str = None, gif_speed: int = 150):
    """Render standard views with rotation around Y and X axes"""
    
    input_p = Path(input_path)
    
    if output_dir:
        output_d = Path(output_dir)
    else:
        output_d = input_p.parent / f"{input_p.stem}_views"
    
    output_d.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Loading: {input_path}")
    pcd = load_pointcloud(input_path)
    n_points = len(pcd.points)
    print(f"   Points: {n_points:,}")
    
    basename = input_p.stem
    all_images = []
    
    # Y-axis rotation (left-right)
    print(f"📷 Rendering Y-axis rotation ({n_per_axis} frames)...")
    y_images = render_axis_rotation(pcd, 'y', n_per_axis, output_d, basename, point_size)
    all_images.extend(y_images)
    
    # X-axis rotation (up-down)
    print(f"📷 Rendering X-axis rotation ({n_per_axis} frames)...")
    x_images = render_axis_rotation(pcd, 'x', n_per_axis, output_d, basename, point_size)
    all_images.extend(x_images)
    
    print(f"✅ Done! {len(all_images)} views saved to: {output_d}")
    
    # Create GIF
    if gif_path:
        print(f"📦 Creating GIF...")
        # Create separate GIFs for each axis, then combine
        y_gif = output_d / f"{basename}_y_rotation.gif"
        x_gif = output_d / f"{basename}_x_rotation.gif"
        
        create_gif(y_images, str(y_gif), gif_speed)
        create_gif(x_images, str(x_gif), gif_speed)
        
        # Also create combined GIF
        create_gif(all_images, gif_path, gif_speed)
    
    return all_images

def main():
    parser = argparse.ArgumentParser(description="Render standard multi-views for SAM3D output")
    parser.add_argument("input", help="Input PLY or GLB file")
    parser.add_argument("-o", "--output-dir", help="Output directory")
    parser.add_argument("-n", "--n-per-axis", type=int, default=10, 
                        help="Number of frames per axis (default 10)")
    parser.add_argument("--point-size", type=float, default=3.0, help="Point size")
    parser.add_argument("--gif", help="Output GIF path")
    parser.add_argument("--gif-speed", type=int, default=150, help="GIF frame duration in ms")
    
    args = parser.parse_args()
    
    render_standard_views(
        args.input, 
        args.output_dir, 
        args.n_per_axis,
        args.point_size,
        args.gif,
        args.gif_speed
    )

if __name__ == "__main__":
    main()
