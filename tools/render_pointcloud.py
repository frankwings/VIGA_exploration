#!/usr/bin/env python3
"""
Render point cloud as image using Open3D with multiple camera angles
"""
import argparse
import open3d as o3d
import numpy as np
from pathlib import Path

# Preset camera views: (front_vector, up_vector)
PRESET_VIEWS = {
    'front':  ([0, -1, 0.3], [0, 0, 1]),
    'back':   ([0, 1, 0.3], [0, 0, 1]),
    'left':   ([-1, 0, 0.3], [0, 0, 1]),
    'right':  ([1, 0, 0.3], [0, 0, 1]),
    'top':    ([0, 0, 1], [0, 1, 0]),
    'bottom': ([0, 0, -1], [0, 1, 0]),
    'iso':    ([0.5, -0.5, 0.3], [0, 0, 1]),      # isometric
    'iso2':   ([-0.5, -0.5, 0.3], [0, 0, 1]),     # isometric from other side
    'iso3':   ([0.5, 0.5, 0.3], [0, 0, 1]),       # isometric back
    'iso4':   ([-0.5, 0.5, 0.3], [0, 0, 1]),      # isometric back other side
}

def render_pointcloud(input_path: str, output_path: str, point_size: float = 3.0, 
                      view: str = 'iso', zoom: float = 0.7):
    """Render point cloud and save as image"""
    
    # Load point cloud
    print(f"📂 Loading: {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)
    n_points = len(pcd.points)
    print(f"   Points: {n_points:,}")
    
    # Get view parameters
    if view in PRESET_VIEWS:
        front, up = PRESET_VIEWS[view]
    else:
        # Parse custom view: "x,y,z"
        try:
            front = [float(x) for x in view.split(',')]
            up = [0, 0, 1]
        except:
            print(f"⚠️ Unknown view '{view}', using 'iso'")
            front, up = PRESET_VIEWS['iso']
    
    print(f"   View: {view} (front={front})")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1024, height=768)
    
    # Add point cloud
    vis.add_geometry(pcd)
    
    # Set render options
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    
    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(zoom)
    ctr.set_front(front)
    ctr.set_lookat(pcd.get_center())
    ctr.set_up(up)
    
    # Render
    vis.poll_events()
    vis.update_renderer()
    
    # Save
    vis.capture_screen_image(output_path, do_render=True)
    print(f"💾 Saved: {output_path}")
    
    vis.destroy_window()

def render_all_views(input_path: str, output_dir: str, point_size: float = 3.0, 
                     views: list = None):
    """Render point cloud from multiple views"""
    
    if views is None:
        views = ['front', 'back', 'left', 'right', 'top', 'iso']
    
    input_p = Path(input_path)
    output_d = Path(output_dir)
    output_d.mkdir(parents=True, exist_ok=True)
    
    for view in views:
        output_path = output_d / f"{input_p.stem}_{view}.png"
        render_pointcloud(input_path, str(output_path), point_size, view)

def main():
    parser = argparse.ArgumentParser(description="Render point cloud as image")
    parser.add_argument("input", help="Input PLY file")
    parser.add_argument("-o", "--output", help="Output image path")
    parser.add_argument("--point-size", type=float, default=3.0, help="Point size")
    parser.add_argument("--view", default="iso", 
                        help="Camera view: front/back/left/right/top/bottom/iso/iso2/iso3/iso4 or custom 'x,y,z'")
    parser.add_argument("--zoom", type=float, default=0.7, help="Zoom level (default 0.7)")
    parser.add_argument("--all-views", action="store_true", help="Render all preset views")
    parser.add_argument("--output-dir", help="Output directory for --all-views")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if args.all_views:
        output_dir = args.output_dir or "renders"
        render_all_views(str(input_path), output_dir, args.point_size)
    else:
        if args.output:
            output_path = args.output
        else:
            output_path = str(input_path.with_suffix('.png'))
        
        render_pointcloud(str(input_path), output_path, args.point_size, args.view, args.zoom)

if __name__ == "__main__":
    main()
