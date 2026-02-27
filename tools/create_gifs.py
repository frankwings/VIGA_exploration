from PIL import Image
from pathlib import Path
import sys

def create_pingpong_gif(frame_dir, pattern, output_path, duration=80):
    """Create ping-pong GIF: 1,2,3...29,30,29,28...2,1"""
    frames = sorted(Path(frame_dir).glob(pattern))
    
    if not frames:
        print(f"No frames found: {frame_dir}/{pattern}")
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
    # Get output dir from args or use default
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path(r'D:\Projects\ProjectGenesis\GenesisVIGA\docs\test_results_images\bottle_blender_v2')
    
    # Find basename from first file
    files = list(output_dir.glob('*_y_00.png'))
    if files:
        basename = files[0].stem.rsplit('_y_', 1)[0]
    else:
        basename = 'ito_en_green_tea_bottle'
    
    print(f"Output dir: {output_dir}")
    print(f"Basename: {basename}")
    
    # Y-axis GIF
    print("\nCreating Y-axis ping-pong GIF...")
    create_pingpong_gif(output_dir, f'{basename}_y_*.png', output_dir / f'{basename}_y_rotation.gif')
    
    # X-axis GIF
    print("\nCreating X-axis ping-pong GIF...")
    create_pingpong_gif(output_dir, f'{basename}_x_*.png', output_dir / f'{basename}_x_rotation.gif')
    
    print("\nDone!")
