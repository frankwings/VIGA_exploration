#!/usr/bin/env python3
"""
Process ALL masks through SAM3D pipeline and generate rotation GIFs.

For each mask:
1. Run SAM3D inference -> GLB mesh
2. Run Blender -> rotation frames -> GIF

Expected time: ~9 min per mask x 6 masks = ~54 min total for SAM3D
Plus Blender rendering time.
"""
import os
import sys
import json
import time
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

# Configuration
VIGA_ROOT = Path(r"D:\Projects\ProjectGenesis\GenesisVIGA")
SAM3D_PATH = VIGA_ROOT / "utils" / "third_party" / "sam3d"
CONFIG_PATH = SAM3D_PATH / "checkpoints" / "hf" / "checkpoints" / "pipeline.yaml"

INPUT_IMAGE = VIGA_ROOT / "docs" / "test_results_images" / "01_greentea_input.jpg"
MASKS_DIR = VIGA_ROOT / "output" / "test_sam"
ALL_MASKS_FILE = MASKS_DIR / "all_masks.npy"
NAMES_FILE = MASKS_DIR / "all_masks_object_names.json"

OUTPUT_DIR = VIGA_ROOT / "output" / "all_masks_sam3d"
RESULTS_DIR = VIGA_ROOT / "docs" / "test_results_images" / "all_masks_results"

BLENDER_SCRIPT = VIGA_ROOT / "tools" / "blender_render_rotation.py"
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"

N_FRAMES = 30  # Frames per rotation axis

# Setup paths
sys.path.insert(0, str(SAM3D_PATH))
sys.path.insert(0, str(SAM3D_PATH / "notebook"))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def log(msg):
    """Print with timestamp"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def run_sam3d_for_mask(image_path, mask, obj_name, output_dir):
    """
    Run SAM3D inference for a single mask.
    Returns path to generated GLB or None on failure.
    """
    try:
        import numpy as np
        from inference import Inference, load_image
        
        log(f"  Loading SAM3D inference engine...")
        inference = Inference(str(CONFIG_PATH), compile=False)
        
        log(f"  Loading image: {image_path}")
        image = load_image(str(image_path))
        
        # Mask should be True where the object is
        # Our masks store 0=object, 255=background, so invert
        bool_mask = mask == 0
        coverage = 100 * np.sum(bool_mask) / bool_mask.size
        log(f"  Mask coverage: {coverage:.1f}%")
        
        if coverage < 0.1:
            log(f"  [WARN] Mask too small, skipping...")
            return None
        
        log(f"  Running SAM3D inference (this takes ~9 min)...")
        start = time.time()
        output = inference(image, bool_mask, seed=42)
        elapsed = time.time() - start
        log(f"  [OK] Inference done in {elapsed/60:.1f} min")
        
        # Save GLB
        glb_path = output_dir / f"{obj_name}.glb"
        output["glb"].export(str(glb_path))
        log(f"  [OK] Saved: {glb_path}")
        
        return glb_path
        
    except Exception as e:
        log(f"  [FAIL] SAM3D failed: {e}")
        traceback.print_exc()
        return None


def render_rotation_gif(glb_path, output_dir, obj_name):
    """
    Use Blender to render rotation frames and create GIFs.
    """
    try:
        frames_dir = output_dir / f"{obj_name}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        log(f"  Running Blender render...")
        cmd = [
            BLENDER_EXE,
            "--background",
            "--python", str(BLENDER_SCRIPT),
            "--",
            str(glb_path),
            str(frames_dir),
            "--frames", str(N_FRAMES)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            log(f"  [WARN] Blender stderr: {result.stderr[:500]}")
        
        # Create GIFs using pillow
        create_gif(frames_dir, output_dir, obj_name, "x")
        create_gif(frames_dir, output_dir, obj_name, "y")
        
        log(f"  [OK] Blender render complete")
        return True
        
    except Exception as e:
        log(f"  [FAIL] Blender render failed: {e}")
        traceback.print_exc()
        return False


def create_gif(frames_dir, output_dir, obj_name, axis):
    """Create GIF from rendered frames"""
    try:
        from PIL import Image
        import glob
        
        pattern = str(frames_dir / f"{obj_name}_{axis}_*.png")
        frames = sorted(glob.glob(pattern))
        
        if not frames:
            log(f"    No frames found for {axis}-axis")
            return
        
        images = [Image.open(f) for f in frames]
        gif_path = output_dir / f"{obj_name}_{axis}_rotation.gif"
        
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0
        )
        log(f"    [OK] Created: {gif_path.name}")
        
    except Exception as e:
        log(f"    [WARN] GIF creation failed: {e}")


def main():
    import numpy as np
    
    log("=" * 60)
    log("SAM3D All Masks Processing Pipeline")
    log("=" * 60)
    
    # Load mask names
    with open(NAMES_FILE) as f:
        names_data = json.load(f)
    object_names = names_data["object_names"]
    
    log(f"Found {len(object_names)} masks to process:")
    for i, name in enumerate(object_names):
        log(f"  {i+1}. {name}")
    
    # Load all masks
    all_masks = np.load(ALL_MASKS_FILE)
    log(f"Masks array shape: {all_masks.shape}")
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each mask
    results = []
    total_start = time.time()
    
    for i, obj_name in enumerate(object_names):
        log("")
        log(f"{'='*60}")
        log(f"Processing [{i+1}/{len(object_names)}]: {obj_name}")
        log(f"{'='*60}")
        
        mask = all_masks[i]
        
        # Step 1: SAM3D inference
        glb_path = run_sam3d_for_mask(INPUT_IMAGE, mask, obj_name, OUTPUT_DIR)
        
        if glb_path and glb_path.exists():
            # Step 2: Blender render
            render_success = render_rotation_gif(glb_path, RESULTS_DIR, obj_name)
            results.append({
                "name": obj_name,
                "glb": str(glb_path),
                "success": render_success
            })
        else:
            results.append({
                "name": obj_name,
                "glb": None,
                "success": False
            })
    
    # Summary
    total_time = time.time() - total_start
    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"Total time: {total_time/60:.1f} minutes")
    log("")
    
    success_count = sum(1 for r in results if r["success"])
    log(f"Success: {success_count}/{len(results)}")
    
    for r in results:
        status = "[OK]" if r["success"] else "[FAIL]"
        log(f"  {status} {r['name']}")
    
    # Save results
    results_file = OUTPUT_DIR / "processing_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved: {results_file}")
    
    log("\n[DONE] Pipeline complete!")


if __name__ == "__main__":
    main()
