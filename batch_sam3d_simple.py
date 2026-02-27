"""Batch SAM3D 3D Reconstruction Script - Simple Version

Executes 3D reconstruction for multiple segmented objects using VIGA pipeline.
"""

import os
import subprocess
import time
import json
from pathlib import Path

# Configuration
INPUT_DIR = "output/test/greentea/sam_init"
OUTPUT_DIR = "output/sam3d_reconstruction_batch"
CONFIG_FILE = "utils/third_party/sam3d/checkpoints/hf/checkpoints/pipeline.yaml"

# Objects to process
OBJECTS = [
    "green_tea_bottle",
    "ito_en_bottle", 
    "alienware_keyboard",
    "headphones",
    "envelope"
]

def setup_environment():
    """Setup the environment variables for VIGA."""
    os.environ["PYTHONPATH"] = str(Path(__file__).parent.absolute())
    print(f"Set PYTHONPATH to: {os.environ['PYTHONPATH']}")

def run_sam3d_for_object(obj_name):
    """Run SAM3D reconstruction for a single object."""
    print(f"\n{'='*60}")
    print(f"Processing: {obj_name}")
    print(f"{'='*60}")
    
    # Paths
    image_path = f"{INPUT_DIR}/{obj_name}.png"
    mask_path = f"{INPUT_DIR}/{obj_name}.npy" 
    glb_path = f"{OUTPUT_DIR}/{obj_name}.glb"
    log_path = f"{OUTPUT_DIR}/{obj_name}_sam3d.log"
    
    # Check input files exist
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return False
    if not os.path.exists(mask_path):
        print(f"ERROR: Mask not found: {mask_path}")
        return False
        
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Build command
    cmd = [
        "conda", "run", "-n", "sam3d_py311",
        "python", "tools/sam3d/sam3d_worker.py",
        "--image", image_path,
        "--mask", mask_path,
        "--config", CONFIG_FILE,
        "--glb", glb_path
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Logging to: {log_path}")
    
    # Run command with logging
    start_time = time.time()
    
    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            process = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.dirname(__file__)
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if process.returncode == 0:
            if os.path.exists(glb_path):
                file_size = os.path.getsize(glb_path) / (1024*1024)  # MB
                print(f"SUCCESS! GLB: {glb_path} ({file_size:.1f}MB)")
                print(f"Duration: {duration:.1f}s")
                return True
            else:
                print(f"WARNING: Process completed but GLB not found: {glb_path}")
                return False
        else:
            print(f"FAILED with exit code: {process.returncode}")
            print(f"Check log: {log_path}")
            return False
            
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False

def main():
    """Main function to process all objects."""
    print("Starting Batch SAM3D 3D Reconstruction")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Objects: {', '.join(OBJECTS)}")
    
    setup_environment()
    
    results = {}
    total_start = time.time()
    
    for obj_name in OBJECTS:
        print(f"\nProcessing {obj_name}...")
        success = run_sam3d_for_object(obj_name)
        results[obj_name] = success
        
        if success:
            print(f"SUCCESS: {obj_name} completed successfully")
        else:
            print(f"FAILED: {obj_name} failed")
    
    # Summary
    total_time = time.time() - total_start
    successful = sum(results.values())
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{total}")
    print(f"Failed: {total-successful}/{total}")
    print(f"Total Time: {total_time:.1f}s")
    
    # Save results
    results_file = f"{OUTPUT_DIR}/batch_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': total_time,
            'results': results,
            'summary': {
                'successful': successful,
                'failed': total-successful,
                'total': total
            }
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    if successful == total:
        print("All objects processed successfully!")
    else:
        print("Some objects failed. Check individual logs for details.")

if __name__ == "__main__":
    main()