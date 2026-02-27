import sys
import os
sys.path.append(r"D:\Projects\ProjectGenesis\TRELLIS.2")

try:
    print("Attempting to import trellis2...")
    import trellis2
    print(f"trellis2 imported: {trellis2}")
    
    print("Attempting to import trellis2.pipelines...")
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    print("Trellis2ImageTo3DPipeline imported successfully")
except Exception as e:
    print(f"Import Failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
