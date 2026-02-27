"""Quick test of TRELLIS.2 pipeline on L4 GPU."""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline

# Load pipeline
print("Loading pipeline...")
t0 = time.time()
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()
t1 = time.time()
print(f"Pipeline loaded in {t1-t0:.1f}s")

# Run on a test image
image = Image.open("assets/example_image/T.png")
print(f"Input image: {image.size}")

print("Running inference...")
t2 = time.time()
mesh = pipeline.run(image)[0]
t3 = time.time()
print(f"Inference done in {t3-t2:.1f}s")
print(f"Mesh vertices: {mesh.vertices.shape[0]}, faces: {mesh.faces.shape[0]}")

# Export GLB
import o_voxel
mesh.simplify(16777216)
glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices,
    faces=mesh.faces,
    attr_volume=mesh.attrs,
    coords=mesh.coords,
    attr_layout=mesh.layout,
    voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=100000,
    texture_size=2048,
    remesh=True,
    verbose=True
)
out_path = "/tmp/trellis2_test.glb"
glb.export(out_path, extension_webp=True)
t4 = time.time()
print(f"GLB export done in {t4-t3:.1f}s")
print(f"Total pipeline: {t4-t0:.1f}s")
sz = os.path.getsize(out_path)
print(f"GLB size: {sz / 1024 / 1024:.1f} MB")
