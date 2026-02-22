"""TRELLIS2 3D Reconstruction Worker.

Generates 3D meshes from masked images using TRELLIS.2-4B.
Runs in the `trellis2` conda environment.

Batch mode: loads pipeline once, processes multiple objects via JSON manifest.

Manifest format:
{
  "objects": [
    {
      "name": "object_name",
      "image": "path/to/masked_rgba.png",
      "glb": "path/to/output.glb",
      "mesh": "path/to/output_mesh.npz"    // raw mesh for pose alignment
    },
    ...
  ]
}
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

# TRELLIS.2 is not pip-installed; add to sys.path
TRELLIS2_ROOT = os.path.join(
    os.path.expanduser("~"), "workspace", "TRELLIS.2"
)
if os.path.isdir(TRELLIS2_ROOT):
    sys.path.insert(0, TRELLIS2_ROOT)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_masked_image(image_path, mask_path=None):
    """Load image and optionally apply mask, returning RGBA PIL Image.

    If mask_path is provided, applies the mask as alpha channel.
    If image is already RGBA (mask baked in), uses it directly.
    """
    img = Image.open(image_path).convert("RGBA")

    if mask_path is not None:
        mask = np.load(mask_path)
        mask = (mask > 0).astype(np.uint8) * 255
        if mask.ndim == 3:
            mask = mask[..., 0]
        img_np = np.array(img)
        img_np[..., 3] = mask
        img = Image.fromarray(img_np)

    return img


def process_single_object(pipeline, obj, decimation_target=100000, texture_size=2048):
    """Generate 3D mesh for a single object using TRELLIS2."""
    import o_voxel

    name = obj["name"]
    t0 = time.time()

    # Load image
    image = load_masked_image(obj["image"], obj.get("mask"))
    print(f"[T2] {name}: image {image.size}, mode={image.mode}", flush=True)

    # Run TRELLIS2 inference
    t_inf = time.time()
    mesh_result = pipeline.run(image, seed=42)[0]
    t_inf_end = time.time()
    print(f"[T2] {name}: inference {t_inf_end - t_inf:.1f}s, "
          f"verts={mesh_result.vertices.shape[0]}, faces={mesh_result.faces.shape[0]}",
          flush=True)

    # Save raw mesh data (Z-up canonical frame, before any coordinate transform)
    # This is used by pose_align_worker.py for layout_post_optimization
    mesh_npz_path = obj.get("mesh")
    if mesh_npz_path:
        os.makedirs(os.path.dirname(mesh_npz_path), exist_ok=True)
        raw_verts = mesh_result.vertices.cpu().numpy().astype(np.float32)
        raw_faces = mesh_result.faces.cpu().numpy().astype(np.int32)
        np.savez(mesh_npz_path, vertices=raw_verts, faces=raw_faces)
        print(f"[T2] {name}: saved raw mesh {raw_verts.shape[0]} verts → {mesh_npz_path}",
              flush=True)

    # Export PBR-textured GLB via o_voxel
    glb_path = obj["glb"]
    os.makedirs(os.path.dirname(glb_path), exist_ok=True)

    t_glb = time.time()
    # Simplify to nvdiffrast limit before GLB export
    mesh_result.simplify(16777216)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh_result.vertices,
        faces=mesh_result.faces,
        attr_volume=mesh_result.attrs,
        coords=mesh_result.coords,
        attr_layout=mesh_result.layout,
        voxel_size=mesh_result.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        verbose=False,
    )
    glb.export(glb_path, extension_webp=True)
    t_glb_end = time.time()

    glb_mb = os.path.getsize(glb_path) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"[T2] {name}: GLB export {t_glb_end - t_glb:.1f}s, "
          f"{glb_mb:.1f}MB → {glb_path}", flush=True)
    print(f"[T2] {name}: total {elapsed:.1f}s", flush=True)

    return {
        "name": name,
        "glb_path": glb_path,
        "mesh_path": mesh_npz_path,
        "inference_time": t_inf_end - t_inf,
        "glb_export_time": t_glb_end - t_glb,
        "total_time": elapsed,
    }


def main():
    p = argparse.ArgumentParser(description="TRELLIS2 batch 3D reconstruction worker")
    p.add_argument("--manifest", required=True, help="Path to JSON manifest file")
    p.add_argument("--model", default="microsoft/TRELLIS.2-4B",
                   help="HuggingFace model name (default: microsoft/TRELLIS.2-4B)")
    p.add_argument("--decimation-target", type=int, default=100000,
                   help="GLB decimation target (default: 100000)")
    p.add_argument("--texture-size", type=int, default=2048,
                   help="GLB texture resolution (default: 2048)")
    args = p.parse_args()

    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    objects = manifest["objects"]
    print(f"[T2] Loading TRELLIS2 pipeline: {args.model}...", flush=True)
    t_load = time.time()

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    load_time = time.time() - t_load
    print(f"[T2] Pipeline loaded in {load_time:.1f}s", flush=True)

    # Process each object
    results = {}
    total_start = time.time()
    for i, obj in enumerate(objects):
        name = obj["name"]
        glb_path = obj["glb"]

        # Skip if already completed
        if os.path.exists(glb_path):
            mesh_path = obj.get("mesh")
            if mesh_path is None or os.path.exists(mesh_path):
                print(f"[T2] {name}: already exists, skipping", flush=True)
                results[name] = {"name": name, "glb_path": glb_path, "skipped": True}
                continue

        print(f"[T2] Processing {name} ({i+1}/{len(objects)})...", flush=True)
        try:
            info = process_single_object(
                pipeline, obj,
                decimation_target=args.decimation_target,
                texture_size=args.texture_size,
            )
            results[name] = info
        except Exception as e:
            import traceback
            print(f"[T2] {name}: FAILED - {e}", flush=True)
            traceback.print_exc()
            results[name] = {"name": name, "error": str(e)}

    total_time = time.time() - total_start
    success = sum(1 for v in results.values() if "error" not in v)
    print(f"\n[T2] Completed: {success}/{len(objects)} in {total_time:.1f}s "
          f"(model load: {load_time:.1f}s)", flush=True)

    # Write batch summary
    manifest_dir = os.path.dirname(args.manifest)
    if manifest_dir:
        summary_path = os.path.join(manifest_dir, "trellis2_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                "model": args.model,
                "model_load_time": load_time,
                "total_time": total_time,
                "objects": results,
            }, f, indent=2)


if __name__ == "__main__":
    main()
