"""SS Pose Estimation Worker.

Loads the SAM3D (Meta) pipeline and runs ONLY Stage 1 (Sparse Structure) to
predict rotation/translation/scale for objects that lack this information.
This is primarily for TRELLIS2 objects which don't have their own pose prediction.

The SS model jointly predicts voxel occupancy AND 6D rotation + translation + scale
from a masked image + MoGe pointmap. We use stage1_only=True to skip the expensive
SLAT and decoder stages.

Manifest format:
{
    "config": "path/to/sam3d/pipeline.yaml",
    "scene_image": "path/to/scene.jpg",
    "pointmap": "path/to/pointmap.npz",
    "objects": [
        {
            "name": "object_name",
            "image": "path/to/scene.jpg",
            "mask": "path/to/mask.npy",
            "checkpoint": "path/to/output_checkpoint.npz"
        }, ...
    ]
}

Conda env: sam3d_py311 (Python 3.11, same as sam3d_batch_worker)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAM3D_ROOT = os.path.join(ROOT, "utils", "third_party", "sam3d")
sys.path.insert(0, SAM3D_ROOT)
sys.path.insert(0, os.path.join(SAM3D_ROOT, "notebook"))

# Skip sam3d_objects.init module (not needed, causes ImportError)
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

if "CONDA_PREFIX" not in os.environ:
    python_bin = sys.executable
    conda_env = os.path.dirname(os.path.dirname(python_bin))
    os.environ["CONDA_PREFIX"] = conda_env


def load_scene_image(path):
    """Load scene image as RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def make_masked_rgba(scene_rgb, mask_npy_path):
    """Create RGBA image with object mask as alpha channel.

    The SS model needs to see which object to predict pose for via the alpha mask.
    The full scene RGB provides context for depth estimation.
    If the scene image and mask have different resolutions, the scene is resized
    to match the mask (the SS model pads to square and resizes to 518x518 anyway).
    """
    mask = np.load(mask_npy_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    # Resize scene to match mask dimensions if needed
    h_mask, w_mask = mask_uint8.shape
    h_scene, w_scene = scene_rgb.shape[:2]
    if h_scene != h_mask or w_scene != w_mask:
        from PIL import Image as _PILImage
        scene_rgb = np.array(
            _PILImage.fromarray(scene_rgb).resize((w_mask, h_mask), _PILImage.LANCZOS)
        )
        print(f"[SS_POSE] Resized scene {w_scene}x{h_scene} → {w_mask}x{h_mask} to match mask",
              flush=True)

    rgba = np.zeros((scene_rgb.shape[0], scene_rgb.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = scene_rgb
    rgba[..., 3] = mask_uint8
    return rgba


def main():
    p = argparse.ArgumentParser(description="SS Pose Estimation Worker")
    p.add_argument("--manifest", required=True, help="Path to JSON manifest file")
    args = p.parse_args()

    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    config_path = manifest["config"]
    objects = manifest["objects"]
    pointmap_path = manifest["pointmap"]
    scene_image_path = manifest["scene_image"]

    # Load pointmap from Module 3 (monodepth)
    # Saved as (3, H, W) float32 in PyTorch3D camera space.
    # Pipeline expects (H, W, 3) tensor.
    pm_data = np.load(pointmap_path)
    pointmap_3hw = pm_data["pointmap"]  # (3, H, W)
    pointmap_hw3 = torch.from_numpy(
        pointmap_3hw.transpose(1, 2, 0).copy()
    ).float()
    print(f"[SS_POSE] Pointmap loaded: shape={pointmap_3hw.shape}", flush=True)

    # Load scene image
    scene_rgb = load_scene_image(scene_image_path)
    print(f"[SS_POSE] Scene image: {scene_rgb.shape}", flush=True)

    # Load SAM3D pipeline (loads all models including SLAT/decoders — unused but
    # needed for the Inference wrapper; stage1_only=True skips them at runtime)
    from inference import Inference
    print(f"[SS_POSE] Loading SAM3D pipeline: {config_path}...", flush=True)
    t_load = time.time()
    inference = Inference(config_path, compile=False)
    pipeline = inference._pipeline
    load_time = time.time() - t_load
    print(f"[SS_POSE] Pipeline loaded in {load_time:.1f}s", flush=True)

    # Process each object
    results = {}
    total_start = time.time()

    for i, obj in enumerate(objects):
        name = obj["name"]
        ckpt_path = obj["checkpoint"]

        # Skip if checkpoint already exists with valid pose data
        if os.path.exists(ckpt_path):
            try:
                ckpt = np.load(ckpt_path)
                if all(k in ckpt for k in ("rotation", "translation", "scale")):
                    print(f"[SS_POSE] {name}: already done, skipping", flush=True)
                    results[name] = {"status": "skipped"}
                    continue
            except Exception:
                pass  # file corrupt, recompute

        print(f"[SS_POSE] Processing {name} ({i+1}/{len(objects)})...", flush=True)
        t0 = time.time()

        # Create RGBA image with object mask
        rgba = make_masked_rgba(scene_rgb, obj["mask"])

        try:
            # Run Stage 1 only: SS generator + pose_decoder
            # Passing pointmap skips MoGe (already computed in Module 3).
            # Passing scene_image ensures pointmap uses scene dimensions for pts_color.
            result = pipeline.run(
                rgba,
                mask=None,
                seed=42,
                stage1_only=True,
                with_mesh_postprocess=False,
                with_texture_baking=False,
                with_layout_postprocess=False,
                use_vertex_color=False,
                stage1_inference_steps=None,
                pointmap=pointmap_hw3,
                scene_image=scene_rgb,
            )

            # Extract pose predictions
            rotation = result["rotation"].cpu().numpy()
            translation = result["translation"].cpu().numpy()
            scale = result["scale"].cpu().numpy()

            # Save checkpoint in same format as sam3d_batch_worker
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            np.savez(
                ckpt_path,
                rotation=rotation.astype(np.float32),
                translation=translation.astype(np.float32),
                scale=scale.astype(np.float32),
            )

            elapsed = time.time() - t0
            print(
                f"[SS_POSE] {name}: done in {elapsed:.1f}s\n"
                f"  rotation:    {rotation.flatten().tolist()}\n"
                f"  translation: {translation.flatten().tolist()}\n"
                f"  scale:       {scale.flatten().tolist()}",
                flush=True,
            )
            results[name] = {
                "status": "ok",
                "time": elapsed,
                "checkpoint": ckpt_path,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[SS_POSE] {name}: FAILED - {e}", flush=True)
            results[name] = {"status": "error", "error": str(e)}

    total_time = time.time() - total_start
    ok = sum(1 for v in results.values() if v.get("status") in ("ok", "skipped"))
    print(
        f"\n[SS_POSE] Done: {ok}/{len(objects)} in {total_time:.1f}s "
        f"(model load: {load_time:.1f}s)",
        flush=True,
    )

    # Write summary
    manifest_dir = os.path.dirname(args.manifest)
    if manifest_dir:
        summary_path = os.path.join(manifest_dir, "ss_pose_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                "model_load_time": load_time,
                "total_time": total_time,
                "objects": results,
            }, f, indent=2)


if __name__ == "__main__":
    main()
