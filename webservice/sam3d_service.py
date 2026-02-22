"""
SAM3D Service — wraps SAM segmentation and TRELLIS reconstruction as subprocess calls.

Designed for the GCP VM (genesisforge-gpu, Linux).
Conda envs: sam, sam3d_py311, trellis2.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conda env python paths (Linux — GCP VM)
# ---------------------------------------------------------------------------
HOME = Path.home()
CONDA_BASE = HOME / "miniconda3" / "envs"


def _python(env: str) -> str:
    """Return python path for a conda env."""
    if sys.platform == "win32":
        return str(CONDA_BASE / env / "python.exe")
    return str(CONDA_BASE / env / "bin" / "python")


# SAM3D config (TRELLIS1 checkpoint)
SAM3D_CONFIG = "utils/third_party/sam3d/checkpoints/hf/checkpoints/pipeline.yaml"

# Mask overlay colours (RGBA) — distinct, semi-transparent
MASK_COLORS = [
    (255, 0, 0, 128),     # red
    (0, 200, 0, 128),     # green
    (0, 100, 255, 128),   # blue
    (255, 200, 0, 128),   # yellow
    (200, 0, 255, 128),   # purple
    (0, 220, 220, 128),   # cyan
    (255, 128, 0, 128),   # orange
    (255, 0, 200, 128),   # pink
    (128, 255, 0, 128),   # lime
    (0, 128, 255, 128),   # sky
]


class SAM3DService:
    def __init__(self, project_root: Path, jobs_dir: Path):
        self.project_root = project_root
        self.jobs_dir = jobs_dir

    # ------------------------------------------------------------------
    # SAM Segmentation
    # ------------------------------------------------------------------
    def segment(self, job_id: str, image_path: str) -> dict:
        """Run SAM segmentation, produce masks + overlay image.

        Returns dict with:
            masks: list of {id, name, color, mask_file, bbox}
            overlay: path to overlay PNG
        """
        job_dir = self.jobs_dir / job_id
        masks_npy = job_dir / "all_masks.npy"

        cmd = [
            _python("sam"),
            str(self.project_root / "tools" / "sam3d" / "sam_worker.py"),
            "--image", str(image_path),
            "--out", str(masks_npy),
        ]
        log.info("SAM cmd: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, cwd=str(self.project_root),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=300,
        )
        log_path = job_dir / "sam.log"
        log_path.write_text(result.stdout or "", encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(f"SAM failed (rc={result.returncode}). See {log_path}")

        # Load masks
        all_masks = np.load(str(masks_npy))  # (N, H, W) uint8 0/255

        # Load object names
        names_file = job_dir / "all_masks_object_names.json"
        if names_file.exists():
            names_data = json.loads(names_file.read_text(encoding="utf-8"))
            obj_names = names_data.get("object_names", [])
        else:
            obj_names = [f"object_{i}" for i in range(len(all_masks))]

        # Build mask metadata + overlay
        input_img = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", input_img.size, (0, 0, 0, 0))

        masks_meta = []
        for i, mask in enumerate(all_masks):
            name = obj_names[i] if i < len(obj_names) else f"object_{i}"
            color = MASK_COLORS[i % len(MASK_COLORS)]

            # Save individual mask
            mask_file = job_dir / f"{name}.npy"
            np.save(str(mask_file), mask)

            # Compute bounding box
            ys, xs = np.where(mask > 127)
            if len(ys) == 0:
                continue
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

            # Compute centroid
            cx, cy = int(xs.mean()), int(ys.mean())

            masks_meta.append({
                "id": i,
                "name": name,
                "color": list(color[:3]),
                "mask_file": str(mask_file),
                "bbox": bbox,
                "centroid": [cx, cy],
            })

            # Paint overlay
            colored = np.zeros((*mask.shape, 4), dtype=np.uint8)
            m = mask > 127
            colored[m] = color
            overlay = Image.alpha_composite(overlay, Image.fromarray(colored, "RGBA"))

        # Composite: input + overlay
        composite = Image.alpha_composite(input_img, overlay)
        overlay_path = job_dir / "overlay.png"
        composite.save(str(overlay_path))

        # Also save a mask-only overlay (for click detection)
        mask_map = np.full(all_masks.shape[1:], -1, dtype=np.int16)
        for i in range(len(all_masks) - 1, -1, -1):
            mask_map[all_masks[i] > 127] = i
        mask_map_path = job_dir / "mask_map.npy"
        np.save(str(mask_map_path), mask_map)

        # Save mask_map as a JSON-friendly lookup (indexed PNG)
        # For frontend: we'll transmit mask_id per pixel as a flat image
        # Encode mask_map as a single-channel PNG (mask_id + 1, 0 = background)
        mask_map_img = (mask_map.astype(np.int16) + 1).clip(0, 255).astype(np.uint8)
        Image.fromarray(mask_map_img, "L").save(str(job_dir / "mask_map.png"))

        log.info("Segmented %d masks for job %s", len(masks_meta), job_id)
        return {"masks": masks_meta, "overlay": str(overlay_path)}

    # ------------------------------------------------------------------
    # TRELLIS Reconstruction
    # ------------------------------------------------------------------
    def reconstruct(self, job_id: str, mask_id: int, image_path: str,
                    mask_info: dict, trellis_version: int) -> str:
        """Run TRELLIS reconstruction for one mask. Returns GLB path."""
        job_dir = self.jobs_dir / job_id
        name = mask_info["name"]

        if trellis_version == 1:
            return self._reconstruct_trellis1(job_dir, name, image_path, mask_info)
        else:
            return self._reconstruct_trellis2(job_dir, name, image_path, mask_info)

    def _reconstruct_trellis1(self, job_dir: Path, name: str,
                               image_path: str, mask_info: dict) -> str:
        """TRELLIS1 single-object reconstruction."""
        glb_path = job_dir / f"{name}.glb"
        info_path = job_dir / f"{name}_info.json"

        cmd = [
            _python("sam3d_py311"),
            str(self.project_root / "tools" / "sam3d" / "sam3d_worker.py"),
            "--image", str(image_path),
            "--mask", mask_info["mask_file"],
            "--config", str(self.project_root / SAM3D_CONFIG),
            "--glb", str(glb_path),
            "--info", str(info_path),
            "--scene-image", str(image_path),
        ]
        log.info("TRELLIS1 cmd: %s", " ".join(cmd))

        env = os.environ.copy()
        env["LIDRA_SKIP_INIT"] = "1"

        result = subprocess.run(
            cmd, cwd=str(self.project_root),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=1800, env=env,
        )
        log_path = job_dir / f"{name}_trellis1.log"
        log_path.write_text(result.stdout or "", encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(f"TRELLIS1 failed for {name} (rc={result.returncode}). See {log_path}")

        return str(glb_path)

    def _reconstruct_trellis2(self, job_dir: Path, name: str,
                               image_path: str, mask_info: dict) -> str:
        """TRELLIS2 reconstruction: trellis2_worker → pose_align_worker."""
        # Step 1: Create masked RGBA image for TRELLIS2
        mask = np.load(mask_info["mask_file"])
        img = Image.open(image_path).convert("RGBA")
        img_arr = np.array(img)
        img_arr[mask < 128, 3] = 0  # transparent where mask is 0
        masked_path = job_dir / f"{name}_masked.png"
        Image.fromarray(img_arr, "RGBA").save(str(masked_path))

        # Step 2: TRELLIS2 reconstruction
        pbr_glb = job_dir / f"{name}_pbr.glb"
        mesh_npz = job_dir / f"{name}_mesh.npz"

        t2_manifest = {
            "objects": [{
                "name": name,
                "image": str(masked_path),
                "mask": mask_info["mask_file"],
                "glb": str(pbr_glb),
                "mesh": str(mesh_npz),
            }]
        }
        t2_manifest_path = job_dir / f"{name}_t2_manifest.json"
        t2_manifest_path.write_text(json.dumps(t2_manifest, indent=2), encoding="utf-8")

        cmd = [
            _python("trellis2"),
            str(self.project_root / "tools" / "sam3d" / "trellis2_worker.py"),
            "--manifest", str(t2_manifest_path),
        ]
        log.info("TRELLIS2 cmd: %s", " ".join(cmd))

        result = subprocess.run(
            cmd, cwd=str(self.project_root),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=1800,
        )
        log_path = job_dir / f"{name}_trellis2.log"
        log_path.write_text(result.stdout or "", encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(f"TRELLIS2 failed for {name}. See {log_path}")

        # Step 3: Pose alignment
        aligned_glb = job_dir / f"{name}_pbr_aligned.glb"
        info_path = job_dir / f"{name}_info.json"

        pose_manifest = {
            "scene_image": str(image_path),
            "objects": [{
                "name": name,
                "mesh": str(mesh_npz),
                "mask": mask_info["mask_file"],
                "glb": str(job_dir / f"{name}_aligned.glb"),
                "pbr_glb": str(pbr_glb),
                "aligned_pbr": str(aligned_glb),
                "info": str(info_path),
            }]
        }
        pose_manifest_path = job_dir / f"{name}_pose_manifest.json"
        pose_manifest_path.write_text(json.dumps(pose_manifest, indent=2), encoding="utf-8")

        env = os.environ.copy()
        env["LIDRA_SKIP_INIT"] = "1"

        cmd = [
            _python("sam3d_py311"),
            str(self.project_root / "tools" / "sam3d" / "pose_align_worker.py"),
            "--manifest", str(pose_manifest_path),
        ]
        log.info("Pose align cmd: %s", " ".join(cmd))

        result = subprocess.run(
            cmd, cwd=str(self.project_root),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=600, env=env,
        )
        pose_log = job_dir / f"{name}_pose_align.log"
        pose_log.write_text(result.stdout or "", encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(f"Pose alignment failed for {name}. See {pose_log}")

        # Return the PBR-aligned GLB (best quality)
        if aligned_glb.exists():
            return str(aligned_glb)
        return str(pbr_glb)
