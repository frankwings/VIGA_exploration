"""Pose Alignment Worker — MoGe depth + layout_post_optimization.

Takes canonical-frame meshes (from TRELLIS2 or any source) and aligns them
to the scene using MoGe depth estimation and SAM3D's layout_post_optimization.

Runs in the `sam3d_py311` conda environment (needs MoGe, PyTorch3D, Open3D).

Manifest format:
{
  "scene_image": "path/to/scene.jpg",
  "objects": [
    {
      "name": "object_name",
      "mesh": "path/to/canonical_mesh.npz",   // raw vertices+faces in Z-up frame
      "mask": "path/to/mask.npy",              // SAM binary mask
      "glb": "path/to/output_aligned.glb",     // output aligned GLB
      "pbr_glb": "path/to/canonical_pbr.glb",  // optional: PBR GLB to also align
      "info": "path/to/info.json"              // output pose info
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
import torch.nn.functional as F
import trimesh

# SAM3D submodule paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAM3D_ROOT = os.path.join(ROOT, "utils", "third_party", "sam3d")
sys.path.insert(0, SAM3D_ROOT)

# Skip sam3d_objects.init module (not needed, causes ImportError)
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

if "CONDA_PREFIX" not in os.environ:
    python_bin = sys.executable
    conda_env = os.path.dirname(os.path.dirname(python_bin))
    os.environ["CONDA_PREFIX"] = conda_env


# Pre-transform: Z-up (TRELLIS canonical) → Y-up (PyTorch3D camera space)
# Same transform used by both TRELLIS1 and TRELLIS2
R_zup_to_yup = torch.tensor(
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32
)


def load_moge_model(device="cuda"):
    """Load MoGe depth estimation model standalone."""
    from moge.model.v1 import MoGeModel
    from sam3d_objects.pipeline.depth_models.moge import MoGe

    print("[POSE] Loading MoGe model: Ruicheng/moge-vitl...", flush=True)
    t0 = time.time()
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl")
    depth_model = MoGe(model=moge_model, device=device)
    print(f"[POSE] MoGe loaded in {time.time() - t0:.1f}s", flush=True)
    return depth_model


def compute_pointmap(depth_model, image_np, device="cuda", dtype=torch.float16):
    """Run MoGe depth estimation and convert to PyTorch3D camera space.

    Replicates InferencePipelinePointMap.compute_pointmap() without loading
    the full TRELLIS pipeline.
    """
    from pytorch3d.transforms import Transform3d
    from pytorch3d.renderer import look_at_view_transform
    from sam3d_objects.data.dataset.tdfy.transforms_3d import DecomposedTransform
    from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap

    # Convert image to float [0, 1] and to tensor (3, H, W)
    if image_np.dtype == np.uint8:
        image_float = image_np.astype(np.float32) / 255.0
    else:
        image_float = image_np.astype(np.float32)
    if image_float.ndim == 2:
        image_float = np.stack([image_float] * 3, axis=-1)
    if image_float.shape[-1] == 4:
        image_float = image_float[..., :3]
    image_tensor = torch.from_numpy(image_float).permute(2, 0, 1).contiguous()  # (3, H, W)

    # Run MoGe
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = depth_model(image_tensor)
    pointmaps = output["pointmaps"]  # (H, W, 3) in R3 camera space

    # Convert R3 camera space → PyTorch3D camera space
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )
    cam_transform = Transform3d().rotate(r3_to_p3d_R).to(device)
    points_tensor = cam_transform.transform_points(pointmaps)

    # Convert to (3, H, W)
    points_tensor = points_tensor.permute(2, 0, 1)

    # Infer intrinsics
    intrinsics = output.get("intrinsics", None)
    if intrinsics is None:
        intrinsics_result = infer_intrinsics_from_pointmap(
            points_tensor.permute(1, 2, 0), device=device
        )
        intrinsics = intrinsics_result["intrinsics"]

    return {
        "pointmap": points_tensor,
        "intrinsics": intrinsics,
        "pts_color": image_tensor,
    }


def create_initial_pose(mesh_vertices, pointmap, mask, device="cuda"):
    """Compute rough initial pose from MoGe pointmap and mesh bounds.

    Returns identity rotation, height-based scale, and centroid translation.
    These will be refined by layout_post_optimization.
    """
    # Apply R_zup_to_yup to canonical vertices
    verts_yup = mesh_vertices @ R_zup_to_yup.to(device)

    # Mesh extent in Y-up space
    mesh_min = verts_yup.min(dim=0).values
    mesh_max = verts_yup.max(dim=0).values
    mesh_extent = mesh_max - mesh_min  # (3,)
    # Use max extent (not just Y) so flat objects don't get inflated scale.
    # For flat objects like newspaper or chair_cover, Y is the thin dimension
    # (0.11-0.15) while X or Z can be ~1.0, causing scale to blow up by 7-30x.
    mesh_max_dim = mesh_extent.max().item()

    # Object points from pointmap where mask is True
    pm = pointmap.permute(1, 2, 0)  # (H, W, 3)
    mask_bool = mask > 0
    if mask_bool.ndim == 3:
        mask_bool = mask_bool[..., 0]
    mask_bool = mask_bool.to(device)

    obj_points = pm[mask_bool]  # (N, 3)
    if obj_points.shape[0] == 0:
        print("[POSE] WARNING: no points in mask, using identity pose", flush=True)
        return (
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),  # quaternion (wxyz)
            torch.zeros(1, 3, device=device),  # translation
            torch.ones(1, 3, device=device),  # scale
        )

    # Remove NaN
    valid = ~torch.isnan(obj_points).any(dim=-1)
    obj_points = obj_points[valid]
    if obj_points.shape[0] == 0:
        print("[POSE] WARNING: all points are NaN, using identity pose", flush=True)
        return (
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
            torch.zeros(1, 3, device=device),
            torch.ones(1, 3, device=device),
        )

    # Remove depth outliers (beyond 90th percentile)
    depths = obj_points[:, 2]
    p90 = torch.quantile(depths, 0.9)
    obj_points = obj_points[depths <= p90]

    # Object extent in pointmap (XYZ bounding box)
    obj_min = obj_points.min(dim=0).values
    obj_max = obj_points.max(dim=0).values
    obj_extent = obj_max - obj_min
    # Match max extent of mesh to max extent in pointmap — rotation-agnostic.
    obj_max_dim = obj_extent.max().item()

    # Scale: ratio of pointmap max-extent to mesh max-extent
    if mesh_max_dim > 1e-6 and obj_max_dim > 1e-6:
        scale_val = obj_max_dim / mesh_max_dim
    else:
        scale_val = 1.0

    # Translation: centroid of object points (after scaling mesh)
    mesh_center = (mesh_min + mesh_max) / 2.0 * scale_val
    obj_center = (obj_min + obj_max) / 2.0
    translation = obj_center - mesh_center

    # Identity rotation
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scale = torch.tensor([[scale_val, scale_val, scale_val]], device=device)
    translation = translation.unsqueeze(0)

    print(f"[POSE] Initial pose: scale={scale_val:.4f} "
          f"(mesh_max_dim={mesh_max_dim:.3f}, obj_max_dim={obj_max_dim:.3f}), "
          f"T=[{translation[0, 0]:.3f}, {translation[0, 1]:.3f}, {translation[0, 2]:.3f}]",
          flush=True)

    return quaternion, translation, scale


def quaternion_to_matrix(quaternions):
    """Convert quaternions (wxyz) to rotation matrices."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


class Transform3dSimple:
    """Minimal Transform3d using PyTorch3D row-vector convention (points @ M)."""

    def __init__(self, dtype=torch.float32, device="cpu"):
        self.dtype = dtype
        self.device = device
        self._matrix = torch.eye(4, dtype=dtype, device=device)

    def scale(self, s):
        if isinstance(s, (int, float)):
            s = torch.tensor([s, s, s], dtype=self.dtype, device=self.device)
        elif isinstance(s, torch.Tensor) and s.numel() == 1:
            s = s.expand(3)
        S = torch.eye(4, dtype=self.dtype, device=self.device)
        S[0, 0], S[1, 1], S[2, 2] = s[0], s[1], s[2]
        self._matrix = self._matrix @ S
        return self

    def rotate(self, R):
        R4 = torch.eye(4, dtype=self.dtype, device=self.device)
        R4[:3, :3] = R
        self._matrix = self._matrix @ R4
        return self

    def translate(self, x, y, z):
        T = torch.eye(4, dtype=self.dtype, device=self.device)
        T[3, 0], T[3, 1], T[3, 2] = x, y, z
        self._matrix = self._matrix @ T
        return self

    def transform_points(self, points):
        if points.dim() == 2:
            points = points.unsqueeze(0)
        B, N, _ = points.shape
        ones = torch.ones(B, N, 1, dtype=self.dtype, device=self.device)
        points_h = torch.cat([points, ones], dim=-1)
        transformed = points_h @ self._matrix
        return transformed[..., :3]


def transform_and_export_glb(mesh_path, glb_path, quaternion, translation, scale,
                             pbr_glb_path=None, aligned_pbr_path=None,
                             canonical_glb_path=None):
    """Transform mesh vertices from canonical Z-up to aligned PyTorch3D camera space.

    If canonical_glb_path is provided, loads the original textured GLB and applies
    the transform to its vertices — preserving all UV maps, materials, and textures.
    Otherwise falls back to creating a geometry-only GLB from the NPZ.

    Also optionally transforms PBR GLB vertices (which are already in Y-up from to_glb).
    Dense TRELLIS2 meshes (500K–2.3M vertices) are decimated to MAX_FACES_FOR_EXPORT
    to keep GLBs small enough for Blender to import.
    """
    MAX_FACES_FOR_EXPORT = 100000
    device = "cpu"

    data = np.load(mesh_path)
    vertices = torch.tensor(data["vertices"], dtype=torch.float32, device=device)
    faces = data["faces"]

    # Build the S/R/T transform
    vertices_yup = vertices @ R_zup_to_yup.to(device)
    R_mat = quaternion_to_matrix(quaternion.to(device))
    if R_mat.dim() == 3:
        R_mat = R_mat[0]

    s = scale.to(device)
    if s.dim() == 2:
        s = s[0]
    t = translation.to(device)
    if t.dim() == 2:
        t = t[0]

    tfm = Transform3dSimple(dtype=torch.float32, device=device)
    tfm = tfm.scale(s).rotate(R_mat).translate(t[0], t[1], t[2])

    os.makedirs(os.path.dirname(glb_path), exist_ok=True)

    # Try to use original textured GLB for export (preserves UV/materials)
    if canonical_glb_path and os.path.exists(canonical_glb_path):
        try:
            scene = trimesh.load(canonical_glb_path)
            if isinstance(scene, trimesh.Scene):
                for geom_name, geom in scene.geometry.items():
                    if hasattr(geom, 'vertices'):
                        v = torch.tensor(geom.vertices, dtype=torch.float32, device=device)
                        # Canonical GLB is in Z-up; apply R_zup_to_yup then S/R/T
                        v_yup = v @ R_zup_to_yup.to(device)
                        v_aligned = tfm.transform_points(v_yup.unsqueeze(0))[0]
                        geom.vertices = v_aligned.numpy().astype(np.float32)
                scene.export(glb_path)
            else:
                v = torch.tensor(scene.vertices, dtype=torch.float32, device=device)
                v_yup = v @ R_zup_to_yup.to(device)
                v_aligned = tfm.transform_points(v_yup.unsqueeze(0))[0]
                scene.vertices = v_aligned.numpy().astype(np.float32)
                scene.export(glb_path)
            print(f"[POSE] Exported textured GLB → {glb_path}", flush=True)
        except Exception as e:
            print(f"[POSE] WARNING: Textured GLB export failed ({e}), falling back to NPZ",
                  flush=True)
            canonical_glb_path = None  # Fall through to NPZ path

    if not canonical_glb_path or not os.path.exists(str(canonical_glb_path) if canonical_glb_path else ""):
        # Fallback: geometry-only GLB from NPZ
        # Decimate dense meshes for export
        if faces.shape[0] > MAX_FACES_FOR_EXPORT:
            import open3d as o3d
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices.numpy())
            mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
            mesh_o3d.remove_duplicated_vertices()
            mesh_o3d.remove_degenerate_triangles()
            mesh_o3d = mesh_o3d.simplify_quadric_decimation(MAX_FACES_FOR_EXPORT)
            vertices = torch.tensor(np.asarray(mesh_o3d.vertices), dtype=torch.float32, device=device)
            faces = np.asarray(mesh_o3d.triangles).astype(np.int32)
            print(f"[POSE] Export decimated to {faces.shape[0]} faces, {vertices.shape[0]} verts",
                  flush=True)

        vertices_aligned = tfm.transform_points(vertices_yup.unsqueeze(0))[0]

        # Include vertex colors if available
        vc = data.get("vertex_colors")
        mesh = trimesh.Trimesh(
            vertices=vertices_aligned.numpy().astype(np.float32),
            faces=faces,
            vertex_colors=vc,
        )
        mesh.export(glb_path)
        print(f"[POSE] Exported geometry-only GLB → {glb_path}", flush=True)

    # Also transform PBR GLB if provided
    if pbr_glb_path and aligned_pbr_path and os.path.exists(pbr_glb_path):
        try:
            pbr_scene = trimesh.load(pbr_glb_path)
            # Handle both single mesh and scene
            if isinstance(pbr_scene, trimesh.Scene):
                for geom_name, geom in pbr_scene.geometry.items():
                    if hasattr(geom, 'vertices'):
                        v = torch.tensor(geom.vertices, dtype=torch.float32, device=device)
                        # PBR GLB already has R_zup_to_yup applied (from o_voxel.to_glb)
                        # Apply S/R/T directly
                        v_aligned = tfm.transform_points(v.unsqueeze(0))[0]
                        geom.vertices = v_aligned.numpy().astype(np.float32)
                os.makedirs(os.path.dirname(aligned_pbr_path), exist_ok=True)
                pbr_scene.export(aligned_pbr_path)
            else:
                v = torch.tensor(pbr_scene.vertices, dtype=torch.float32, device=device)
                v_aligned = tfm.transform_points(v.unsqueeze(0))[0]
                pbr_scene.vertices = v_aligned.numpy().astype(np.float32)
                os.makedirs(os.path.dirname(aligned_pbr_path), exist_ok=True)
                pbr_scene.export(aligned_pbr_path)
            print(f"[POSE] Aligned PBR GLB → {aligned_pbr_path}", flush=True)
        except Exception as e:
            print(f"[POSE] WARNING: Failed to align PBR GLB: {e}", flush=True)


def process_single_object(depth_model, pointmap_data, obj, device="cuda"):
    """Process one object: load mesh, run layout_post_optimization, export."""
    from sam3d_objects.pipeline.inference_utils import layout_post_optimization

    name = obj["name"]
    t0 = time.time()

    # Load canonical mesh
    mesh_path = obj["mesh"]
    data = np.load(mesh_path)
    vertices = data["vertices"]
    faces = data["faces"]
    print(f"[POSE] {name}: mesh {vertices.shape[0]} verts, {faces.shape[0]} faces", flush=True)

    # Pre-decimate dense TRELLIS2 meshes before layout_post_optimization.
    # TRELLIS2 raw meshes have 500K–2.3M vertices (voxel decoder output),
    # while TRELLIS1 produces 3K–17K.  The optimizer's internal simplification
    # (simplify_quadric_decimation to 5000 triangles) struggles with extreme
    # 400:1 reductions, producing distorted meshes and poor alignment.
    # Pre-decimating to ~20K faces gives the optimizer a clean, manageable mesh.
    MAX_FACES_FOR_ALIGNMENT = 20000
    if faces.shape[0] > MAX_FACES_FOR_ALIGNMENT:
        import open3d as o3d
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(MAX_FACES_FOR_ALIGNMENT)
        vertices_dec = np.asarray(mesh_o3d.vertices).astype(np.float32)
        faces_dec = np.asarray(mesh_o3d.triangles).astype(np.int32)
        print(f"[POSE] {name}: pre-decimated {faces.shape[0]} → {faces_dec.shape[0]} faces "
              f"({vertices.shape[0]} → {vertices_dec.shape[0]} verts)", flush=True)
        vertices = vertices_dec
        faces = faces_dec

    # Create trimesh for layout_post_optimization
    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy())

    # Load mask
    mask = np.load(obj["mask"])
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

    # Get pointmap and intrinsics
    pointmap = pointmap_data["pointmap"]  # (3, H, W)
    intrinsics = pointmap_data["intrinsics"]  # (3, 3)

    # Resize mask and pointmap to same size
    pm_h, pm_w = pointmap.shape[1], pointmap.shape[2]
    if mask_tensor.shape[0] != pm_h or mask_tensor.shape[1] != pm_w:
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0),
            size=(pm_h, pm_w), mode="nearest"
        ).squeeze()

    # Compute initial pose
    mesh_verts_tensor = torch.tensor(vertices, dtype=torch.float32, device=device)
    quaternion, translation, scale = create_initial_pose(
        mesh_verts_tensor, pointmap, mask_tensor, device=device
    )

    # Force isotropic intrinsics to match TRELLIS1 pipeline behavior.
    # InferencePipelinePointMap.run_post_optimization() (line 293-297) does:
    #   re_focal = min(fx, fy); intrinsics[0,0] = intrinsics[1,1] = re_focal
    # This is required for correct silhouette rendering in layout_post_optimization.
    # We keep the original intrinsics for downstream scene rendering.
    intr_iso = intrinsics.clone()
    fx_n, fy_n = intr_iso[0, 0].item(), intr_iso[1, 1].item()
    re_focal = min(fx_n, fy_n)
    intr_iso[0, 0] = re_focal
    intr_iso[1, 1] = re_focal
    print(f"[POSE] {name}: intrinsics fx={fx_n:.4f} fy={fy_n:.4f} → isotropic {re_focal:.4f}",
          flush=True)

    # Pass pointmap directly — layout_post_optimization handles its own resize.
    # Do NOT pad-to-square with NaN then bilinear-resize, as bilinear interpolation
    # with NaN neighbors spreads NaN to all surrounding pixels.
    point_map = pointmap.float().permute(1, 2, 0)  # (3, H, W) → (H, W, 3)

    # Run layout_post_optimization
    print(f"[POSE] {name}: running layout_post_optimization...", flush=True)
    t_opt = time.time()
    try:
        revised_quat, revised_t, revised_scale, final_iou, flag_icp, flag_optim = (
            layout_post_optimization(
                mesh,
                quaternion.unsqueeze(1),  # (1, 1, 4)
                translation,              # (1, 3)
                scale,                    # (1, 3)
                mask_tensor,              # (H, W)
                point_map,                # (H, W, 3)
                intr_iso,                 # (3, 3) — isotropic for optimizer
                min_size=518,
            )
        )
    except Exception as e:
        print(f"[POSE] {name}: layout_post_optimization FAILED: {e}", flush=True)
        # Fall back to initial pose
        revised_quat = quaternion
        revised_t = translation
        revised_scale = scale
        final_iou = -1.0
    t_opt_end = time.time()

    print(f"[POSE] {name}: post-opt {t_opt_end - t_opt:.1f}s, IoU={final_iou:.4f}",
          flush=True)

    # Extract final pose
    if hasattr(revised_quat, 'squeeze'):
        R_final = revised_quat.squeeze().cpu().float()
    else:
        R_final = torch.tensor(revised_quat, dtype=torch.float32)
    if hasattr(revised_t, 'squeeze'):
        T_final = revised_t.squeeze().cpu().float()
    else:
        T_final = torch.tensor(revised_t, dtype=torch.float32)
    if hasattr(revised_scale, 'squeeze'):
        S_final = revised_scale.squeeze().cpu().float()
    else:
        S_final = torch.tensor(revised_scale, dtype=torch.float32)

    # Ensure correct shapes for transform_and_export_glb
    if R_final.dim() == 1:
        R_final = R_final.unsqueeze(0)
    if T_final.dim() == 1:
        T_final = T_final.unsqueeze(0)
    if S_final.dim() == 1:
        S_final = S_final.unsqueeze(0)

    # Export aligned GLB from full-resolution raw mesh (decimated to 100K faces
    # for export if needed — separate from the 20K alignment decimation).
    glb_path = obj["glb"]
    pbr_glb_path = obj.get("pbr_glb")
    aligned_pbr_path = obj.get("aligned_pbr")
    canonical_glb_path = obj.get("canonical_glb")
    transform_and_export_glb(
        mesh_path, glb_path, R_final, T_final, S_final,
        pbr_glb_path=pbr_glb_path,
        aligned_pbr_path=aligned_pbr_path,
        canonical_glb_path=canonical_glb_path,
    )

    # Build info dict — store ORIGINAL (non-isotropic) intrinsics for downstream
    # scene rendering.  The optimizer received isotropic intrinsics (intr_iso),
    # but rendering needs the true camera model.
    info = {
        "object_name": name,
        "glb_path": glb_path,
        "translation": T_final.squeeze().tolist(),
        "rotation": R_final.squeeze().tolist(),
        "scale": S_final.squeeze().tolist(),
        "iou": float(final_iou),
        "intrinsics": intrinsics.cpu().tolist() if hasattr(intrinsics, 'cpu') else intrinsics,
        "pointmap_shape": [pm_h, pm_w],
    }

    # Write info JSON
    info_path = obj.get("info")
    if info_path:
        os.makedirs(os.path.dirname(info_path), exist_ok=True)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)

    elapsed = time.time() - t0
    glb_mb = os.path.getsize(glb_path) / (1024 * 1024)
    print(f"[POSE] {name}: OK ({elapsed:.1f}s, {glb_mb:.1f}MB, IoU={final_iou:.4f})",
          flush=True)
    return info


def main():
    p = argparse.ArgumentParser(description="Pose alignment worker — MoGe + layout_post_opt")
    p.add_argument("--manifest", required=True, help="Path to JSON manifest file")
    args = p.parse_args()

    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    objects = manifest["objects"]
    scene_image_path = manifest["scene_image"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load MoGe model
    depth_model = load_moge_model(device=device)

    # Run MoGe on scene image (once, shared across all objects)
    from PIL import Image
    scene_img = np.array(Image.open(scene_image_path).convert("RGB"))
    print(f"[POSE] Computing scene pointmap ({scene_img.shape})...", flush=True)
    t_pm = time.time()
    pointmap_data = compute_pointmap(depth_model, scene_img, device=device)
    print(f"[POSE] Pointmap computed in {time.time() - t_pm:.1f}s", flush=True)

    # Free MoGe model from GPU (not needed during post-opt)
    depth_model.model.cpu()
    torch.cuda.empty_cache()

    # Process each object
    results = {}
    total_start = time.time()
    for i, obj in enumerate(objects):
        name = obj["name"]
        glb_path = obj["glb"]
        info_path = obj.get("info")

        # Skip if already completed
        if os.path.exists(glb_path) and info_path and os.path.exists(info_path):
            print(f"[POSE] {name}: already exists, skipping", flush=True)
            with open(info_path, 'r', encoding='utf-8') as f:
                results[name] = json.load(f)
            continue

        print(f"[POSE] Processing {name} ({i+1}/{len(objects)})...", flush=True)
        try:
            info = process_single_object(depth_model, pointmap_data, obj, device=device)
            results[name] = info
        except Exception as e:
            import traceback
            print(f"[POSE] {name}: FAILED - {e}", flush=True)
            traceback.print_exc()
            results[name] = {"name": name, "error": str(e)}

    total_time = time.time() - total_start
    success = sum(1 for v in results.values() if "error" not in v)
    print(f"\n[POSE] Completed: {success}/{len(objects)} in {total_time:.1f}s", flush=True)

    # Write summary
    manifest_dir = os.path.dirname(args.manifest)
    if manifest_dir:
        summary_path = os.path.join(manifest_dir, "pose_align_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                "total_time": total_time,
                "objects": {
                    name: {
                        "success": "error" not in v,
                        "iou": v.get("iou"),
                    }
                    for name, v in results.items()
                },
            }, f, indent=2)


if __name__ == "__main__":
    main()
