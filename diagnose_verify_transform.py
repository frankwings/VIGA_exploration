"""Verify the corrected GLB vertices by projecting to 2D and comparing with MoGe pointmap.

Run in sam3d_py311 env:
    C:/Users/kingy/miniconda3/envs/sam3d_py311/python.exe diagnose_verify_transform.py
"""
import json
import os
import numpy as np
import torch
import trimesh

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAM_INIT_DIR = os.path.join(
    PROJECT_ROOT, "output", "static_scene", "20260210_043534", "greentea", "sam_init"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "alignment_diagnostic")

OBJECTS = [
    "green_tea_bottle",
    "green_tea_bottle_1",
    "alienware_keyboard",
    "headphones",
]


def quaternion_to_matrix(q):
    """Convert quaternion [r, i, j, k] to 3x3 rotation matrix."""
    r, i, j, k = q
    two_s = 2.0 / (q * q).sum()
    return torch.tensor([
        [1-two_s*(j*j+k*k), two_s*(i*j-k*r), two_s*(i*k+j*r)],
        [two_s*(i*j+k*r), 1-two_s*(i*i+k*k), two_s*(j*k-i*r)],
        [two_s*(i*k-j*r), two_s*(j*k+i*r), 1-two_s*(i*i+j*j)],
    ], dtype=torch.float32)


def main():
    transforms_path = os.path.join(SAM_INIT_DIR, "object_transforms.json")
    with open(transforms_path, "r", encoding="utf-8") as f:
        transforms = json.load(f)

    for obj_name in OBJECTS:
        print(f"\n{'='*60}")
        print(f"  {obj_name}")
        print(f"{'='*60}")

        glb_path = os.path.join(SAM_INIT_DIR, f"{obj_name}.glb")
        npz_path = os.path.join(OUTPUT_DIR, f"{obj_name}_moge.npz")

        if not os.path.exists(npz_path):
            print("  [SKIP] No MoGe npz")
            continue

        # Load MoGe data
        moge = np.load(npz_path)
        intrinsics_px = moge["intrinsics_px"]
        points = moge["points"]
        img_w = int(moge["image_width"])
        img_h = int(moge["image_height"])
        fx = intrinsics_px[0, 0]
        fy = intrinsics_px[1, 1]
        cx = intrinsics_px[0, 2]
        cy = intrinsics_px[1, 2]

        # MoGe pointmap center (average of finite points)
        finite_mask = np.isfinite(points).all(axis=-1)
        pts_f = points[finite_mask]
        moge_center = pts_f.mean(axis=0)
        print(f"  MoGe pointmap center: ({moge_center[0]:.4f}, {moge_center[1]:.4f}, {moge_center[2]:.4f})")

        # MoGe projects as: u = fx * X/Z + cx, v = fy * Y/Z + cy (OpenCV convention)
        # Verify: center of object in pixel coordinates
        u_moge = fx * moge_center[0] / moge_center[2] + cx
        v_moge = fy * moge_center[1] / moge_center[2] + cy
        print(f"  MoGe center projection: u={u_moge:.1f}, v={v_moge:.1f} (image {img_w}x{img_h})")

        # Load broken GLB
        scene = trimesh.load(glb_path)
        if hasattr(scene, 'geometry'):
            meshes = list(scene.geometry.values())
            mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        else:
            mesh = scene
        v_broken = torch.tensor(mesh.vertices, dtype=torch.float32)

        # Get transform
        idx = [o for o in ["green_tea_bottle", "green_tea_bottle_1", "alienware_keyboard",
               "alienware_keyboard_1", "envelope", "headphones"]].index(obj_name)
        t = transforms[idx]
        R_quat = torch.tensor(t['rotation'], dtype=torch.float32)
        S = torch.tensor(t['scale'][0], dtype=torch.float32)
        T = torch.tensor(t['translation'], dtype=torch.float32)
        R_mat = quaternion_to_matrix(R_quat)

        print(f"  T = ({T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f})")
        print(f"  S = {S:.4f}")

        # Pre-transforms
        R_flip_z = torch.tensor([[1,0,0],[0,1,0],[0,0,-1]], dtype=torch.float32)
        R_yup_to_zup = torch.tensor([[-1,0,0],[0,0,1],[0,1,0]], dtype=torch.float32)
        pre = R_flip_z @ R_yup_to_zup

        SR = (torch.eye(3) * S) @ R_mat

        # Method 1: Current code (undo pre, apply SR + T, negate X,Y for OpenCV)
        SR_inv = torch.inverse(SR)
        pre_inv = torch.inverse(pre)
        v_raw = v_broken @ SR_inv @ pre_inv
        v_correct = v_raw @ SR + T
        v_opencv_m1 = v_correct.clone()
        v_opencv_m1[:, 0] = -v_correct[:, 0]
        v_opencv_m1[:, 1] = -v_correct[:, 1]
        center_m1 = v_opencv_m1.mean(0)
        u_m1 = fx * center_m1[0].item() / center_m1[2].item() + cx
        v_m1 = fy * center_m1[1].item() / center_m1[2].item() + cy
        print(f"\n  Method 1 (undo pre, apply SR+T, negate XY):")
        print(f"    OpenCV center: ({center_m1[0]:.4f}, {center_m1[1]:.4f}, {center_m1[2]:.4f})")
        print(f"    Projection: u={u_m1:.1f}, v={v_m1:.1f}")

        # Method 2: Just add T to broken, negate X,Y for OpenCV
        v_with_t = v_broken + T
        v_opencv_m2 = v_with_t.clone()
        v_opencv_m2[:, 0] = -v_with_t[:, 0]
        v_opencv_m2[:, 1] = -v_with_t[:, 1]
        center_m2 = v_opencv_m2.mean(0)
        u_m2 = fx * center_m2[0].item() / center_m2[2].item() + cx
        v_m2 = fy * center_m2[1].item() / center_m2[2].item() + cy
        print(f"\n  Method 2 (broken + T, negate XY):")
        print(f"    OpenCV center: ({center_m2[0]:.4f}, {center_m2[1]:.4f}, {center_m2[2]:.4f})")
        print(f"    Projection: u={u_m2:.1f}, v={v_m2:.1f}")

        # Method 3: Undo pre only, add T, negate X,Y
        v_no_pre = v_broken @ pre_inv
        # v_no_pre = v_raw @ SR (pre undone)
        v_no_pre_t = v_no_pre + T
        v_opencv_m3 = v_no_pre_t.clone()
        v_opencv_m3[:, 0] = -v_no_pre_t[:, 0]
        v_opencv_m3[:, 1] = -v_no_pre_t[:, 1]
        center_m3 = v_opencv_m3.mean(0)
        u_m3 = fx * center_m3[0].item() / center_m3[2].item() + cx
        v_m3 = fy * center_m3[1].item() / center_m3[2].item() + cy
        print(f"\n  Method 3 (undo pre, add T, negate XY):")
        print(f"    OpenCV center: ({center_m3[0]:.4f}, {center_m3[1]:.4f}, {center_m3[2]:.4f})")
        print(f"    Projection: u={u_m3:.1f}, v={v_m3:.1f}")

        # Method 4: Undo pre@SR entirely, reapply SR+T, negate XY
        # Same as Method 1 (just verifying)
        v_raw2 = v_broken @ torch.inverse(pre @ SR)
        v_correct2 = v_raw2 @ SR + T
        v_opencv_m4 = v_correct2.clone()
        v_opencv_m4[:, 0] = -v_correct2[:, 0]
        v_opencv_m4[:, 1] = -v_correct2[:, 1]
        center_m4 = v_opencv_m4.mean(0)
        u_m4 = fx * center_m4[0].item() / center_m4[2].item() + cx
        v_m4 = fy * center_m4[1].item() / center_m4[2].item() + cy
        print(f"\n  Method 4 (undo pre@SR, reapply SR+T, negate XY):")
        print(f"    OpenCV center: ({center_m4[0]:.4f}, {center_m4[1]:.4f}, {center_m4[2]:.4f})")
        print(f"    Projection: u={u_m4:.1f}, v={v_m4:.1f}")

        # Method 5: Undo ONLY pre (not SR), then add T, negate X,Y
        # v_broken = v_raw @ pre @ SR
        # v_broken @ pre_inv = v_raw @ pre @ SR @ pre_inv (NOT the same as v_raw @ SR)
        # Let's try: undo = SR^-1 @ pre^-1, then reapply SR
        # This is method 1 again.

        # Method 6: Don't negate X,Y -- treat broken as already OpenCV-ish
        v_opencv_m6 = v_broken + T  # just add T, no negate
        center_m6 = v_opencv_m6.mean(0)
        u_m6 = fx * center_m6[0].item() / center_m6[2].item() + cx
        v_m6 = fy * center_m6[1].item() / center_m6[2].item() + cy
        print(f"\n  Method 6 (broken + T, no negate):")
        print(f"    OpenCV center: ({center_m6[0]:.4f}, {center_m6[1]:.4f}, {center_m6[2]:.4f})")
        print(f"    Projection: u={u_m6:.1f}, v={v_m6:.1f}")

        # Method 7: Post-transforms were identity, but what if we apply pytorch3d_to_cam only?
        # In PyTorch3D camera: u = fx * (-X)/Z + cx (negate X), v = fy * (-Y)/Z + cy (negate Y)
        # The broken vertices are v_raw @ pre @ SR (no T)
        # What if the pre-transforms ARE correct for the TRELLIS model space?
        # Then the broken vertices are in PyTorch3D camera space (minus T).
        # Adding T gives PyTorch3D camera coords. Project with negate:
        v_pt3d = v_broken + T
        center_pt3d = v_pt3d.mean(0)
        u_pt3d = fx * (-center_pt3d[0].item()) / center_pt3d[2].item() + cx
        v_pt3d_proj = fy * (-center_pt3d[1].item()) / center_pt3d[2].item() + cy
        print(f"\n  Method 7 (broken + T, PyTorch3D projection -X/Z, -Y/Z):")
        print(f"    PT3D center: ({center_pt3d[0]:.4f}, {center_pt3d[1]:.4f}, {center_pt3d[2]:.4f})")
        print(f"    Projection: u={u_pt3d:.1f}, v={v_pt3d_proj:.1f}")

        print(f"\n  TARGET (MoGe): u={u_moge:.1f}, v={v_moge:.1f}")


if __name__ == "__main__":
    main()
