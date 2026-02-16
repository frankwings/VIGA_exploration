"""Render GLB mesh using PyTorch3D with MoGe intrinsics - matches the original pipeline's rendering.

Usage:
    C:/Users/kingy/miniconda3/envs/sam3d_py311/python.exe diagnose_pt3d_render.py

This script uses the SAME rendering setup as layout_post_optimization to verify
that the corrected vertices project correctly.
"""
import json
import os
import numpy as np
import torch
import trimesh
from PIL import Image

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
)
from pytorch3d.transforms import Transform3d, quaternion_to_matrix

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAM_INIT_DIR = os.path.join(
    PROJECT_ROOT, "output", "static_scene", "20260210_043534", "greentea", "sam_init"
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "alignment_diagnostic")

OBJECTS = [
    "green_tea_bottle",
    "green_tea_bottle_1",
    "alienware_keyboard",
    "alienware_keyboard_1",
    "envelope",
    "headphones",
]


def load_mesh_raw(glb_path):
    """Load raw TRELLIS mesh vertices (before any transform)."""
    scene = trimesh.load(glb_path)
    if hasattr(scene, 'geometry'):
        meshes = list(scene.geometry.values())
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        mesh = scene
    return mesh


def apply_correct_transform(mesh, rotation_q, translation, scale, device):
    """Apply correct transform chain matching layout_post_optimization_utils.get_mesh().

    1. Pre-transform: z-up to y-up conversion (from get_mesh line 116)
    2. compose_transform(S, R, T) via PyTorch3D's real Transform3d
    """
    vertices = mesh.vertices.copy()

    # Pre-transform from layout_post_optimization_utils.py line 116:
    # mesh_vertices = mesh_vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
    pre = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T  # = [[1,0,0],[0,0,1],[0,-1,0]]
    vertices = vertices @ pre

    vertices = torch.from_numpy(vertices).float().to(device)

    # Build transform using REAL PyTorch3D Transform3d
    R_mat = quaternion_to_matrix(rotation_q.unsqueeze(0).to(device))  # (1, 3, 3)
    S = scale.to(device)
    T = translation.to(device)

    tfm = Transform3d(device=device)
    tfm = tfm.scale(S.unsqueeze(0)).rotate(R_mat).translate(T.unsqueeze(0))

    vertices_world = tfm.transform_points(vertices.unsqueeze(0))
    return vertices_world[0]


def render_mesh(vertices, faces, intrinsics_norm, img_h, img_w, device):
    """Render mesh using PyTorch3D matching the original pipeline's camera setup."""
    # Denormalize intrinsics
    fx = float(intrinsics_norm[0, 0]) * img_w
    fy = float(intrinsics_norm[1, 1]) * img_h
    cx = float(intrinsics_norm[0, 2]) * img_w
    cy = float(intrinsics_norm[1, 2]) * img_h

    # PyTorch3D PerspectiveCameras with screen-space intrinsics
    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], device=device, dtype=torch.float32),
        principal_point=torch.tensor([[cx, cy]], device=device, dtype=torch.float32),
        image_size=torch.tensor([[img_h, img_w]], device=device, dtype=torch.float32),
        in_ndc=False,
        device=device,
    )

    raster_settings = RasterizationSettings(
        image_size=(img_h, img_w),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, -3.0]],
        ambient_color=[[0.5, 0.5, 0.5]],
        diffuse_color=[[0.5, 0.5, 0.5]],
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    verts = vertices.unsqueeze(0).to(device)
    faces_t = torch.from_numpy(faces).long().unsqueeze(0).to(device)
    textures = TexturesVertex(verts_features=torch.ones_like(verts) * 0.7)
    mesh = Meshes(verts=verts, faces=faces_t, textures=textures)

    with torch.no_grad():
        images = renderer(mesh)

    return images[0, ..., :4].cpu().numpy()


def make_comparison(input_png, render_img, output_png, label):
    """Create side-by-side comparison."""
    img_2d = Image.open(input_png).convert("RGBA")

    # Convert render to PIL
    render_rgba = (np.clip(render_img, 0, 1) * 255).astype(np.uint8)
    img_3d = Image.fromarray(render_rgba, 'RGBA')

    target_h = 512
    w1 = int(img_2d.width * target_h / img_2d.height)
    w2 = int(img_3d.width * target_h / img_3d.height)
    img_2d = img_2d.resize((w1, target_h), Image.LANCZOS)
    img_3d = img_3d.resize((w2, target_h), Image.LANCZOS)

    bg1 = Image.new("RGB", (w1, target_h), (255, 255, 255))
    bg1.paste(img_2d, mask=img_2d.split()[3])
    bg2 = Image.new("RGB", (w2, target_h), (240, 240, 240))
    bg2.paste(img_3d, mask=img_3d.split()[3])

    gap = 4
    total_w = w1 + gap + w2
    label_h = 30
    canvas = Image.new("RGB", (total_w, target_h + label_h), (255, 255, 255))
    canvas.paste(bg1, (0, label_h))
    canvas.paste(bg2, (w1 + gap, label_h))

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        draw.text((4, 4), f"2D Input: {label}", fill=(0, 0, 0), font=font)
        draw.text((w1 + gap + 4, 4), "PT3D Render (correct transform)", fill=(0, 0, 0), font=font)
    except ImportError:
        pass

    canvas.save(output_png)
    print(f"  Saved: {output_png}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    transforms_path = os.path.join(SAM_INIT_DIR, "object_transforms.json")
    with open(transforms_path, "r", encoding="utf-8") as f:
        transforms = json.load(f)

    print("=" * 60)
    print("PT3D DIAGNOSTIC RENDER")
    print("=" * 60)

    for i, obj_name in enumerate(OBJECTS):
        print(f"\n[{i+1}/{len(OBJECTS)}] {obj_name}")

        png_path = os.path.join(SAM_INIT_DIR, f"{obj_name}.png")
        glb_path = os.path.join(SAM_INIT_DIR, f"{obj_name}.glb")
        npz_path = os.path.join(OUTPUT_DIR, f"{obj_name}_moge.npz")
        compare_path = os.path.join(OUTPUT_DIR, f"{obj_name}_pt3d_compare.png")

        if not os.path.exists(npz_path):
            print("  [SKIP] No MoGe data")
            continue

        t = transforms[i]
        rotation_q = torch.tensor(t['rotation'], dtype=torch.float32)
        translation = torch.tensor(t['translation'], dtype=torch.float32)
        scale = torch.tensor(t['scale'], dtype=torch.float32)

        # Load raw mesh and apply correct transform
        mesh = load_mesh_raw(glb_path)

        # But we have the BROKEN GLB, not the raw TRELLIS output.
        # The broken GLB has: v_broken = v_raw @ pre_broken @ SR (no T)
        # We need to undo pre_broken @ SR and recover v_raw first.

        # Broken pre: [[-1,0,0],[0,0,1],[0,-1,0]]
        pre_broken = torch.tensor([[-1,0,0],[0,0,1],[0,-1,0]], dtype=torch.float32)
        R_mat = quaternion_to_matrix(rotation_q.unsqueeze(0))[0]
        S_val = scale[0]
        SR = (torch.eye(3) * S_val) @ R_mat

        v_broken = torch.tensor(mesh.vertices, dtype=torch.float32)
        v_raw = v_broken @ torch.inverse(pre_broken @ SR)

        # Now apply correct transform using get_mesh's pre-transform
        mesh_raw = trimesh.Trimesh(vertices=v_raw.numpy(), faces=mesh.faces, process=False)
        vertices_world = apply_correct_transform(
            mesh_raw, rotation_q, translation, scale, device
        )

        center = vertices_world.mean(0)
        print(f"  PT3D center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")

        # Load MoGe intrinsics
        moge = np.load(npz_path)
        intrinsics_norm = moge['intrinsics_norm']
        img_w = int(moge['image_width'])
        img_h = int(moge['image_height'])

        # Render
        render_img = render_mesh(
            vertices_world, mesh.faces, intrinsics_norm, img_h, img_w, device
        )

        # Check if anything rendered
        alpha = render_img[:, :, 3]
        visible_pixels = (alpha > 0.01).sum()
        print(f"  Visible pixels: {visible_pixels}")

        if visible_pixels > 0:
            # Find rendered region
            rows, cols = np.where(alpha > 0.01)
            print(f"  Render region: rows=[{rows.min()}, {rows.max()}], cols=[{cols.min()}, {cols.max()}]")

        make_comparison(png_path, render_img, compare_path, obj_name)

    print("\nDone!")


if __name__ == "__main__":
    main()
