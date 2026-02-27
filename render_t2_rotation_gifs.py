#!/usr/bin/env python3
"""
Generate Y-axis and X-axis rotation GIFs for TRELLIS2 GLBs.
Uses matplotlib 3D rendering (no OpenGL/Blender dependency).
Extracts mesh faces + UV-baked approximate colors from GLB PBR materials.
"""

import os
import math
import numpy as np
from pathlib import Path
import io

import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

GLB_DIR = Path("d:/Projects/ProjectGenesis/GenesisVIGA/output/sam3d_dining_t2")
OUTPUT_DIR = GLB_DIR / "rotation_gifs"
RESOLUTION = 512
N_FRAMES = 24
FPS_DELAY = 80  # ms per frame for GIF
BG_COLOR = (0.25, 0.25, 0.25)
MAX_FACES = 8000  # Subsample for speed


def extract_mesh_data(glb_path: str):
    """Load GLB, extract vertices, faces, and per-face colors."""
    tm = trimesh.load(str(glb_path))

    if isinstance(tm, trimesh.Scene):
        meshes = [g for g in tm.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No trimesh geometry found in scene")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = tm

    verts = mesh.vertices
    faces = mesh.faces

    # Try to get face colors
    face_colors = None
    try:
        # trimesh resolves UV + texture to face colors
        fc = mesh.visual.face_colors
        if fc is not None and len(fc) == len(faces):
            face_colors = fc[:, :3] / 255.0
    except Exception:
        pass

    if face_colors is None:
        try:
            vc = mesh.visual.vertex_colors
            if vc is not None and len(vc) == len(verts):
                # Average vertex colors per face
                face_colors = vc[faces].mean(axis=1)[:, :3] / 255.0
        except Exception:
            pass

    if face_colors is None:
        # Default grey
        face_colors = np.full((len(faces), 3), 0.6)

    # Subsample if too many faces
    if len(faces) > MAX_FACES:
        idx = np.random.choice(len(faces), MAX_FACES, replace=False)
        faces = faces[idx]
        face_colors = face_colors[idx]

    return verts, faces, face_colors


def render_frame(verts, faces, face_colors, elev, azim, center, radius):
    """Render one frame using matplotlib 3D."""
    fig = plt.figure(figsize=(RESOLUTION / 100, RESOLUTION / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    # Set background
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    # Hide axes
    ax.set_axis_off()

    # Create polygon collection
    triangles = verts[faces]
    poly = Poly3DCollection(triangles, linewidths=0.0, edgecolors='none')
    poly.set_facecolor(face_colors)
    ax.add_collection3d(poly)

    # Set view
    ax.view_init(elev=elev, azim=azim)

    # Set limits centered on object
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout(pad=0)

    # Render to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.05, dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
    return img


def render_glb_rotation(glb_path: str, name: str, output_dir: Path):
    """Render Y and X rotation GIFs for a single GLB."""
    print(f"  Loading and extracting mesh data...")
    verts, faces, face_colors = extract_mesh_data(glb_path)
    print(f"  Mesh: {len(verts)} verts, {len(faces)} faces")

    # Compute bounds
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    extents = verts.max(axis=0) - verts.min(axis=0)
    radius = max(extents) * 0.6

    for axis in ['y', 'x']:
        frames = []
        print(f"    Rendering {axis}-axis ({N_FRAMES} frames)...", end='', flush=True)

        for i in range(N_FRAMES):
            angle_deg = 360.0 * i / N_FRAMES

            if axis == 'y':
                # Turntable: azimuth rotates, elevation fixed
                elev = 20
                azim = angle_deg
            else:
                # Tumble: elevation rotates, azimuth fixed
                elev = -90 + angle_deg  # -90 to 270
                azim = 45

            img = render_frame(verts, faces, face_colors, elev, azim, center, radius)
            frames.append(img)
            print('.', end='', flush=True)

        print()

        # Save GIF with ping-pong
        gif_path = output_dir / f"{name}_{axis}_rotation.gif"
        pingpong = frames + frames[-2:0:-1]
        pingpong[0].save(
            gif_path,
            save_all=True,
            append_images=pingpong[1:],
            duration=FPS_DELAY,
            loop=0
        )
        size_kb = gif_path.stat().st_size / 1024
        print(f"    Saved {gif_path.name} ({len(pingpong)} frames, {size_kb:.0f} KB)")

    return True


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.random.seed(42)

    objects = [
        "sofa_cover", "tablecloth", "chair_cover", "chair",
        "pillow_and_blanket", "newspaper", "pot_and_trivet", "plant"
    ]

    results = {}
    for name in objects:
        glb_path = GLB_DIR / f"{name}.glb"
        if not glb_path.exists():
            print(f"  SKIP {name}: GLB not found")
            results[name] = False
            continue

        size_mb = glb_path.stat().st_size / (1024 * 1024)
        print(f"\nProcessing {name} ({size_mb:.1f} MB)...")
        try:
            ok = render_glb_rotation(str(glb_path), name, OUTPUT_DIR)
            results[name] = ok
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print("\n=== Results ===")
    for name, ok in results.items():
        status = 'OK' if ok else 'FAILED'
        print(f"  {name}: {status}")
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
