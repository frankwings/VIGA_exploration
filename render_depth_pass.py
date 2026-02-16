#!/usr/bin/env python
"""Blender depth-pass renderer for SAM3D scene.

Imports all GLBs, sets up the MoGe camera, renders Z-buffer depth pass
and optional RGB reference.

Usage:
    blender -b -P render_depth_pass.py -- <glb_dir> <moge_npz> <output_depth.npy> [output_depth_vis.png] [output_rgb.png]

Arguments (after --):
    glb_dir             Directory containing *.glb files
    moge_npz            Path to target_moge.npz (camera intrinsics + depth)
    output_depth.npy    Where to save the raw Z-buffer depth
    output_depth_vis    (optional) Path for colourised depth visualisation PNG
    output_rgb          (optional) Path for RGB reference render

Requirements: Blender 4.x with Python (bpy, numpy)
"""

import sys
import os
import math
from pathlib import Path

import bpy
import numpy as np


def parse_args():
    """Parse arguments after '--' in the blender command line."""
    argv = sys.argv
    if "--" not in argv:
        print("ERROR: Pass arguments after '--'.  Example:")
        print("  blender -b -P render_depth_pass.py -- <glb_dir> <moge_npz> <output.npy>")
        sys.exit(1)

    args = argv[argv.index("--") + 1 :]
    if len(args) < 3:
        print("ERROR: Need at least 3 args: <glb_dir> <moge_npz> <output_depth.npy>")
        sys.exit(1)

    return {
        "glb_dir": Path(args[0]),
        "moge_npz": Path(args[1]),
        "output_depth": Path(args[2]),
        "output_depth_vis": Path(args[3]) if len(args) > 3 else None,
        "output_rgb": Path(args[4]) if len(args) > 4 else None,
    }


def clear_scene():
    """Remove all objects, meshes, materials from the default scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)
    for block in bpy.data.cameras:
        bpy.data.cameras.remove(block)
    for block in bpy.data.lights:
        bpy.data.lights.remove(block)


def import_all_glbs(glb_dir: Path):
    """Import all .glb files from directory.

    GLB vertices are already in PyTorch3D camera space (X-left, Y-up, Z-fwd).
    In Blender's coordinate system (X-right, Y-forward, Z-up), we need to
    convert:
        Blender.X = -PyTorch3D.X  (right = -left)
        Blender.Y =  PyTorch3D.Z  (forward = forward)
        Blender.Z = -PyTorch3D.Y  (up = -up... wait)

    Actually, since glTF importer applies its own Y-up to Z-up conversion,
    we need to be careful. Let's import with default settings and check.

    The key insight: GLB files store vertices in their own local space.
    The glTF spec uses Y-up. Blender's glTF importer converts Y-up to Z-up
    automatically. So if vertices in the GLB are in PyTorch3D space
    (X-left, Y-up, Z-forward), after Blender import they become
    (X-left, Z-forward→Y, Y-up→Z).

    For depth rendering, we set up a camera that matches the MoGe convention.
    """
    glb_files = sorted(glb_dir.glob("*.glb"))
    if not glb_files:
        print(f"WARNING: No .glb files found in {glb_dir}")
        return []

    imported = []
    for glb_path in glb_files:
        print(f"  Importing: {glb_path.name}")
        bpy.ops.import_scene.gltf(filepath=str(glb_path))
        imported.append(glb_path.name)

    return imported


def setup_camera_from_moge(moge_path: Path):
    """Create camera matching MoGe intrinsics.

    MoGe uses OpenCV convention: X-right, Y-down, Z-forward.
    The intrinsics matrix K defines:
        u = fx * X/Z + cx
        v = fy * Y/Z + cy

    In Blender, the camera looks down -Z in camera local space,
    with X-right, Y-up.  To match OpenCV, we need to place the
    camera at origin looking down +Z (world), which in Blender
    means rotating the camera.

    Since GLB vertices go through Blender's Y-up→Z-up conversion:
        Blender coords: (X_glb, -Z_glb, Y_glb) from glTF spec
        For PyTorch3D vertices stored as glTF: the importer maps
        (X, Y, Z) → (X, -Z, Y) in Blender space.

    PyTorch3D (X-left, Y-up, Z-fwd) in GLB Y-up convention
    → Blender converts: Bx = X_pt3d, By = -Z_pt3d, Bz = Y_pt3d

    So in Blender space:
        Bx = X_pt3d = -X_cv   (left)
        By = -Z_pt3d = -Z_cv  (negative depth, so behind camera!)
        Bz = Y_pt3d = -Y_cv   (up)

    Hmm, the objects would be behind the camera at By < 0.

    This is because the glTF import conversion assumes Y-up → Z-up but
    our data is in camera space, not world space.  Let's take a different
    approach: we'll directly manipulate the mesh data in Blender after
    import to get proper OpenCV coordinates, then set up a standard camera.
    """
    data = np.load(str(moge_path))
    K = data["intrinsics_px"]
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    W = int(data["image_width"])
    H = int(data["image_height"])

    print(f"  MoGe camera: {W}x{H}, fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # ---- Transform all meshes from Blender-imported space to OpenCV space ----
    # After glTF Y-up → Blender Z-up import, the mapping is:
    #   Bx = X_pt3d, By = -Z_pt3d, Bz = Y_pt3d
    # We want OpenCV space for depth rendering:
    #   X_cv = -X_pt3d, Y_cv = -Y_pt3d, Z_cv = Z_pt3d
    # So: X_cv = -Bx, Y_cv = -Bz, Z_cv = -By
    # In Blender, we'll set camera at origin looking down +Y (Blender's +Y = OpenCV +Z).
    # But Blender camera looks down local -Z by default.
    #
    # Simpler approach: transform all mesh vertices to a space where
    # Blender camera at origin looking down -Z (default) sees the scene correctly.
    # That means we need:  Blender camera space = OpenCV space mapped to Blender camera:
    #   Cam_x = X_cv (right)
    #   Cam_y = -Y_cv (Blender camera Y is up, OpenCV Y is down)
    #   Cam_z = -Z_cv (Blender camera looks down -Z, OpenCV depth is +Z)
    #
    # From Blender-imported coords (Bx, By, Bz):
    #   Cam_x = -Bx     (X_cv = -X_pt3d = -Bx)
    #   Cam_y = Bz       (-Y_cv = -(-Y_pt3d) = Y_pt3d = Bz)
    #   Cam_z = By        (-Z_cv = -Z_pt3d = By)

    import mathutils

    # Create the coordinate transform matrix
    # [Cam_x]   [-1  0  0] [Bx]
    # [Cam_y] = [ 0  0  1] [By]
    # [Cam_z]   [ 0  1  0] [Bz]
    coord_matrix = mathutils.Matrix((
        (-1, 0, 0, 0),
        (0, 0, 1, 0),
        (0, 1, 0, 0),
        (0, 0, 0, 1),
    ))

    for obj in bpy.data.objects:
        if obj.type == "MESH":
            # Apply existing transforms first
            obj.data.transform(obj.matrix_world)
            obj.matrix_world = mathutils.Matrix.Identity(4)
            # Now apply coordinate conversion
            obj.data.transform(coord_matrix)
            obj.data.update()

    # ---- Create camera ----
    cam_data = bpy.data.cameras.new("MoGeCamera")
    cam_obj = bpy.data.objects.new("MoGeCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    # Camera at origin, looking down -Z (default in Blender)
    cam_obj.location = (0, 0, 0)
    cam_obj.rotation_euler = (0, 0, 0)

    # Set sensor / focal length to match intrinsics
    cam_data.type = "PERSP"
    cam_data.sensor_fit = "HORIZONTAL" if W >= H else "VERTICAL"

    # Blender focal_length = fx * sensor_width / image_width
    # We fix sensor_width and compute focal_length
    sensor_w = 36.0  # mm (standard)
    sensor_h = sensor_w * H / W
    cam_data.sensor_width = sensor_w
    cam_data.lens = fx * sensor_w / W

    # Handle principal point offset (shift_x, shift_y)
    # Blender shift is in fraction of sensor dimension:
    #   shift_x = (cx - W/2) / W  (positive = shift right)
    #   shift_y = (cy - H/2) / H  (positive = shift up, but OpenCV cy increases downward)
    # Actually Blender's shift_y positive means shift up in camera view,
    # but OpenCV cy increasing means going down, so:
    cam_data.shift_x = (cx - W / 2.0) / W
    cam_data.shift_y = -(cy - H / 2.0) / H  # negate for OpenCV→Blender

    # ---- Set render resolution ----
    scene = bpy.context.scene
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100

    # Clip range to encompass depth range
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0

    return W, H, fx, fy, cx, cy


def setup_depth_pass(depth_exr_path: str):
    """Configure compositor to output Z-buffer depth via EXR file.

    We use a File Output node writing OpenEXR so the raw Z values are
    preserved at full float32 precision, avoiding Viewer Node resolution
    issues.
    """
    scene = bpy.context.scene

    # Use EEVEE Next for speed (Blender 4.x)
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 1  # minimal for depth

    # Enable compositor and Z-pass
    scene.use_nodes = True
    scene.view_layers[0].use_pass_z = True

    # Build compositor node tree
    tree = scene.node_tree
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Render layers node
    rl_node = tree.nodes.new("CompositorNodeRLayers")
    rl_node.location = (0, 0)

    # File Output node for depth EXR
    fo_node = tree.nodes.new("CompositorNodeOutputFile")
    fo_node.location = (400, 0)
    fo_node.base_path = str(Path(depth_exr_path).parent)
    fo_node.format.file_format = "OPEN_EXR"
    fo_node.format.color_depth = "32"
    fo_node.format.color_mode = "BW"
    # Set the file slot name (this becomes the filename suffix)
    fo_node.file_slots[0].path = Path(depth_exr_path).stem

    # Connect Z pass to file output
    tree.links.new(rl_node.outputs["Depth"], fo_node.inputs[0])

    # Also keep the composite output for RGB
    comp_node = tree.nodes.new("CompositorNodeComposite")
    comp_node.location = (400, -200)
    tree.links.new(rl_node.outputs["Image"], comp_node.inputs["Image"])

    return fo_node


def render_and_extract_depth(W: int, H: int, depth_exr_path: str) -> np.ndarray:
    """Render the scene and read Z-buffer from the EXR file output."""
    import tempfile

    # Render (this triggers the file output node to write the EXR)
    bpy.ops.render.render(write_still=False)

    # File Output node appends frame number: stem0001.exr
    exr_dir = Path(depth_exr_path).parent
    exr_stem = Path(depth_exr_path).stem
    frame = bpy.context.scene.frame_current
    actual_exr = exr_dir / f"{exr_stem}{frame:04d}.exr"

    if not actual_exr.exists():
        # Try without frame number
        actual_exr = exr_dir / f"{exr_stem}.exr"

    if not actual_exr.exists():
        # Search for any matching EXR
        candidates = sorted(exr_dir.glob(f"{exr_stem}*.exr"))
        if candidates:
            actual_exr = candidates[0]
        else:
            print(f"ERROR: No EXR file found matching {exr_stem}*.exr in {exr_dir}")
            return np.zeros((H, W), dtype=np.float32)

    print(f"  Reading depth EXR: {actual_exr}")

    # Load EXR via Blender's image API
    exr_img = bpy.data.images.load(str(actual_exr))
    pixels = np.array(exr_img.pixels[:], dtype=np.float32)
    channels = exr_img.channels
    img_w = exr_img.size[0]
    img_h = exr_img.size[1]
    bpy.data.images.remove(exr_img)

    print(f"  EXR size: {img_w}x{img_h}, channels={channels}")
    pixels = pixels.reshape(img_h, img_w, channels)
    depth = pixels[:, :, 0]  # Z value is in the first channel

    # Blender stores images bottom-up, flip to top-down (matching OpenCV/MoGe)
    depth = np.flipud(depth)

    # Clean up the temp EXR
    try:
        actual_exr.unlink()
    except OSError:
        pass

    return depth


def save_depth_vis(depth: np.ndarray, path: Path, moge_depth: np.ndarray | None = None):
    """Save a colourised depth visualisation."""
    valid = (depth > 0) & (depth < 1e4) & (~np.isinf(depth))

    if moge_depth is not None:
        vmin = moge_depth.min()
        vmax = moge_depth.max()
    elif valid.any():
        vmin = depth[valid].min()
        vmax = depth[valid].max()
    else:
        vmin, vmax = 0, 1

    norm = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    norm[~valid] = 0

    # Turbo-ish colourmap (R,G,B)
    r = np.clip(1.0 - 2.0 * np.abs(norm - 0.75), 0, 1)
    g = np.clip(1.0 - 2.0 * np.abs(norm - 0.5), 0, 1)
    b = np.clip(1.0 - 2.0 * np.abs(norm - 0.25), 0, 1)

    img = np.stack([r, g, b], axis=-1)
    img[~valid] = 0
    img = (img * 255).astype(np.uint8)

    # Save using Blender's image API (no PIL needed)
    bl_img = bpy.data.images.new("DepthVis", width=depth.shape[1], height=depth.shape[0])
    # Blender expects bottom-up RGBA
    img_flipped = np.flipud(img)
    alpha = np.ones((*img_flipped.shape[:2], 1), dtype=np.uint8) * 255
    rgba = np.concatenate([img_flipped, alpha], axis=-1).astype(np.float32) / 255.0
    bl_img.pixels = rgba.flatten().tolist()
    bl_img.filepath_raw = str(path)
    bl_img.file_format = "PNG"
    bl_img.save_render(str(path))
    bpy.data.images.remove(bl_img)

    print(f"  Depth vis saved: {path}")


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("Depth Pass Renderer")
    print("=" * 70)

    # Clear default scene
    clear_scene()

    # Import GLBs
    print(f"\nImporting GLBs from: {args['glb_dir']}")
    imported = import_all_glbs(args["glb_dir"])
    print(f"  Imported {len(imported)} objects\n")

    # Setup camera
    print("Setting up MoGe camera:")
    W, H, fx, fy, cx, cy = setup_camera_from_moge(args["moge_npz"])

    # Setup depth pass compositing — use a temp EXR path next to the output
    depth_exr_path = str(args["output_depth"].parent / "_depth_exr_tmp")
    print("\nConfiguring depth pass...")
    setup_depth_pass(depth_exr_path)

    # Add minimal lighting (needed for RGB, doesn't affect Z pass)
    light_data = bpy.data.lights.new("Sun", type="SUN")
    light_data.energy = 3.0
    light_obj = bpy.data.objects.new("Sun", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.rotation_euler = (math.radians(45), 0, math.radians(45))

    # Render
    print("\nRendering depth pass...")
    depth = render_and_extract_depth(W, H, depth_exr_path)

    # Save raw depth
    np.save(str(args["output_depth"]), depth)
    print(f"  Raw depth saved: {args['output_depth']}")
    print(f"  Shape: {depth.shape}, range: [{depth[depth > 0].min():.4f}, {depth[(depth > 0) & (depth < 1e4)].max():.4f}]"
          if (depth > 0).any() and ((depth > 0) & (depth < 1e4)).any()
          else f"  Shape: {depth.shape}, all zero/invalid")

    # Save visualisation
    if args["output_depth_vis"]:
        moge_data = np.load(str(args["moge_npz"]))
        save_depth_vis(depth, args["output_depth_vis"], moge_data.get("depth"))

    # Save RGB reference
    if args["output_rgb"]:
        scene = bpy.context.scene
        scene.render.filepath = str(args["output_rgb"])
        scene.render.image_settings.file_format = "PNG"
        bpy.ops.render.render(write_still=True)
        print(f"  RGB render saved: {args['output_rgb']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
