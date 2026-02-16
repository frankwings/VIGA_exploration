"""Apply real-world sizing to SAM3D GLB objects.

Each TRELLIS-reconstructed GLB is normalized to roughly [-0.5, 0.5].
This module scales them to match real-world dimensions in meters.

Usage:
    from tools.sam3d.object_sizing import get_scale_factor, OBJECT_SIZES

    scale = get_scale_factor(mesh, "green_tea_bottle")
    mesh.vertices *= scale
"""

# Real-world sizes in meters (largest dimension)
OBJECT_SIZES = {
    "green_tea_bottle": 0.20,       # ~20cm tall (Ito En green tea)
    "green_tea_bottle_1": 0.20,     # same object, different segment
    "alienware_keyboard": 0.45,     # ~45cm wide (standard keyboard)
    "alienware_keyboard_1": 0.45,   # same object, different segment
    "envelope": 0.24,               # ~24cm long (standard letter envelope)
    "headphones": 0.18,             # ~18cm wide (over-ear headphones)
}


def get_scale_factor(mesh, object_name):
    """Compute scale factor to resize mesh to real-world size.

    Args:
        mesh: trimesh.Trimesh with vertices in raw TRELLIS space
        object_name: key into OBJECT_SIZES

    Returns:
        float scale factor (multiply vertices by this)
    """
    if object_name not in OBJECT_SIZES:
        return 1.0

    target_size = OBJECT_SIZES[object_name]
    current_size = max(mesh.bounding_box.extents)
    if current_size < 1e-8:
        return 1.0
    return target_size / current_size
