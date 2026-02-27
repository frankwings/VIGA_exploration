"""Compare old single-ICP run vs new two-pass ICP run — projected onto scene."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw

ROOT        = Path(__file__).parent.resolve()
TARGET_IMG  = ROOT / "data/static_scene/dining/target_resized.jpg"
MOGE_NPZ    = ROOT / "output/sam3d_dining_v4/target_moge.npz"
MASK_NPY    = ROOT / "output/sam3d_dining_v4/wooden_chair.npy"

OLD_DIR     = ROOT / "output/experiment_viga_sam3d"
NEW_DIR     = ROOT / "output/sam3d_two_pass_icp"

OUT_DIR     = ROOT / "output/compare_two_pass_icp"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cam(npz: Path) -> dict:
    d = np.load(npz)
    K = d["intrinsics_px"].astype(np.float64)
    return {
        "fx": K[0,0], "fy": K[1,1],
        "cx": K[0,2], "cy": K[1,2],
        "width": int(d["image_width"]),
        "height": int(d["image_height"]),
    }


def project_glb(glb_path: Path, info_path: Path, scene_img: Image.Image,
                cam: dict, color=(0, 200, 255)) -> Image.Image:
    import trimesh
    scene = trimesh.load(str(glb_path), force="scene")
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if hasattr(g, "vertices")])
    else:
        mesh = scene
    verts = np.array(mesh.vertices, dtype=np.float64)
    if len(verts) > 8000:
        idx = np.random.choice(len(verts), 8000, replace=False)
        verts = verts[idx]

    # PyTorch3D cam space -> OpenCV
    x =  -verts[:, 0]
    y =  -verts[:, 1]
    z =   verts[:, 2]
    valid = z > 0.01
    x, y, z = x[valid], y[valid], z[valid]

    u = (cam["fx"] * x / z + cam["cx"]).astype(np.int32)
    v = (cam["fy"] * y / z + cam["cy"]).astype(np.int32)

    W, H = scene_img.size
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[ok], v[ok]

    info = json.loads(info_path.read_text())
    iou = info.get("iou") or info.get("final_iou")

    result = scene_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    r, g, b = color
    for ux, vy in zip(u, v):
        draw.ellipse([ux-3, vy-3, ux+3, vy+3], fill=(r, g, b, 220))
    result = Image.alpha_composite(result, overlay).convert("RGB")

    if iou is not None:
        d2 = ImageDraw.Draw(result)
        d2.rectangle([0, 0, 130, 25], fill=(0,0,0))
        d2.text((5, 5), f"IoU: {iou:.4f}", fill=(255,255,0))
    return result


def mask_overlay(scene_img: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.array(scene_img.convert("RGB"), dtype=np.float32)
    ov  = arr.copy()
    m   = mask > 0
    ov[m, 0] = 0; ov[m, 1] = 200; ov[m, 2] = 0
    blended = (0.6 * arr + 0.4 * ov).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def main():
    np.random.seed(42)
    cam  = load_cam(MOGE_NPZ)
    mask = np.load(MASK_NPY)

    # Scale to display width 550
    DISP_W = 550
    scale  = DISP_W / cam["width"]
    disp_h = int(cam["height"] * scale)
    cam_d  = {k: cam[k]*scale for k in ("fx","fy","cx","cy")}

    scene_full = Image.open(TARGET_IMG).convert("RGB")
    scene_moge = scene_full.resize((cam["width"], cam["height"]), Image.LANCZOS)
    scene_disp = scene_moge.resize((DISP_W, disp_h), Image.LANCZOS)

    mask_pil  = Image.fromarray((mask > 0).astype(np.uint8)*255, "L").resize((DISP_W, disp_h), Image.NEAREST)
    mask_rsz  = np.array(mask_pil) > 0

    mask_vis  = mask_overlay(scene_disp, mask_rsz)
    old_proj  = project_glb(OLD_DIR/"wooden_chair.glb", OLD_DIR/"wooden_chair_info.json",
                            scene_disp, cam_d, color=(255, 80, 80))
    new_proj  = project_glb(NEW_DIR/"wooden_chair.glb", NEW_DIR/"wooden_chair_info.json",
                            scene_disp, cam_d, color=(0, 200, 255))

    # Save individual projections
    old_proj.save(str(OLD_DIR / "wooden_chair_proj.png"))
    new_proj.save(str(NEW_DIR  / "wooden_chair_proj.png"))

    # 3-panel comparison
    PAD = 8
    W, H = DISP_W, disp_h
    panels = [
        (mask_vis,  "SAM mask (green = wooden_chair)"),
        (old_proj,  "Old: single ICP (0.05m)"),
        (new_proj,  "New: two-pass ICP (0.1→0.05m)"),
    ]
    cmp_w = len(panels) * W + (len(panels)+1) * PAD
    cmp_h = H + 70
    cmp   = Image.new("RGB", (cmp_w, cmp_h), (20,20,20))
    draw  = ImageDraw.Draw(cmp)
    draw.text((PAD, 5), "wooden_chair — Single ICP vs Two-Pass ICP projection", fill=(220,220,220))
    for i, (img, lbl) in enumerate(panels):
        x = PAD + i * (W + PAD)
        cmp.paste(img, (x, 35))
        draw.text((x+4, 35+H+6), lbl, fill=(180,220,255))

    out = OUT_DIR / "comparison.png"
    cmp.save(str(out))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
