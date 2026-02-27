"""3-way comparison: SAM mask | TRELLIS-reverted | ICP-preserved."""
from pathlib import Path
import json, numpy as np
from PIL import Image, ImageDraw

ROOT       = Path(__file__).parent.resolve()
TARGET_IMG = ROOT / "data/static_scene/dining/target_resized.jpg"
MOGE_NPZ   = ROOT / "output/sam3d_dining_v4/target_moge.npz"
MASK_NPY   = ROOT / "output/sam3d_dining_v4/wooden_chair.npy"
OLD_DIR    = ROOT / "output/sam3d_two_pass_icp"    # two-pass ICP, but Adam reverted
NEW_DIR    = ROOT / "output/sam3d_icp_preserve"    # two-pass ICP + ICP preserved
OUT_DIR    = ROOT / "output/compare_icp_preserve"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_cam(npz):
    d = np.load(npz)
    K = d["intrinsics_px"].astype(np.float64)
    return {"fx":K[0,0],"fy":K[1,1],"cx":K[0,2],"cy":K[1,2],
            "width":int(d["image_width"]),"height":int(d["image_height"])}

def project_glb(glb_path, info_path, scene_img, cam, color):
    import trimesh
    scene = trimesh.load(str(glb_path), force="scene")
    mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if hasattr(g,"vertices")]) \
           if isinstance(scene, trimesh.Scene) else scene
    verts = np.array(mesh.vertices, dtype=np.float64)
    if len(verts) > 8000:
        verts = verts[np.random.choice(len(verts), 8000, replace=False)]
    x, y, z = -verts[:,0], -verts[:,1], verts[:,2]
    ok = z > 0.01
    x, y, z = x[ok], y[ok], z[ok]
    u = (cam["fx"]*x/z + cam["cx"]).astype(np.int32)
    v = (cam["fy"]*y/z + cam["cy"]).astype(np.int32)
    W, H = scene_img.size
    ok2 = (u>=0)&(u<W)&(v>=0)&(v<H)
    u, v = u[ok2], v[ok2]
    info = json.loads(Path(info_path).read_text())
    iou = info.get("iou") or info.get("final_iou")
    result = scene_img.copy().convert("RGBA")
    ov = Image.new("RGBA", result.size, (0,0,0,0))
    draw = ImageDraw.Draw(ov)
    r,g,b = color
    for ux,vy in zip(u,v):
        draw.ellipse([ux-3,vy-3,ux+3,vy+3], fill=(r,g,b,220))
    result = Image.alpha_composite(result, ov).convert("RGB")
    if iou is not None:
        d2 = ImageDraw.Draw(result)
        d2.rectangle([0,0,150,25], fill=(0,0,0))
        d2.text((5,5), f"IoU: {iou:.4f}", fill=(255,255,0))
    return result

def mask_overlay(scene_img, mask):
    arr = np.array(scene_img.convert("RGB"), dtype=np.float32)
    ov = arr.copy(); m = mask > 0
    ov[m,0]=0; ov[m,1]=200; ov[m,2]=0
    return Image.fromarray((0.6*arr+0.4*ov).clip(0,255).astype(np.uint8))

def main():
    np.random.seed(42)
    cam  = load_cam(MOGE_NPZ)
    mask = np.load(MASK_NPY)
    DISP_W = 550
    scale  = DISP_W / cam["width"]
    disp_h = int(cam["height"] * scale)
    cam_d  = {k: cam[k]*scale for k in ("fx","fy","cx","cy")}
    scene  = Image.open(TARGET_IMG).convert("RGB").resize((cam["width"],cam["height"]),Image.LANCZOS).resize((DISP_W,disp_h),Image.LANCZOS)
    mask_r = np.array(Image.fromarray((mask>0).astype(np.uint8)*255,"L").resize((DISP_W,disp_h),Image.NEAREST)) > 0

    panels = [
        (mask_overlay(scene, mask_r),      "SAM mask (green = wooden_chair)"),
        (project_glb(OLD_DIR/"wooden_chair.glb", OLD_DIR/"wooden_chair_info.json",
                     scene, cam_d, (255,80,80)),   "Two-pass ICP — Adam reverted (TRELLIS pose)"),
        (project_glb(NEW_DIR/"wooden_chair.glb", NEW_DIR/"wooden_chair_info.json",
                     scene, cam_d, (0,200,255)),   "Two-pass ICP — ICP preserved (fix applied)"),
    ]
    PAD=8; W,H=DISP_W,disp_h
    cmp = Image.new("RGB", (len(panels)*W+(len(panels)+1)*PAD, H+70), (20,20,20))
    draw = ImageDraw.Draw(cmp)
    draw.text((PAD,5), "wooden_chair — Before vs After ICP-preserve fix", fill=(220,220,220))
    for i,(img,lbl) in enumerate(panels):
        x = PAD + i*(W+PAD)
        cmp.paste(img, (x,35))
        draw.text((x+4, 35+H+6), lbl, fill=(180,220,255))
    out = OUT_DIR/"comparison.png"
    cmp.save(str(out))
    print(f"Saved: {out}")

if __name__=="__main__":
    main()
