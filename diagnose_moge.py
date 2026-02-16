"""Run MoGe on a segmented PNG and save intrinsics + pointmap.

Must run in sam3d_viga conda env:
  C:/Users/kingy/miniconda3/envs/sam3d_viga/python.exe diagnose_moge.py --image <png> --output <npz>
"""
import argparse
import os
import numpy as np
import torch
from PIL import Image
from moge.model.v1 import MoGeModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="RGBA segmented PNG")
    p.add_argument("--output", required=True, help="Output .npz path")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    model.eval()

    img = Image.open(args.image).convert("RGB")
    W, H = img.size
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model.infer(img_tensor, force_projection=False)

    points = output["points"].cpu().numpy()    # (H, W, 3)
    intrinsics = output["intrinsics"].cpu().numpy()  # (3, 3) -- normalized
    depth = output.get("depth")
    if depth is not None:
        depth = depth.cpu().numpy()

    # Denormalize intrinsics to pixel coordinates
    fx_norm = intrinsics[0, 0]
    fy_norm = intrinsics[1, 1]
    cx_norm = intrinsics[0, 2]
    cy_norm = intrinsics[1, 2]

    fx_px = fx_norm * W
    fy_px = fy_norm * H
    cx_px = cx_norm * W
    cy_px = cy_norm * H

    intrinsics_px = np.array([
        [fx_px, 0.0, cx_px],
        [0.0, fy_px, cy_px],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    print(f"Image size: {W}x{H}")
    print(f"MoGe intrinsics (normalized): fx={fx_norm:.4f} fy={fy_norm:.4f} cx={cx_norm:.4f} cy={cy_norm:.4f}")
    print(f"MoGe intrinsics (pixels):     fx={fx_px:.1f} fy={fy_px:.1f} cx={cx_px:.1f} cy={cy_px:.1f}")
    print(f"Pointmap shape: {points.shape}")
    finite_mask = np.isfinite(points).all(axis=-1)
    pts_finite = points[finite_mask]
    print(f"Pointmap range (finite only, {finite_mask.sum()} of {finite_mask.size} pixels):")
    print(f"  X=[{pts_finite[:,0].min():.3f}, {pts_finite[:,0].max():.3f}]  "
          f"Y=[{pts_finite[:,1].min():.3f}, {pts_finite[:,1].max():.3f}]  "
          f"Z=[{pts_finite[:,2].min():.3f}, {pts_finite[:,2].max():.3f}]")

    save_dict = {
        "points": points,
        "intrinsics_norm": intrinsics,
        "intrinsics_px": intrinsics_px,
        "image_width": W,
        "image_height": H,
    }
    if depth is not None:
        save_dict["depth"] = depth

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, **save_dict)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
