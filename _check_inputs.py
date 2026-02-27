"""Quick check of SAM init input images."""
from PIL import Image
import numpy as np
import os

d = "output/test/greentea/sam_init"
for f in sorted(os.listdir(d)):
    fp = os.path.join(d, f)
    if f.endswith(".png"):
        img = Image.open(fp)
        arr = np.array(img)
        nonzero = np.any(arr > 0, axis=-1) if arr.ndim == 3 else arr > 0
        pct = 100 * nonzero.sum() / nonzero.size
        print(f"{f}: size={img.size}, mode={img.mode}, visible={pct:.1f}%")
    elif f.endswith(".npy"):
        m = np.load(fp)
        print(f"{f}: shape={m.shape}, unique={np.unique(m)[:5]}")

# Check if there's a full scene image
for candidate in ["target.png", "scene.png", "greentea.png"]:
    for root in ["data/static_scene/greentea", "output/test/greentea"]:
        fp = os.path.join(root, candidate)
        if os.path.exists(fp):
            img = Image.open(fp)
            print(f"\nScene image found: {fp}, size={img.size}, mode={img.mode}")
