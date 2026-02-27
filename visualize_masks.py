import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# 设置路径
MASKS_DIR = Path(r"D:\Projects\ProjectGenesis\GenesisVIGA\output\test_sam")
INPUT_IMAGE = Path(r"D:\Projects\ProjectGenesis\GenesisVIGA\docs\test_results_images\01_greentea_input.jpg")

# 加载遮罩和名称
masks = np.load(MASKS_DIR / "all_masks.npy")
with open(MASKS_DIR / "all_masks_object_names.json") as f:
    names_data = json.load(f)
object_names = names_data["object_names"]

# 读取原始图像
import matplotlib.image as mpimg
original_image = mpimg.imread(INPUT_IMAGE)

# 创建输出目录
output_dir = Path(r"D:\Projects\ProjectGenesis\GenesisVIGA\output\mask_visualization")
output_dir.mkdir(parents=True, exist_ok=True)

# 绘制遮罩
plt.figure(figsize=(20, 15))

# 原始图像
plt.subplot(2, len(object_names)//2 + 1, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis('off')

# 遮罩图像
for i, (mask, name) in enumerate(zip(masks, object_names), 2):
    plt.subplot(2, len(object_names)//2 + 1, i)
    
    # 遮罩为0的区域标记为对象
    object_mask = mask == 0
    
    # 创建RGB遮罩，对象区域为红色半透明
    mask_rgb = np.zeros((*object_mask.shape, 3), dtype=np.float32)
    mask_rgb[object_mask] = [1, 0, 0]  # 红色
    
    # 叠加遮罩和原始图像
    overlaid_image = original_image.copy()
    alpha = 0.5
    overlaid_image = (1-alpha) * overlaid_image + alpha * mask_rgb
    
    plt.imshow(overlaid_image)
    plt.title(f"{name}\n{100 * np.sum(object_mask) / object_mask.size:.1f}%")
    plt.axis('off')

plt.tight_layout()
plt.savefig(output_dir / "mask_comparison.png", dpi=300, bbox_inches='tight')
print(f"Mask visualization saved to: {output_dir / 'mask_comparison.png'}")