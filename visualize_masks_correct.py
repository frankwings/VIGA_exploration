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

# 设置图形大小和布局
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# 原始图像
axes[0].imshow(original_image)
axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
axes[0].axis('off')

# 显示每个遮罩
for i, (mask, name) in enumerate(zip(masks, object_names), 1):
    if i < len(axes):
        # 遮罩为0的区域是对象，255是背景
        # 显示为黑白图像：对象=白色(255)，背景=黑色(0)
        object_mask = (mask == 0).astype(np.uint8) * 255
        
        axes[i].imshow(object_mask, cmap='gray', vmin=0, vmax=255)
        
        # 计算覆盖率
        coverage = 100 * np.sum(mask == 0) / mask.size
        axes[i].set_title(f"{name}\n{coverage:.1f}% coverage", fontsize=12)
        axes[i].axis('off')

# 隐藏未使用的子图
for i in range(len(object_names) + 1, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(output_dir / "mask_comparison_blackwhite.png", dpi=300, bbox_inches='tight')
plt.close()

# 生成单独的遮罩文件
for i, (mask, name) in enumerate(zip(masks, object_names)):
    plt.figure(figsize=(8, 6))
    
    # 转换为黑白遮罩
    object_mask = (mask == 0).astype(np.uint8) * 255
    
    plt.imshow(object_mask, cmap='gray', vmin=0, vmax=255)
    plt.title(f"{name} - {100 * np.sum(mask == 0) / mask.size:.1f}% coverage", fontsize=14)
    plt.axis('off')
    
    # 保存单独文件
    plt.savefig(output_dir / f"{name}_mask.png", dpi=150, bbox_inches='tight')
    plt.close()

print(f"Black-white mask visualization saved to: {output_dir}")
print(f"Main comparison: {output_dir / 'mask_comparison_blackwhite.png'}")
print(f"Individual masks: {len(object_names)} separate PNG files")

# 生成遮罩统计
print(f"\nMask Statistics:")
print(f"{'Object Name':<25} {'Coverage %':<12} {'Pixels (Object)':<15} {'Total Pixels':<15}")
print("-" * 70)
for mask, name in zip(masks, object_names):
    obj_pixels = np.sum(mask == 0)
    total_pixels = mask.size
    coverage = 100 * obj_pixels / total_pixels
    print(f"{name:<25} {coverage:<12.1f} {obj_pixels:<15} {total_pixels:<15}")