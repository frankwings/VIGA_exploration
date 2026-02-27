import numpy as np

masks = np.load(r'D:\Projects\ProjectGenesis\GenesisVIGA\output\test_sam\all_masks.npy')
print(f'Shape: {masks.shape}')
print(f'Unique values: {np.unique(masks)}')
print(f'Coverage percentages:')
for i, m in enumerate(masks):
    print(f'{i}: {100 * np.sum(m == 0) / m.size:.1f}%')