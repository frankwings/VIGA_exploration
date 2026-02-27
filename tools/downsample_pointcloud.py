#!/usr/bin/env python3
"""
点云降采样工具
支持 PLY 和 GLB 格式

用法:
    python downsample_pointcloud.py input.ply --voxel 0.02 -o output.ply
    python downsample_pointcloud.py input.ply --uniform 2 -o output.ply
"""

import argparse
import numpy as np
from pathlib import Path

def load_ply(path: str):
    """加载 PLY 点云"""
    import open3d as o3d
    return o3d.io.read_point_cloud(str(path))

def load_glb_points(path: str):
    """从 GLB 提取顶点作为点云"""
    import trimesh
    import open3d as o3d
    
    mesh = trimesh.load(str(path))
    if isinstance(mesh, trimesh.Scene):
        # 合并所有 geometry
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError("GLB 中没有找到 mesh")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    
    # 如果有顶点颜色
    if mesh.visual.vertex_colors is not None:
        colors = mesh.visual.vertex_colors[:, :3] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def voxel_downsample(pcd, voxel_size: float):
    """体素降采样"""
    return pcd.voxel_down_sample(voxel_size)

def uniform_downsample(pcd, every_k: int):
    """均匀降采样 (每k个点取1个)"""
    return pcd.uniform_down_sample(every_k)

def random_downsample(pcd, ratio: float):
    """随机降采样"""
    import open3d as o3d
    
    points = np.asarray(pcd.points)
    n_points = len(points)
    n_keep = int(n_points * ratio)
    
    indices = np.random.choice(n_points, n_keep, replace=False)
    indices = np.sort(indices)
    
    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(points[indices])
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        pcd_down.colors = o3d.utility.Vector3dVector(colors[indices])
    
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        pcd_down.normals = o3d.utility.Vector3dVector(normals[indices])
    
    return pcd_down

def save_ply(pcd, path: str):
    """保存为 PLY"""
    import open3d as o3d
    o3d.io.write_point_cloud(str(path), pcd)

def main():
    parser = argparse.ArgumentParser(description="点云降采样工具")
    parser.add_argument("input", help="输入文件 (PLY 或 GLB)")
    parser.add_argument("-o", "--output", help="输出文件 (PLY)")
    parser.add_argument("--voxel", type=float, help="体素降采样大小 (推荐 0.01-0.05)")
    parser.add_argument("--uniform", type=int, help="均匀降采样，每k个点取1个")
    parser.add_argument("--random", type=float, help="随机降采样比例 (0-1)")
    parser.add_argument("--info", action="store_true", help="只显示点云信息")
    
    args = parser.parse_args()
    
    # 加载点云
    input_path = Path(args.input)
    print(f"📂 加载: {input_path}")
    
    if input_path.suffix.lower() == ".glb":
        pcd = load_glb_points(args.input)
    else:
        pcd = load_ply(args.input)
    
    n_original = len(pcd.points)
    print(f"   原始点数: {n_original:,}")
    
    if args.info:
        # 只显示信息
        points = np.asarray(pcd.points)
        print(f"   范围 X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"   范围 Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"   范围 Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        print(f"   有颜色: {pcd.has_colors()}")
        print(f"   有法线: {pcd.has_normals()}")
        return
    
    # 降采样
    if args.voxel:
        print(f"🔽 体素降采样 (voxel_size={args.voxel})")
        pcd = voxel_downsample(pcd, args.voxel)
    elif args.uniform:
        print(f"🔽 均匀降采样 (every_k={args.uniform})")
        pcd = uniform_downsample(pcd, args.uniform)
    elif args.random:
        print(f"🔽 随机降采样 (ratio={args.random})")
        pcd = random_downsample(pcd, args.random)
    else:
        print("⚠️ 未指定降采样方法 (--voxel, --uniform, 或 --random)")
        return
    
    n_after = len(pcd.points)
    ratio = n_after / n_original * 100
    print(f"   降采样后: {n_after:,} ({ratio:.1f}%)")
    
    # 保存
    if args.output:
        output_path = Path(args.output)
        print(f"💾 保存: {output_path}")
        save_ply(pcd, args.output)
        print("✅ 完成")
    else:
        print("⚠️ 未指定输出文件 (-o)")

if __name__ == "__main__":
    main()
