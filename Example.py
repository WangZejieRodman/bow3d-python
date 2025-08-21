#!/usr/bin/env python3
"""
BoW3D示例程序 - 基于KITTI数据集的3D LiDAR SLAM回环检测演示

这个程序演示了如何使用BoW3D进行实时回环检测：
1. 读取KITTI数据集的.bin格式点云文件
2. 使用LinK3D提取器提取特征
3. 使用BoW3D进行回环检测
4. 输出检测结果和相对位姿
"""

import os
import sys
import time
import glob
import numpy as np
import open3d as o3d
from pathlib import Path

# 导入自定义模块
from LinK3D_Extractor import LinK3D_Extractor
from Frame import Frame
from BoW3D import BoW3D


class KittiDataLoader:
    """KITTI数据集加载器"""

    def __init__(self, dataset_path, sequence="00"):
        """
        初始化KITTI数据加载器

        Args:
            dataset_path: KITTI数据集根路径
            sequence: 序列号 (如 "00", "02", "05" 等)
        """
        self.dataset_path = Path(dataset_path)
        self.sequence = sequence
        self.velodyne_path = self.dataset_path / sequence / "velodyne"

        # 检查路径是否存在
        if not self.velodyne_path.exists():
            raise FileNotFoundError(f"KITTI数据路径不存在: {self.velodyne_path}")

        # 获取所有.bin文件
        self.bin_files = sorted(glob.glob(str(self.velodyne_path / "*.bin")))
        print(f"找到 {len(self.bin_files)} 个点云文件")

    def read_bin_file(self, file_path):
        """
        读取KITTI的.bin格式点云文件

        Args:
            file_path: .bin文件路径

        Returns:
            numpy.ndarray: 点云数据 (N, 4) - [x, y, z, intensity]
        """
        try:
            # 读取二进制文件
            points = np.fromfile(file_path, dtype=np.float32)

            # 重塑为 (N, 4) 格式：x, y, z, intensity
            points = points.reshape(-1, 4)

            return points

        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return np.array([])

    def get_point_cloud(self, index):
        """
        获取指定索引的点云数据

        Args:
            index: 点云文件索引

        Returns:
            open3d.geometry.PointCloud: Open3D点云对象
        """
        if index >= len(self.bin_files):
            return None

        # 读取.bin文件
        points_with_intensity = self.read_bin_file(self.bin_files[index])

        if len(points_with_intensity) == 0:
            return None

        # 提取XYZ坐标（忽略intensity）
        points_xyz = points_with_intensity[:, :3]

        # 创建Open3D点云对象
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_xyz)

        return point_cloud

    def __len__(self):
        return len(self.bin_files)


def main():
    """主函数"""
    print("=== BoW3D 3D LiDAR SLAM 回环检测演示 ===")

    # ========================
    # 1. 参数设置
    # ========================

    # LinK3D参数
    n_scans = 64  # LiDAR扫描线数
    scan_period = 0.1  # 扫描周期
    minimum_range = 0.1  # 最小距离
    distance_th = 0.4  # 距离阈值
    match_th = 6  # 匹配阈值

    # BoW3D参数
    thr = 3.5  # 比率阈值
    thf = 5  # 频率阈值
    num_add_retrieve_features = 5  # 每帧特征数量

    # 数据集路径设置（请修改为您的KITTI数据集路径）
    sequence = "00"  # 可以改为 00, 02, 05, 06, 07, 08

    # 尝试几个常见的KITTI数据集路径
    possible_paths = [
        f"/home/{os.getenv('USER', 'user')}/pan1/Data/KITTI",
    ]

    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break

    if dataset_path is None:
        print("错误: 未找到KITTI数据集")
        print("请将KITTI数据集放置在以下路径之一:")
        for path in possible_paths:
            print(f"  - {path}")
        print("或修改脚本中的数据集路径")
        return

    print(f"数据集路径: {dataset_path}/{sequence}")

    # ========================
    # 2. 初始化组件
    # ========================

    try:
        # 初始化数据加载器
        data_loader = KittiDataLoader(dataset_path, sequence)

        # 初始化LinK3D特征提取器
        link3d_extractor = LinK3D_Extractor(
            n_scans=n_scans,
            scan_period=scan_period,
            minimum_range=minimum_range,
            distance_th=distance_th,
            match_th=match_th
        )

        # 初始化BoW3D
        bow3d = BoW3D(
            link3d_extractor=link3d_extractor,
            thr=thr,
            thf=thf,
            num_add_retrieve_features=num_add_retrieve_features
        )

        print("组件初始化完成")

    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # ========================
    # 3. 主处理循环
    # ========================

    print("开始处理点云数据...")
    print("-" * 60)

    # 统计信息
    total_frames = 0
    loop_detections = 0
    total_processing_time = 0

    # 设置处理频率（模拟10Hz）
    target_frequency = 10.0
    target_period = 1.0 / target_frequency

    for cloud_index in range(len(data_loader)):
        start_time = time.time()

        # 读取点云数据
        point_cloud = data_loader.get_point_cloud(cloud_index)
        if point_cloud is None:
            print(f"跳过无效点云: {cloud_index}")
            continue

        print(f"处理帧 {cloud_index}: {len(point_cloud.points)} 个点")

        try:
            # 创建Frame对象并提取特征
            current_frame = Frame(link3d_extractor, point_cloud)

            if not current_frame.has_valid_features():
                print(f"  帧 {cloud_index}: 特征提取失败，跳过")
                continue

            total_frames += 1

            # 前两帧仅用于初始化
            if current_frame.id < 2:
                bow3d.update(current_frame)
                print(f"  帧 {current_frame.id}: 初始化阶段")
            else:
                # 进行回环检测
                detection_start = time.time()

                loop_frame_id, loop_rel_r, loop_rel_t = bow3d.retrieve(current_frame)

                detection_time = time.time() - detection_start
                total_processing_time += detection_time

                # 更新词汇表
                bow3d.update(current_frame)

                # 输出结果
                if loop_frame_id == -1:
                    print("  " + "=" * 25)
                    print(f"  检测时间: {detection_time:.4f}s")
                    print(f"  帧 {current_frame.id}: 未检测到回环")
                else:
                    loop_detections += 1
                    print("  " + "=" * 35)
                    print(f"  检测时间: {detection_time:.4f}s")
                    print(f"  帧 {current_frame.id}: 检测到回环帧 {loop_frame_id}")

                    if loop_rel_r is not None and loop_rel_t is not None:
                        print("  回环相对旋转矩阵:")
                        for row in loop_rel_r:
                            print(f"    [{row[0]:8.5f} {row[1]:8.5f} {row[2]:8.5f}]")

                        print("  回环相对平移向量:")
                        print(f"    [{loop_rel_t[0]:8.5f} {loop_rel_t[1]:8.5f} {loop_rel_t[2]:8.5f}]")
                        print(f"  平移距离: {np.linalg.norm(loop_rel_t):.3f}m")

            # 显示当前状态
            keypoint_count = current_frame.get_keypoint_count()
            cluster_info = current_frame.get_cluster_info()
            vocab_stats = bow3d.get_vocabulary_stats()

            print(f"  关键点: {keypoint_count}, "
                  f"聚类: {cluster_info['cluster_count']}, "
                  f"词汇: {vocab_stats['total_words']}")

        except Exception as e:
            print(f"  处理帧 {cloud_index} 时出错: {e}")
            continue

        # 控制处理频率
        processing_time = time.time() - start_time
        if processing_time < target_period:
            time.sleep(target_period - processing_time)

        print(f"  总耗时: {time.time() - start_time:.4f}s")
        print("-" * 60)

        # 可选：限制处理帧数用于测试
        # if total_frames >= 100:
        #     break

    # ========================
    # 4. 输出统计信息
    # ========================

    print("\n" + "=" * 60)
    print("处理完成 - 统计信息:")
    print("=" * 60)
    print(f"总处理帧数: {total_frames}")
    print(f"回环检测次数: {loop_detections}")
    print(f"回环检测率: {loop_detections / max(total_frames - 2, 1) * 100:.2f}%")
    print(f"平均检测时间: {total_processing_time / max(total_frames - 2, 1):.4f}s")

    # 最终词汇表统计
    final_stats = bow3d.get_vocabulary_stats()
    print(f"最终词汇表大小: {final_stats['total_words']}")
    print(f"总位置数: {final_stats['total_places']}")
    print(f"平均每词位置数: {final_stats['avg_places_per_word']:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback

        traceback.print_exc()