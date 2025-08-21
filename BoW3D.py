import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import math
from Frame import Frame
from LinK3D_Extractor import LinK3D_Extractor, PointXYZSCA


class BoW3D:
    """BoW3D词袋模型类 - 用于3D LiDAR SLAM中的实时回环检测"""

    def __init__(self, link3d_extractor, thr=3.5, thf=5, num_add_retrieve_features=5):
        """
        初始化BoW3D词袋模型

        Args:
            link3d_extractor: LinK3D特征提取器实例
            thr: 比率阈值，用于过滤高频词汇
            thf: 频率阈值，用于回环检测
            num_add_retrieve_features: 每帧添加或检索的特征数量
        """
        self.link3d_extractor = link3d_extractor
        self.thr = thr  # 比率阈值
        self.thf = thf  # 频率阈值
        self.num_add_retrieve_features = num_add_retrieve_features

        # 词汇表：(维度值, 维度ID) -> {(帧ID, 描述符ID), ...}
        self.vocabulary = defaultdict(set)

        # 统计信息：用于计算比率
        self.n_nw_ratio = [0, 0]  # [新词数量, 总词数量]

        # 存储所有帧
        self.frames = []

    def update(self, current_frame):
        """
        更新词汇表with当前帧的特征

        Args:
            current_frame: Frame对象
        """
        self.frames.append(current_frame)

        descriptors = current_frame.descriptors
        frame_id = current_frame.id

        if descriptors.size == 0:
            return

        num_features = descriptors.shape[0]

        # 确定要处理的特征数量
        features_to_process = min(num_features, self.num_add_retrieve_features)

        for i in range(features_to_process):
            descriptor = descriptors[i]

            # 遍历描述符的每个维度
            for j in range(len(descriptor)):
                if descriptor[j] != 0:
                    # 创建词汇：(维度值, 维度ID)
                    word = (descriptor[j], j)
                    place = (frame_id, i)

                    # 检查是否为新词汇
                    if word not in self.vocabulary:
                        self.vocabulary[word] = set()
                        self.n_nw_ratio[0] += 1  # 新词数量+1

                    # 添加位置信息
                    self.vocabulary[word].add(place)
                    self.n_nw_ratio[1] += 1  # 总词数量+1

    def retrieve(self, current_frame):
        """
        检索可能的回环帧

        Args:
            current_frame: Frame对象

        Returns:
            tuple: (回环帧ID, 相对旋转矩阵, 相对平移向量) 或 (-1, None, None)
        """
        frame_id = current_frame.id
        descriptors = current_frame.descriptors

        if descriptors.size == 0:
            return -1, None, None

        # 候选帧评分
        score_frame_map = {}

        num_features = descriptors.shape[0]
        features_to_process = min(num_features, self.num_add_retrieve_features)

        for i in range(features_to_process):
            descriptor = descriptors[i]
            place_scores = defaultdict(int)

            # 遍历描述符的每个维度
            for j in range(len(descriptor)):
                if descriptor[j] != 0:
                    word = (descriptor[j], j)

                    if word not in self.vocabulary:
                        continue

                    # 计算词汇频率比率
                    if self.n_nw_ratio[0] > 0:
                        avg_places_per_word = self.n_nw_ratio[1] / self.n_nw_ratio[0]
                        current_places_count = len(self.vocabulary[word])
                        ratio = current_places_count / avg_places_per_word

                        # 过滤高频词汇
                        if ratio > self.thr:
                            continue

                    # 为每个位置计分
                    for place in self.vocabulary[word]:
                        place_frame_id, place_descriptor_id = place

                        # 确保回环间隔至少300帧
                        if frame_id - place_frame_id < 300:
                            continue

                        place_scores[place] += 1

            # 统计超过阈值的候选帧
            for place, score in place_scores.items():
                if score > self.thf:
                    place_frame_id = place[0]
                    if score not in score_frame_map or score_frame_map[score] < place_frame_id:
                        score_frame_map[score] = place_frame_id

        if not score_frame_map:
            return -1, None, None

        # 按分数从高到低检查候选帧
        for score in sorted(score_frame_map.keys(), reverse=True):
            loop_frame_id = score_frame_map[score]
            loop_frame = self.frames[loop_frame_id]

            # 进行特征匹配
            matched_indices = self.link3d_extractor.match(
                current_frame.aggregation_keypoints,
                loop_frame.aggregation_keypoints,
                current_frame.descriptors,
                loop_frame.descriptors
            )

            # 进行回环校正
            result, loop_rel_r, loop_rel_t = self.loop_correction(
                current_frame, loop_frame, matched_indices
            )

            # 检查距离约束（小于3米且大于0）
            if (result != -1 and loop_rel_t is not None and
                    np.linalg.norm(loop_rel_t) < 3 and np.linalg.norm(loop_rel_t) > 0):
                return loop_frame_id, loop_rel_r, loop_rel_t

        return -1, None, None

    def loop_correction(self, current_frame, matched_frame, matched_indices):
        """
        执行回环校正，计算相对位姿

        Args:
            current_frame: 当前帧
            matched_frame: 匹配帧
            matched_indices: 匹配的关键点索引对列表

        Returns:
            tuple: (状态码, 旋转矩阵, 平移向量)
        """
        if len(matched_indices) <= 30:
            return -1, None, None

        # 过滤低曲率点
        current_filtered = self.link3d_extractor.filter_low_curv(current_frame.cluster_edge_keypoints)
        matched_filtered = self.link3d_extractor.filter_low_curv(matched_frame.cluster_edge_keypoints)

        # 找到边缘关键点匹配
        matched_edge_points = self.link3d_extractor.find_edge_keypoint_match(
            current_filtered, matched_filtered, matched_indices
        )

        if len(matched_edge_points) <= 100:
            return -1, None, None

        # 准备点云数据用于RANSAC
        source_points = []
        target_points = []

        for pt1, pt2 in matched_edge_points:
            source_points.append([pt1.x, pt1.y, pt1.z])
            target_points.append([pt2.x, pt2.y, pt2.z])

        source_points = np.array(source_points)
        target_points = np.array(target_points)

        # 简化版RANSAC（这里可以使用更复杂的RANSAC实现）
        inliers = self._ransac_correspondence_rejection(source_points, target_points, threshold=0.4)

        if len(inliers) <= 100:
            return -1, None, None

        # 使用内点计算位姿
        inlier_source = source_points[inliers]
        inlier_target = target_points[inliers]

        # 计算质心
        center1 = np.mean(inlier_source, axis=0)
        center2 = np.mean(inlier_target, axis=0)

        # 去质心
        centered_source = inlier_source - center1
        centered_target = inlier_target - center2

        # 计算协方差矩阵
        H = centered_source.T @ centered_target

        # SVD分解求解旋转矩阵
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 确保旋转矩阵的行列式为正
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # 计算平移向量
        t = center2 - R @ center1

        return 1, R, t

    def _ransac_correspondence_rejection(self, source_points, target_points, threshold=0.4, max_iterations=1000):
        """
        简化版RANSAC对应点剔除

        Args:
            source_points: 源点云
            target_points: 目标点云
            threshold: 内点阈值
            max_iterations: 最大迭代次数

        Returns:
            list: 内点索引列表
        """
        best_inliers = []
        n_points = len(source_points)

        if n_points < 3:
            return list(range(n_points))

        for _ in range(max_iterations):
            # 随机选择3个点
            sample_indices = np.random.choice(n_points, 3, replace=False)
            sample_source = source_points[sample_indices]
            sample_target = target_points[sample_indices]

            # 计算变换矩阵
            try:
                center_s = np.mean(sample_source, axis=0)
                center_t = np.mean(sample_target, axis=0)

                centered_s = sample_source - center_s
                centered_t = sample_target - center_t

                H = centered_s.T @ centered_t
                U, _, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T

                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = Vt.T @ U.T

                t = center_t - R @ center_s

                # 计算所有点的误差
                transformed_source = (R @ source_points.T).T + t
                errors = np.linalg.norm(transformed_source - target_points, axis=1)

                # 找到内点
                inliers = np.where(errors < threshold)[0]

                if len(inliers) > len(best_inliers):
                    best_inliers = inliers

            except np.linalg.LinAlgError:
                continue

        return best_inliers.tolist()

    def get_vocabulary_stats(self):
        """获取词汇表统计信息"""
        total_words = len(self.vocabulary)
        total_places = sum(len(places) for places in self.vocabulary.values())
        avg_places_per_word = total_places / total_words if total_words > 0 else 0

        return {
            "total_words": total_words,
            "total_places": total_places,
            "avg_places_per_word": avg_places_per_word,
            "n_nw_ratio": self.n_nw_ratio.copy()
        }

    def __str__(self):
        stats = self.get_vocabulary_stats()
        return (f"BoW3D: {len(self.frames)} 帧, "
                f"{stats['total_words']} 词汇, "
                f"{stats['total_places']} 位置")