import numpy as np
import cv2
import math
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import open3d as o3d


class PointXYZSCA:
    """扩展的点云数据结构，包含扫描位置、曲率和角度信息"""

    def __init__(self, x=0.0, y=0.0, z=0.0, scan_position=0.0, curvature=0.0, angle=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.scan_position = scan_position
        self.curvature = curvature
        self.angle = angle


class LinK3D_Extractor:
    """LinK3D特征提取器 - 用于从3D LiDAR点云中提取边缘关键点和描述符"""

    def __init__(self, n_scans=64, scan_period=0.1, minimum_range=0.1, distance_th=0.4, match_th=6):
        """
        初始化LinK3D特征提取器

        Args:
            n_scans: LiDAR扫描线数
            scan_period: 扫描周期
            minimum_range: 最小距离阈值
            distance_th: 聚类距离阈值
            match_th: 匹配阈值
        """
        self.n_scans = n_scans
        self.scan_period = scan_period
        self.minimum_range = minimum_range
        self.distance_th = distance_th
        self.match_th = match_th
        self.scan_num_th = math.ceil(n_scans / 6)
        self.pt_num_th = math.ceil(1.5 * self.scan_num_th)

    def dist_xy(self, point):
        """计算点到原点的XY平面距离"""
        return math.sqrt(point.x ** 2 + point.y ** 2)

    def dist_pt2pt(self, p1, p2):
        """计算两点之间的3D距离"""
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

    def remove_closed_point_cloud(self, points):
        """移除距离过近的点"""
        filtered_points = []
        for point in points:
            if point[0] ** 2 + point[1] ** 2 + point[2] ** 2 >= self.minimum_range ** 2:
                filtered_points.append(point)
        return np.array(filtered_points)

    def extract_edge_point(self, point_cloud):
        """从点云中提取边缘点"""
        points = np.asarray(point_cloud.points)

        # 移除距离过近的点
        points = self.remove_closed_point_cloud(points)
        if len(points) == 0:
            return [[] for _ in range(self.n_scans)]

        cloud_size = len(points)
        start_ori = -math.atan2(points[0][1], points[0][0])
        end_ori = -math.atan2(points[cloud_size - 1][1], points[cloud_size - 1][0]) + 2 * math.pi

        # 调整角度范围
        if end_ori - start_ori > 3 * math.pi:
            end_ori -= 2 * math.pi
        elif end_ori - start_ori < math.pi:
            end_ori += 2 * math.pi

        # 按扫描线分组点云
        laser_cloud_scans = [[] for _ in range(self.n_scans)]
        half_passed = False

        for i in range(cloud_size):
            point = points[i]
            x, y, z = point[0], point[1], point[2]

            # 计算扫描线ID
            angle = math.atan(z / math.sqrt(x ** 2 + y ** 2)) * 180 / math.pi
            scan_id = 0

            if self.n_scans == 16:
                scan_id = int((angle + 15) / 2 + 0.5)
            elif self.n_scans == 32:
                scan_id = int((angle + 92.0 / 3.0) * 3.0 / 4.0)
            elif self.n_scans == 64:
                if angle >= -8.83:
                    scan_id = int((2 - angle) * 3.0 + 0.5)
                else:
                    scan_id = self.n_scans // 2 + int((-8.83 - angle) * 2.0 + 0.5)

                if angle > 2 or angle < -24.33 or scan_id > 50 or scan_id < 0:
                    continue

            if scan_id >= self.n_scans or scan_id < 0:
                continue

            # 计算方位角
            ori = -math.atan2(y, x)
            if not half_passed:
                if ori < start_ori - math.pi / 2:
                    ori += 2 * math.pi
                elif ori > start_ori + math.pi * 3 / 2:
                    ori -= 2 * math.pi
                if ori - start_ori > math.pi:
                    half_passed = True
            else:
                ori += 2 * math.pi
                if ori < end_ori - math.pi * 3 / 2:
                    ori += 2 * math.pi
                elif ori > end_ori + math.pi / 2:
                    ori -= 2 * math.pi

            # 添加到对应扫描线
            point_with_intensity = [x, y, z, ori]
            laser_cloud_scans[scan_id].append(point_with_intensity)

        # 计算曲率并提取边缘点
        edge_points = [[] for _ in range(self.n_scans)]

        for i in range(self.n_scans):
            scan_size = len(laser_cloud_scans[i])
            if scan_size >= 15:
                for j in range(5, scan_size - 5):
                    # 计算曲率
                    diff_x = sum(laser_cloud_scans[i][j + k][0] for k in range(-5, 6) if k != 0) - 10 * \
                             laser_cloud_scans[i][j][0]
                    diff_y = sum(laser_cloud_scans[i][j + k][1] for k in range(-5, 6) if k != 0) - 10 * \
                             laser_cloud_scans[i][j][1]
                    diff_z = sum(laser_cloud_scans[i][j + k][2] for k in range(-5, 6) if k != 0) - 10 * \
                             laser_cloud_scans[i][j][2]

                    curv = diff_x ** 2 + diff_y ** 2 + diff_z ** 2

                    if 10 < curv < 20000:
                        ori = laser_cloud_scans[i][j][3]
                        rel_time = (ori - start_ori) / (end_ori - start_ori)

                        edge_pt = PointXYZSCA(
                            x=laser_cloud_scans[i][j][0],
                            y=laser_cloud_scans[i][j][1],
                            z=laser_cloud_scans[i][j][2],
                            scan_position=i + self.scan_period * rel_time,
                            curvature=curv,
                            angle=ori
                        )
                        edge_points[i].append(edge_pt)

        return edge_points

    def divide_area(self, scan_cloud):
        """将扫描点云分割到不同的扇形区域"""
        sector_area_cloud = [[] for _ in range(120)]  # 水平面分为120个扇形区域

        for scan_points in scan_cloud:
            for point in scan_points:
                angle = point.angle

                # 计算扇形区域ID
                if 0 < angle < 2 * math.pi:
                    area_id = int((angle / (2 * math.pi)) * 120)
                elif angle > 2 * math.pi:
                    area_id = int(((angle - 2 * math.pi) / (2 * math.pi)) * 120)
                else:  # angle < 0
                    area_id = int(((angle + 2 * math.pi) / (2 * math.pi)) * 120)

                if 0 <= area_id < 120:
                    sector_area_cloud[area_id].append(point)

        return sector_area_cloud

    def compute_cluster_mean(self, cluster):
        """计算聚类中心的平均距离"""
        if not cluster:
            return 0.0
        dist_sum = sum(self.dist_xy(point) for point in cluster)
        return dist_sum / len(cluster)

    def compute_xy_mean(self, cluster):
        """计算聚类的XY坐标均值"""
        if not cluster:
            return (0.0, 0.0)
        x_sum = sum(point.x for point in cluster)
        y_sum = sum(point.y for point in cluster)
        return (x_sum / len(cluster), y_sum / len(cluster))

    def get_cluster(self, sector_area_cloud):
        """对每个扇形区域进行聚类"""
        clusters = []

        for area_points in sector_area_cloud:
            if len(area_points) < 6:
                continue

            # 初始化聚类
            area_clusters = [[area_points[0]]]

            # 依次处理每个点
            for pt in area_points[1:]:
                added = False
                for cluster in area_clusters:
                    mean_dist = self.compute_cluster_mean(cluster)
                    xy_mean = self.compute_xy_mean(cluster)

                    if (abs(self.dist_xy(pt) - mean_dist) < self.distance_th and
                            abs(xy_mean[0] - pt.x) < self.distance_th and
                            abs(xy_mean[1] - pt.y) < self.distance_th):
                        cluster.append(pt)
                        added = True
                        break

                if not added:
                    area_clusters.append([pt])

            # 添加满足条件的聚类
            for cluster in area_clusters:
                if len(cluster) >= self.pt_num_th:
                    clusters.append(cluster)

        # 合并相邻聚类
        merged_clusters = self._merge_clusters(clusters)
        return merged_clusters

    def _merge_clusters(self, clusters):
        """合并相邻的聚类"""
        if not clusters:
            return clusters

        num_clusters = len(clusters)
        to_be_merged = [False] * num_clusters
        merge_map = defaultdict(list)

        # 找到需要合并的聚类对
        for i in range(num_clusters):
            if to_be_merged[i]:
                continue

            mean1 = self.compute_cluster_mean(clusters[i])
            xy_mean1 = self.compute_xy_mean(clusters[i])

            for j in range(i + 1, num_clusters):
                if to_be_merged[j]:
                    continue

                mean2 = self.compute_cluster_mean(clusters[j])
                xy_mean2 = self.compute_xy_mean(clusters[j])

                if (abs(mean1 - mean2) < 2 * self.distance_th and
                        abs(xy_mean1[0] - xy_mean2[0]) < 2 * self.distance_th and
                        abs(xy_mean1[1] - xy_mean2[1]) < 2 * self.distance_th):
                    merge_map[i].append(j)
                    to_be_merged[i] = True
                    to_be_merged[j] = True

        # 执行合并
        merged_clusters = []
        for i in range(num_clusters):
            if not to_be_merged[i]:
                merged_clusters.append(clusters[i])
            elif i in merge_map:
                # 合并聚类
                merged_cluster = clusters[i][:]
                for j in merge_map[i]:
                    merged_cluster.extend(clusters[j])
                merged_clusters.append(merged_cluster)

        return merged_clusters

    def get_mean_keypoint(self, clusters):
        """获取聚类的平均关键点"""
        keypoints = []
        valid_clusters = []

        for cluster in clusters:
            if len(cluster) < self.pt_num_th:
                continue

            # 检查扫描线覆盖度
            scans = set(int(pt.scan_position) for pt in cluster)
            if len(scans) < self.scan_num_th:
                continue

            # 计算平均关键点
            x = sum(pt.x for pt in cluster) / len(cluster)
            y = sum(pt.y for pt in cluster) / len(cluster)
            z = sum(pt.z for pt in cluster) / len(cluster)
            intensity = sum(pt.scan_position for pt in cluster) / len(cluster)

            keypoint = [x, y, z, intensity]
            keypoints.append(keypoint)
            valid_clusters.append(cluster)

        # 按距离排序
        keypoints_with_dist = [(kp, kp[0] ** 2 + kp[1] ** 2 + kp[2] ** 2) for kp in keypoints]
        keypoints_with_dist.sort(key=lambda x: x[1])

        sorted_keypoints = [kp[0] for kp in keypoints_with_dist]
        sorted_clusters = [valid_clusters[i] for i, _ in
                           sorted(enumerate(keypoints), key=lambda x: x[1][0] ** 2 + x[1][1] ** 2 + x[1][2] ** 2)]

        return sorted_keypoints, sorted_clusters

    def f_round(self, value):
        """四舍五入到一位小数"""
        return round(value * 10) / 10.0

    def get_descriptors(self, keypoints):
        """为关键点生成描述符"""
        if not keypoints:
            return np.array([])

        pt_size = len(keypoints)
        descriptors = np.zeros((pt_size, 180), dtype=np.float32)

        # 构建距离和方向表
        distance_tab = np.zeros((pt_size, pt_size))
        direction_tab = np.zeros((pt_size, pt_size, 2))

        for i in range(pt_size):
            for j in range(i + 1, pt_size):
                dist = self.dist_pt2pt_array(keypoints[i], keypoints[j])
                distance_tab[i][j] = distance_tab[j][i] = self.f_round(dist)

                direction = np.array([keypoints[j][0] - keypoints[i][0], keypoints[j][1] - keypoints[i][1]])
                direction_tab[i][j] = direction
                direction_tab[j][i] = -direction

        # 为每个关键点生成描述符
        for i in range(pt_size):
            distances = distance_tab[i].copy()
            distances[i] = float('inf')  # 排除自身
            sorted_indices = np.argsort(distances)

            # 选择最近的3个关键点
            for k in range(min(3, len(sorted_indices))):
                if distances[sorted_indices[k]] == float('inf'):
                    break

                index = sorted_indices[k]
                main_direction = direction_tab[i][index]

                area_distances = [[] for _ in range(180)]
                area_distances[0].append(distance_tab[i][index])

                for j in range(pt_size):
                    if j == i or j == index:
                        continue

                    other_direction = direction_tab[i][j]

                    # 计算角度
                    dot_product = np.dot(main_direction, other_direction)
                    norm_product = np.linalg.norm(main_direction) * np.linalg.norm(other_direction)

                    if norm_product == 0:
                        continue

                    cos_angle = dot_product / norm_product
                    cos_angle = np.clip(cos_angle, -1, 1)

                    angle = math.acos(cos_angle) * 180 / math.pi

                    # 计算行列式判断方向
                    det = main_direction[0] * other_direction[1] - main_direction[1] * other_direction[0]

                    if det > 0:
                        area_num = math.ceil((angle - 1) / 2) if angle > 1 else 0
                    else:
                        if angle <= 2:
                            area_num = 0
                        else:
                            angle = 360 - angle
                            area_num = math.ceil((angle - 1) / 2)

                    if 0 < area_num < 180:
                        area_distances[area_num].append(distance_tab[i][j])

                # 填充描述符
                for area_num in range(180):
                    if area_distances[area_num] and descriptors[i][area_num] == 0:
                        descriptors[i][area_num] = min(area_distances[area_num])

        return descriptors

    def dist_pt2pt_array(self, p1, p2):
        """计算数组形式点的距离"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def match(self, cur_keypoints, to_be_matched_keypoints, cur_descriptors, to_be_matched_descriptors):
        """匹配两组关键点"""
        matched_indices = []

        if len(cur_keypoints) == 0 or len(to_be_matched_keypoints) == 0:
            return matched_indices

        # 为每个当前关键点找到最佳匹配
        matched_scores = {}
        potential_matches = defaultdict(list)

        for i in range(len(cur_keypoints)):
            best_score = 0
            best_match = -1

            for j in range(len(to_be_matched_keypoints)):
                score = 0
                # 计算描述符匹配分数
                for bit_num in range(180):
                    if (cur_descriptors[i][bit_num] != 0 and
                            to_be_matched_descriptors[j][bit_num] != 0 and
                            abs(cur_descriptors[i][bit_num] - to_be_matched_descriptors[j][bit_num]) <= 0.2):
                        score += 1

                    # 提前终止条件
                    if bit_num > 90 and score < 3:
                        break

                if score > best_score:
                    best_score = score
                    best_match = j

            if best_score >= self.match_th:
                matched_scores[i] = best_score
                potential_matches[best_match].append(i)

        # 处理一对多匹配，保留最高分的匹配
        for j, candidates in potential_matches.items():
            if len(candidates) == 1:
                i = candidates[0]
                if matched_scores[i] >= self.match_th:
                    matched_indices.append((i, j))
            else:
                # 选择分数最高的匹配
                best_candidate = max(candidates, key=lambda x: matched_scores[x])
                if matched_scores[best_candidate] >= self.match_th:
                    matched_indices.append((best_candidate, j))

        return matched_indices

    def filter_low_curv(self, clusters):
        """过滤低曲率的边缘关键点"""
        filtered = []

        for cluster in clusters:
            filtered_cluster = []
            scan_groups = defaultdict(list)

            # 按扫描线分组
            for pt in cluster:
                scan_id = int(pt.scan_position)
                scan_groups[scan_id].append(pt)

            # 为每个扫描线选择最高曲率的点
            for scan_points in scan_groups.values():
                if len(scan_points) == 1:
                    filtered_cluster.append(scan_points[0])
                else:
                    max_curv_pt = max(scan_points, key=lambda p: p.curvature)
                    filtered_cluster.append(max_curv_pt)

            filtered.append(filtered_cluster)

        return filtered

    def find_edge_keypoint_match(self, filtered1, filtered2, matched_indices):
        """基于聚合关键点的匹配结果找到边缘关键点匹配"""
        match_points = []

        for i, j in matched_indices:
            if i >= len(filtered1) or j >= len(filtered2):
                continue

            # 构建扫描线ID到索引的映射
            scan_index_map1 = {int(pt.scan_position): idx for idx, pt in enumerate(filtered1[i])}
            scan_index_map2 = {int(pt.scan_position): idx for idx, pt in enumerate(filtered2[j])}

            # 找到相同扫描线的匹配点
            for scan_id in scan_index_map1:
                if scan_id in scan_index_map2:
                    pt1 = filtered1[i][scan_index_map1[scan_id]]
                    pt2 = filtered2[j][scan_index_map2[scan_id]]
                    match_points.append((pt1, pt2))

        return match_points

    def __call__(self, point_cloud):
        """主要处理函数 - 提取关键点和描述符"""
        # 提取边缘点
        edge_points = self.extract_edge_point(point_cloud)

        # 分割区域
        sector_area_cloud = self.divide_area(edge_points)

        # 聚类
        clusters = self.get_cluster(sector_area_cloud)

        # 获取关键点
        keypoints, valid_clusters = self.get_mean_keypoint(clusters)

        # 生成描述符
        descriptors = self.get_descriptors(keypoints)

        return keypoints, descriptors, valid_clusters