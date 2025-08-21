import numpy as np
import open3d as o3d
from LinK3D_Extractor import LinK3D_Extractor


class Frame:
    """帧类 - 存储单帧LiDAR数据的关键点、描述符和边缘点聚类信息"""

    # 静态变量，用于生成唯一帧ID
    next_id = 0

    def __init__(self, link3d_extractor=None, point_cloud=None):
        """
        初始化帧对象

        Args:
            link3d_extractor: LinK3D特征提取器实例
            point_cloud: 输入的点云数据 (Open3D PointCloud对象)
        """
        # 分配唯一的帧ID
        self.id = Frame.next_id
        Frame.next_id += 1

        # 存储特征提取器引用
        self.link3d_extractor = link3d_extractor

        # 初始化数据成员
        self.cluster_edge_keypoints = []  # 聚类边缘关键点
        self.aggregation_keypoints = []  # 聚合关键点
        self.descriptors = np.array([])  # 描述符矩阵

        # 如果提供了特征提取器和点云，则进行特征提取
        if link3d_extractor is not None and point_cloud is not None:
            self._extract_features(point_cloud)

    def _extract_features(self, point_cloud):
        """
        从点云中提取特征

        Args:
            point_cloud: Open3D PointCloud对象
        """
        try:
            # 使用LinK3D提取器提取关键点、描述符和有效聚类
            keypoints, descriptors, valid_clusters = self.link3d_extractor(point_cloud)

            # 存储提取的特征
            self.aggregation_keypoints = keypoints
            self.descriptors = descriptors
            self.cluster_edge_keypoints = valid_clusters

            print(f"帧 {self.id}: 提取到 {len(keypoints)} 个关键点")

        except Exception as e:
            print(f"帧 {self.id} 特征提取失败: {e}")
            # 初始化为空数据
            self.aggregation_keypoints = []
            self.descriptors = np.array([])
            self.cluster_edge_keypoints = []

    def get_keypoint_count(self):
        """获取关键点数量"""
        return len(self.aggregation_keypoints)

    def get_descriptor_shape(self):
        """获取描述符矩阵的形状"""
        return self.descriptors.shape if self.descriptors.size > 0 else (0, 0)

    def has_valid_features(self):
        """检查是否有有效的特征数据"""
        return (len(self.aggregation_keypoints) > 0 and
                self.descriptors.size > 0 and
                len(self.cluster_edge_keypoints) > 0)

    def get_keypoints_as_array(self):
        """获取关键点的numpy数组形式"""
        if not self.aggregation_keypoints:
            return np.array([])
        return np.array(self.aggregation_keypoints)

    def get_cluster_info(self):
        """获取聚类信息统计"""
        if not self.cluster_edge_keypoints:
            return {"cluster_count": 0, "total_points": 0, "avg_points_per_cluster": 0}

        cluster_count = len(self.cluster_edge_keypoints)
        total_points = sum(len(cluster) for cluster in self.cluster_edge_keypoints)
        avg_points = total_points / cluster_count if cluster_count > 0 else 0

        return {
            "cluster_count": cluster_count,
            "total_points": total_points,
            "avg_points_per_cluster": avg_points
        }

    def __str__(self):
        """字符串表示"""
        cluster_info = self.get_cluster_info()
        descriptor_shape = self.get_descriptor_shape()

        return (f"Frame {self.id}: "
                f"{len(self.aggregation_keypoints)} 关键点, "
                f"描述符形状 {descriptor_shape}, "
                f"{cluster_info['cluster_count']} 个聚类")

    def __repr__(self):
        return self.__str__()