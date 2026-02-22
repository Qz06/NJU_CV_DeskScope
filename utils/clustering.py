# utils/clustering.py


import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from enum import Enum

from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter1d


# ============================================================
#                      数据结构定义
# ============================================================

class ZoneType(Enum):
    """区域类型枚举"""
    STUDY = "学习区"
    WORK = "工作区"
    LEISURE = "休闲区"
    DINING = "饮食区"
    CLUTTER = "杂物区"
    CLEANUP = "待清理区"


@dataclass
class ClusterZone:
    """聚类区域数据结构"""
    id: int                                      # 区域 ID
    zone_type: str                               # 区域类型名称
    objects: List[Dict]                          # 区域内的物品列表
    object_indices: List[int]                    # 物品在原列表中的索引
    center: Tuple[float, float]                  # 区域中心点（像素坐标）
    boundary: np.ndarray                         # Alpha Shape 边界点
    color: Tuple[int, int, int]                  # 显示颜色 (BGR)
    icon: str                                    # 图标 emoji
    note: str = ""                               # 备注
    category_distribution: Dict[str, float] = field(default_factory=dict)
    is_singleton: bool = False                   # 是否为单物品区域
    source: str = "dbscan"                       # 来源: 'dbscan', 'kmeans', 'merge', 'isolated'


@dataclass
class ZoneClusteringResult:
    """聚类结果数据结构"""
    zones: List[ClusterZone]                     # 所有区域
    outliers: List[Dict] = field(default_factory=list)  # 未分配的离群点
    params: Dict = field(default_factory=dict)  # 使用的参数
    debug_info: Dict = field(default_factory=dict)  # 调试信息


@dataclass
class ZoneClusteringConfig:
    """聚类配置参数"""
    # 特征融合权重
    position_weight: float = 0.75                # 位置权重 (语义权重 = 1 - position_weight) - 提高位置权重让分区更精细
    
    # 语义距离参数
    same_class_distance: float = 0.0             # 同类别距离
    same_superclass_distance: float = 0.2        # 同 super_class 距离
    same_topcategory_distance: float = 0.5       # 同顶级类别距离
    different_category_distance: float = 1.0    # 不同类别距离
    garbage_penalty: float = 0.3                 # 垃圾额外惩罚
    electronics_bonus: float = 0.1               # 电子设备聚合奖励
    
    # DBSCAN 参数
    eps_percentile: int = 15                     # 距离分位数（降低让聚类更严格，分区更精细）
    min_samples: int = 2                         # 最小邻居数
    
    # 离群点处理
    merge_threshold_ratio: float = 1.8           # 合并阈值 = eps × ratio（降低让合并更谨慎）
    kmeans_min_points: int = 2                   # K-Means 最小点数
    kmeans_max_clusters: int = 5                 # K-Means 最大聚类数（增加允许更多小区域）
    
    # Alpha Shape
    alpha_factor: float = 0.8                    # alpha = 1 / (median_dist × factor) - 降低让边界更宽松包容
    boundary_padding_ratio: float = 0.02         # 边界向外扩展比例（相对于图片对角线）
    boundary_smooth_sigma: float = 1.5           # 边界平滑参数（适度平滑）
    
    # 可视化
    fill_alpha: float = 0.3                      # 填充透明度
    boundary_thickness: int = 3                  # 边界线粗细


# ============================================================
#                   Super Class 定义（用于语义距离计算）
# ============================================================
# 与 analysis.py 中的 super_classes 保持同步

SUPER_CLASSES = {
    # 学习相关物品
    'learning': {
        'book', 'document', 'paper',
        'pen', 'eraser', 'ruler', 'scissors',
        'sticky notes', 'paper clip', 'stapler',
        'calculator', 'pencil case'
    },
    # 电子设备
    'electronics': {
        'computer', 'laptop', 'monitor', 'keyboard', 'mouse',
        'tablet', 'mobile phone', 'printer',
        'headphones', 'earphones', 'speaker'
    },
    # 电源与线缆
    'powerAndCable': {
        'charger', 'cable', 'socket', 'power bank'
    },
    # 饮食相关
    'foodAndDrink': {
        'cup', 'thermos cup', 'plastic cup',
        'bottle', 'plastic bottle', 'can', 'disposable cup',
        'bowl', 'waterpot', 'squeeze bottle',
        'apple', 'banana', 'orange',
        'instant noodles', 'takeout',
        'snack box', 'food container', 'carton'
    },
    # 垃圾与脏乱物品
    'garbageAndDirty': {
        'snack leftovers', 'food wrapper',
        'waste paperball', 'waste package',
        'tissue', 'plastic bag',
        'bottle cap', 'straw',
        'broken items', 'debris'
    },
    # 小物件
    'smallStuff': {
        'keys', 'wallet', 'glasses', 'watch',
        'card', 'comb', 'mirror',
        'mouse pad', 'phone holder'
    },
    # 娱乐物品
    'entertainment': {
        'toy', 'remote controller', 'gamepad'
    },
    # 医药物品
    'medicine': {
        'ointment', 'mask'
    }
}

# 构建类别到 super_class 的映射（用于语义距离）
NAME_TO_SUPER = {}
for super_name, sub_set in SUPER_CLASSES.items():
    for name in sub_set:
        NAME_TO_SUPER[name] = super_name

# 顶级类别映射（用于更粗粒度的语义距离）
TOP_CATEGORIES = {
    'study': ['learning'],
    'work': ['electronics', 'powerAndCable'],
    'leisure': ['foodAndDrink', 'smallStuff', 'entertainment'],
    'mess': ['garbageAndDirty'],
    'health': ['medicine']
}

# 反向映射：super_class -> top_category
SUPER_TO_TOP = {}
for top_cat, super_list in TOP_CATEGORIES.items():
    for super_class in super_list:
        SUPER_TO_TOP[super_class] = top_cat


# ============================================================
#           功能区划分定义（用于区域类型推断）
# ============================================================
# 从 analysis.py 中的 partition_zones 同步
# 注意：功能区划分和语义距离使用不同的分类体系

PARTITION_ZONES = {
    # 主要用于学习、阅读、做题等行为
    'study_zone': {
        'book', 'document', 'paper',
        'pen', 'eraser', 'ruler', 'scissors',
        'sticky notes', 'paper clip', 'stapler',
        'calculator', 'pencil case'
    },
    # 主要用于办公、计算、信息处理等工作行为
    'work_zone': {
        'computer', 'laptop', 'monitor',
        'keyboard', 'mouse', 'printer',
        'tablet', 'lamp', 'mouse pad',
        'phone holder'
    },
    # 主要用于娱乐、放松、休闲行为
    'relax_zone': {
        'headphones', 'earphones', 'speaker',
        'toy', 'remote controller', 'gamepad',
        'mobile phone', 'fan', 'clock'
    },
    # 与进食、饮水直接相关的物品
    'food_zone': {
        'cup', 'thermos cup', 'plastic cup',
        'bottle', 'plastic bottle', 'can',
        'disposable cup', 'bowl', 'waterpot',
        'squeeze bottle',
        'apple', 'banana', 'orange',
        'instant noodles', 'takeout',
        'chopsticks', 'food container'
    },
    # 个人随身物品、小型杂物、功能性但非核心行为物品
    'sundries_zone': {
        'keys', 'wallet', 'glasses', 'watch',
        'card', 'backpack', 'handbag', 'cap',
        'umbrella', 'comb', 'mirror',
        'charger', 'cable', 'power bank',
        'socket'
    },
    # 垃圾、废弃物、明显恶化卫生状况的物品
    'unclean_zone': {
        'package', 'snack box', 'carton', 'box',
        'bottle cap', 'straw',
        'snack leftovers', 'food wrapper',
        'waste paperball', 'waste package',
        'tissue', 'towel', 'plastic bag',
        'broken items', 'debris'
    }
}

# 构建物品名称到功能区的映射
NAME_TO_PARTITION = {}
for zone_name, items in PARTITION_ZONES.items():
    for item in items:
        NAME_TO_PARTITION[item] = zone_name


# ============================================================
#                   区域类型配置
# ============================================================
# 使用 partition_zones 作为唯一划分依据

ZONE_TYPE_CONFIG = {
    ZoneType.STUDY: {
        'name': '学习区',
        'partition_zone': 'study_zone',      # 直接对应 partition_zones
        'min_ratio': 0.20,                   # 该区域物品占比阈值（降低以提高识别率）
        'color': (200, 150, 80),             # BGR: 浅蓝偏紫
        'icon': ''
    },
    ZoneType.WORK: {
        'name': '工作区',
        'partition_zone': 'work_zone',
        'min_ratio': 0.20,
        'color': (100, 200, 100),            # BGR: 绿色
        'icon': ''
    },
    ZoneType.LEISURE: {
        'name': '休闲区',
        'partition_zone': 'relax_zone',
        'min_ratio': 0.15,
        'color': (100, 180, 255),            # BGR: 橙色
        'icon': ''
    },
    ZoneType.DINING: {
        'name': '饮食区',
        'partition_zone': 'food_zone',
        'min_ratio': 0.20,
        'color': (100, 220, 255),            # BGR: 黄色
        'icon': ''
    },
    ZoneType.CLUTTER: {
        'name': '杂物区',
        'partition_zone': 'sundries_zone',
        'min_ratio': 0.15,
        'color': (180, 180, 180),            # BGR: 灰色
        'icon': ''
    },
    ZoneType.CLEANUP: {
        'name': '待清理区',
        'partition_zone': 'unclean_zone',
        'min_ratio': 0.15,
        'color': (80, 80, 220),              # BGR: 红色
        'icon': ''
    }
}


# ============================================================
#                   主聚类引擎
# ============================================================

class ZoneClusteringEngine:
    """桌面功能区域聚类引擎"""
    
    def __init__(self, config: Optional[ZoneClusteringConfig] = None):
        self.config = config or ZoneClusteringConfig()
        
    def cluster(self, 
                objects: List[Dict], 
                image_size: Tuple[int, int]) -> ZoneClusteringResult:
        """
        执行聚类
        
        Args:
            objects: 物品列表，每个物品包含 bbox_xyxy, class 等信息
            image_size: 图像尺寸 (width, height)
        
        Returns:
            ZoneClusteringResult: 聚类结果
        """
        W, H = image_size
        n_objects = len(objects)
        
        # 特殊情况处理
        if n_objects == 0:
            return ZoneClusteringResult(zones=[], params={'n_objects': 0})
        
        if n_objects == 1:
            return self._handle_single_object(objects, image_size)
        
        # Phase 1: 特征工程
        positions, semantic_features = self._extract_features(objects, image_size)
        
        # Phase 1.3: 构建联合距离矩阵
        distance_matrix = self._compute_combined_distance_matrix(
            objects, positions, semantic_features
        )
        
        # Phase 2: DBSCAN 主聚类
        eps = self._compute_eps(distance_matrix)
        min_samples = self._compute_min_samples(n_objects)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)
        
        # 收集聚类结果
        clusters, outlier_indices = self._collect_dbscan_results(objects, labels)
        
        # Phase 3: 离群点二次处理
        clusters = self._handle_outliers(
            objects, outlier_indices, clusters, distance_matrix, eps
        )
        
        # Phase 4: 区域类型推断
        zones = []
        for cluster_id, cluster_data in clusters.items():
            zone = self._create_zone(
                cluster_id, cluster_data, objects, image_size
            )
            zones.append(zone)
        
        # 按区域 ID 排序
        zones.sort(key=lambda z: z.id)
        
        return ZoneClusteringResult(
            zones=zones,
            outliers=[],
            params={
                'n_objects': n_objects,
                'eps': float(eps),
                'min_samples': min_samples,
                'position_weight': self.config.position_weight,
                'n_zones': len(zones)
            },
            debug_info={
                'distance_matrix_shape': distance_matrix.shape,
                'distance_range': (float(distance_matrix.min()), float(distance_matrix.max()))
            }
        )
    
    # ==================== Phase 1: 特征工程 ====================
    
    def _extract_features(self, 
                          objects: List[Dict], 
                          image_size: Tuple[int, int]) -> Tuple[np.ndarray, List[Dict]]:
        """提取位置和语义特征"""
        W, H = image_size
        
        positions = []
        semantic_features = []
        
        for obj in objects:
            # 位置特征：归一化中心点坐标
            bbox = obj['bbox_xyxy']
            cx = (bbox[0] + bbox[2]) / 2 / W
            cy = (bbox[1] + bbox[3]) / 2 / H
            positions.append([cx, cy])
            
            # 语义特征：类别信息
            # 注意：refined_json 中物品名称在 'class' 字段
            # analysis 处理后 'name' 是原始名称，'class' 变成 super_class
            item_name = obj.get('name') or obj.get('class', 'unknown')
            # 优先用 item_name 查询 super_class
            super_class = NAME_TO_SUPER.get(item_name, obj.get('class', None))
            top_category = SUPER_TO_TOP.get(super_class, None) if super_class else None
            
            semantic_features.append({
                'class': item_name,           # 使用物品原始名称
                'super_class': super_class,   # 用于语义距离计算
                'top_category': top_category  # 用于粗粒度语义距离
            })
        
        return np.array(positions), semantic_features
    
    def _compute_semantic_distance(self, 
                                   sem1: Dict, 
                                   sem2: Dict) -> float:
        """计算两个物品之间的语义距离"""
        # Case 1: 同一 class
        if sem1['class'] == sem2['class']:
            return self.config.same_class_distance
        
        # Case 2: 同一 super_class
        if sem1['super_class'] and sem2['super_class']:
            if sem1['super_class'] == sem2['super_class']:
                return self.config.same_superclass_distance
        
        # Case 3: 同一 top_category
        if sem1['top_category'] and sem2['top_category']:
            if sem1['top_category'] == sem2['top_category']:
                return self.config.same_topcategory_distance
        
        # Case 4: 不同类别
        base_distance = self.config.different_category_distance
        
        # 特殊规则：垃圾惩罚
        if sem1['super_class'] == 'garbageAndDirty' or \
           sem2['super_class'] == 'garbageAndDirty':
            if sem1['super_class'] != sem2['super_class']:
                base_distance += self.config.garbage_penalty
        
        # 特殊规则：电子设备聚合奖励
        if sem1['super_class'] in ['electronics', 'powerAndCable'] and \
           sem2['super_class'] in ['electronics', 'powerAndCable']:
            base_distance -= self.config.electronics_bonus
        
        return base_distance
    
    def _compute_combined_distance_matrix(self,
                                          objects: List[Dict],
                                          positions: np.ndarray,
                                          semantic_features: List[Dict]) -> np.ndarray:
        """计算联合距离矩阵"""
        n = len(objects)
        
        # 计算位置距离矩阵
        position_dist = squareform(pdist(positions, metric='euclidean'))
        
        # 归一化到 [0, 1]
        max_pos_dist = position_dist.max()
        if max_pos_dist > 0:
            position_dist_norm = position_dist / max_pos_dist
        else:
            position_dist_norm = position_dist
        
        # 计算语义距离矩阵
        semantic_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._compute_semantic_distance(
                    semantic_features[i], semantic_features[j]
                )
                semantic_dist[i, j] = dist
                semantic_dist[j, i] = dist
        
        # 归一化到 [0, 1]
        max_sem_dist = semantic_dist.max()
        if max_sem_dist > 0:
            semantic_dist_norm = semantic_dist / max_sem_dist
        else:
            semantic_dist_norm = semantic_dist
        
        # 融合
        alpha = self.config.position_weight
        combined_dist = alpha * position_dist_norm + (1 - alpha) * semantic_dist_norm
        
        return combined_dist
    
    # ==================== Phase 2: DBSCAN ====================
    
    def _compute_eps(self, distance_matrix: np.ndarray) -> float:
        """自适应计算 eps"""
        # 提取上三角（不含对角线）
        upper_tri_indices = np.triu_indices_from(distance_matrix, k=1)
        distances = distance_matrix[upper_tri_indices]
        
        # 计算分位数
        eps = np.percentile(distances, self.config.eps_percentile)
        
        # 限制在合理范围
        eps = np.clip(eps, 0.05, 0.5)
        
        return float(eps)
    
    def _compute_min_samples(self, n_objects: int) -> int:
        """自适应计算 min_samples"""
        if n_objects < 10:
            return 2
        elif n_objects < 20:
            return 2
        else:
            return max(2, self.config.min_samples)
    
    def _collect_dbscan_results(self, 
                                objects: List[Dict], 
                                labels: np.ndarray) -> Tuple[Dict, List[int]]:
        """收集 DBSCAN 结果"""
        clusters = {}
        outlier_indices = []
        
        for i, label in enumerate(labels):
            if label == -1:
                outlier_indices.append(i)
            else:
                if label not in clusters:
                    clusters[label] = {
                        'object_indices': [],
                        'is_from_dbscan': True
                    }
                clusters[label]['object_indices'].append(i)
        
        return clusters, outlier_indices
    
    # ==================== Phase 3: 离群点处理 ====================
    
    def _handle_outliers(self,
                        objects: List[Dict],
                        outlier_indices: List[int],
                        clusters: Dict,
                        distance_matrix: np.ndarray,
                        eps: float) -> Dict:
        """处理离群点"""
        if not outlier_indices:
            return clusters
        
        merge_threshold = eps * self.config.merge_threshold_ratio
        
        # Phase 3.1: 尝试合并到现有聚类
        remaining_outliers = []
        for outlier_idx in outlier_indices:
            merged = self._try_merge_to_cluster(
                outlier_idx, objects, clusters, distance_matrix, merge_threshold
            )
            if not merged:
                remaining_outliers.append(outlier_idx)
        
        # Phase 3.2: K-Means 二次聚类
        if len(remaining_outliers) >= self.config.kmeans_min_points:
            clusters = self._kmeans_subclustering(
                remaining_outliers, objects, clusters, distance_matrix
            )
        else:
            # Phase 3.3: 标记为独立区域
            next_cluster_id = max(clusters.keys()) + 1 if clusters else 0
            for outlier_idx in remaining_outliers:
                clusters[next_cluster_id] = {
                    'object_indices': [outlier_idx],
                    'is_from_dbscan': False,
                    'source': 'isolated'
                }
                next_cluster_id += 1
        
        return clusters
    
    def _try_merge_to_cluster(self,
                              outlier_idx: int,
                              objects: List[Dict],
                              clusters: Dict,
                              distance_matrix: np.ndarray,
                              merge_threshold: float) -> bool:
        """尝试将离群点合并到最近的兼容聚类"""
        outlier_obj = objects[outlier_idx]
        # 优先使用 name 字段，如果没有则使用 class 字段
        outlier_name = outlier_obj.get('name') or outlier_obj.get('class', 'unknown')
        outlier_super = NAME_TO_SUPER.get(outlier_name, outlier_obj.get('class', None))
        outlier_top = SUPER_TO_TOP.get(outlier_super, None) if outlier_super else None
        
        best_cluster = None
        best_distance = float('inf')
        
        for cluster_id, cluster_data in clusters.items():
            # 计算到聚类的最小距离
            min_dist = min(distance_matrix[outlier_idx, i] 
                          for i in cluster_data['object_indices'])
            
            if min_dist >= merge_threshold:
                continue
            
            # 检查语义兼容性
            cluster_objects = [objects[i] for i in cluster_data['object_indices']]
            if self._is_semantically_compatible(
                outlier_name, outlier_super, outlier_top, cluster_objects
            ):
                if min_dist < best_distance:
                    best_distance = min_dist
                    best_cluster = cluster_id
        
        if best_cluster is not None:
            clusters[best_cluster]['object_indices'].append(outlier_idx)
            return True
        
        return False
    
    def _is_semantically_compatible(self,
                                   outlier_name: str,
                                   outlier_super: Optional[str],
                                   outlier_top: Optional[str],
                                   cluster_objects: List[Dict]) -> bool:
        """检查语义兼容性"""
        # 统计聚类的主导类别
        super_counts = Counter()
        for obj in cluster_objects:
            # 优先使用 name 字段，如果没有则使用 class 字段
            obj_name = obj.get('name') or obj.get('class', 'unknown')
            obj_super = NAME_TO_SUPER.get(obj_name, obj.get('class', None))
            if obj_super:
                super_counts[obj_super] += 1
        
        if not super_counts:
            return True
        
        dominant_super = super_counts.most_common(1)[0][0]
        dominant_top = SUPER_TO_TOP.get(dominant_super, None)
        
        # 同 super_class 或同 top_category 即兼容
        if outlier_super == dominant_super:
            return True
        if outlier_top and dominant_top and outlier_top == dominant_top:
            return True
        
        return False
    
    def _kmeans_subclustering(self,
                             outlier_indices: List[int],
                             objects: List[Dict],
                             clusters: Dict,
                             distance_matrix: np.ndarray) -> Dict:
        """对剩余离群点进行 K-Means 二次聚类"""
        n_outliers = len(outlier_indices)
        
        # 提取离群点的位置特征
        outlier_objects = [objects[i] for i in outlier_indices]
        positions = []
        for obj in outlier_objects:
            bbox = obj['bbox_xyxy']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            positions.append([cx, cy])
        
        positions = np.array(positions)
        
        # 确定 K 值
        K = max(1, n_outliers // 3)
        K = min(K, self.config.kmeans_max_clusters)
        
        # 执行 K-Means
        if K == 1:
            sub_labels = np.zeros(n_outliers, dtype=int)
        else:
            kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
            sub_labels = kmeans.fit_predict(positions)
        
        # 创建新聚类
        next_cluster_id = max(clusters.keys()) + 1 if clusters else 0
        
        for k in range(K):
            mask = sub_labels == k
            indices = [outlier_indices[i] for i, m in enumerate(mask) if m]
            if indices:
                clusters[next_cluster_id] = {
                    'object_indices': indices,
                    'is_from_dbscan': False,
                    'source': 'kmeans'
                }
                next_cluster_id += 1
        
        return clusters
    
    # ==================== Phase 4 & 5: 区域类型推断 & 边界生成 ====================
    
    def _create_zone(self,
                    cluster_id: int,
                    cluster_data: Dict,
                    objects: List[Dict],
                    image_size: Tuple[int, int]) -> ClusterZone:
        """创建区域对象"""
        indices = cluster_data['object_indices']
        cluster_objects = [objects[i] for i in indices]
        
        # 计算中心
        centers = []
        for obj in cluster_objects:
            bbox = obj['bbox_xyxy']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append([cx, cy])
        center = tuple(np.mean(centers, axis=0))
        
        # 统计类别分布
        category_distribution = self._compute_category_distribution(cluster_objects)
        
        # 推断区域类型
        zone_type, zone_config = self._infer_zone_type(
            cluster_objects, category_distribution
        )
        
        # 生成边界
        boundary = self._compute_zone_boundary(cluster_objects, image_size)
        
        # 判断是否为单物品区域
        is_singleton = len(cluster_objects) == 1
        
        # 确定来源
        source = cluster_data.get('source', 'dbscan' if cluster_data.get('is_from_dbscan', False) else 'kmeans')
        
        return ClusterZone(
            id=cluster_id,
            zone_type=zone_config['name'],
            objects=cluster_objects,
            object_indices=indices,
            center=center,
            boundary=boundary,
            color=zone_config['color'],
            icon=zone_config['icon'],
            note="",
            category_distribution=category_distribution,
            is_singleton=is_singleton,
            source=source
        )
    
    def _compute_category_distribution(self, 
                                      cluster_objects: List[Dict]) -> Dict[str, float]:
        """统计功能区分布（基于 partition_zones）"""
        partition_counts = Counter()
        for obj in cluster_objects:
            # 优先使用 name 字段，如果没有则使用 class 字段
            item_name = obj.get('name') or obj.get('class', 'unknown')
            partition_zone = NAME_TO_PARTITION.get(item_name, 'unknown')
            partition_counts[partition_zone] += 1
        
        total = len(cluster_objects)
        return {zone: count / total for zone, count in partition_counts.items()}
    
    def _infer_zone_type(self,
                        cluster_objects: List[Dict],
                        partition_ratios: Dict[str, float]) -> Tuple[ZoneType, Dict]:
        """
        推断区域类型（基于 partition_zones）
        
        Args:
            cluster_objects: 聚类中的物品列表
            partition_ratios: 功能区分布比例 {partition_zone: ratio}
        """
        # 单物品特殊处理
        if len(cluster_objects) == 1:
            return self._infer_singleton_type(cluster_objects[0])
        
        # partition_zone 到 ZoneType 的映射
        partition_to_zone = {
            'study_zone': ZoneType.STUDY,
            'work_zone': ZoneType.WORK,
            'relax_zone': ZoneType.LEISURE,
            'food_zone': ZoneType.DINING,
            'sundries_zone': ZoneType.CLUTTER,
            'unclean_zone': ZoneType.CLEANUP
        }
        
        # 找出占比最高的有效分区（排除 'unknown'）
        best_partition = None
        best_ratio = 0
        for partition_zone, ratio in partition_ratios.items():
            if partition_zone != 'unknown' and ratio > best_ratio:
                best_ratio = ratio
                best_partition = partition_zone
        
        # 如果没有任何有效分区，默认为杂物区
        if not best_partition:
            return ZoneType.CLUTTER, ZONE_TYPE_CONFIG[ZoneType.CLUTTER]
        
        # 获取对应的 ZoneType，如果找不到则默认为杂物区
        zone_type = partition_to_zone.get(best_partition, ZoneType.CLUTTER)
        config = ZONE_TYPE_CONFIG[zone_type]
        
        return zone_type, config
    
    def _infer_singleton_type(self, obj: Dict) -> Tuple[ZoneType, Dict]:
        """推断单物品区域类型（基于 partition_zones）"""
        # 优先使用 name 字段，如果没有则使用 class 字段
        item_name = obj.get('name') or obj.get('class', 'unknown')
        partition_zone = NAME_TO_PARTITION.get(item_name, None)
        
        # 根据 partition_zone 映射到 ZoneType
        zone_mapping = {
            'study_zone': ZoneType.STUDY,
            'work_zone': ZoneType.WORK,
            'relax_zone': ZoneType.LEISURE,
            'food_zone': ZoneType.DINING,
            'sundries_zone': ZoneType.CLUTTER,
            'unclean_zone': ZoneType.CLEANUP
        }
        
        zone_type = zone_mapping.get(partition_zone, ZoneType.CLUTTER)
        return zone_type, ZONE_TYPE_CONFIG[zone_type]
    
    def _compute_zone_boundary(self,
                              cluster_objects: List[Dict],
                              image_size: Tuple[int, int]) -> np.ndarray:
        """
        计算区域边界（使用 Alpha Shape）
        
        策略：
        1. 优先使用物体的 convex_hull（SAM3分割结果）
        2. 如果没有 convex_hull，使用 bbox 的四个角点
        3. 用 Alpha Shape 生成既不完全贴合也不太圆的边界
        4. 最后向外扩展确保包裹所有物体
        """
        # 提取所有物品的边界点
        all_points = []
        bbox_corners = []  # 保存所有 bbox 角点，用于验证边界是否包裹
        
        for obj in cluster_objects:
            x1, y1, x2, y2 = obj['bbox_xyxy']
            
            # 保存 bbox 四角用于后续验证
            bbox_corners.extend([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ])
            
            # 优先使用 SAM3 的 convex_hull
            sam3_data = obj.get('sam3') or {}
            convex_hull = sam3_data.get('convex_hull', None) if sam3_data else None
            
            if convex_hull and len(convex_hull) >= 3:
                # 使用 convex_hull 的点（采样以控制点数）
                hull_points = np.array(convex_hull)
                # 如果点太多，均匀采样
                if len(hull_points) > 12:
                    indices = np.linspace(0, len(hull_points) - 1, 12, dtype=int)
                    hull_points = hull_points[indices]
                all_points.extend(hull_points.tolist())
            else:
                # 没有 convex_hull，使用 bbox 四角 + 中心
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                all_points.extend([
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2],
                    [cx, cy]
                ])
        
        points = np.array(all_points)
        unique_points = np.unique(points, axis=0)
        bbox_corners = np.array(bbox_corners)
        
        n_points = len(unique_points)
        
        if n_points >= 4:
            # 正常情况：使用 Alpha Shape
            alpha = self._compute_adaptive_alpha(unique_points)
            boundary = self._compute_alpha_shape(unique_points, alpha)
            
            if boundary is not None and len(boundary) >= 3:
                # 扩展边界确保包裹所有物体
                boundary = self._expand_boundary_to_cover(boundary, bbox_corners, image_size)
                return boundary
            else:
                # Alpha Shape 失败，降级到凸包
                hull_boundary = self._compute_convex_hull(unique_points)
                return self._expand_boundary_to_cover(hull_boundary, bbox_corners, image_size)
        
        elif n_points == 3:
            return self._expand_boundary_to_cover(unique_points, bbox_corners, image_size)
        
        elif n_points == 2:
            return self._generate_capsule_shape(unique_points[0], unique_points[1], image_size)
        
        else:  # n_points == 1
            return self._generate_circle_shape(unique_points[0], image_size)
    
    def _expand_boundary_to_cover(self, 
                                   boundary: np.ndarray, 
                                   bbox_corners: np.ndarray,
                                   image_size: Tuple[int, int]) -> np.ndarray:
        """
        扩展边界确保包裹所有 bbox 角点，并添加额外 padding
        """
        if len(boundary) < 3:
            return boundary
        
        boundary = np.array(boundary)
        W, H = image_size
        diagonal = np.sqrt(W**2 + H**2)
        
        # 基础 padding（对角线的 2%）
        base_padding = diagonal * 0.02
        
        # 计算边界质心
        centroid = np.mean(boundary, axis=0)
        
        # 检查每个 bbox 角点是否在边界内，计算需要扩展的比例
        max_expansion_ratio = 1.0
        
        for corner in bbox_corners:
            # 计算角点到质心的方向和距离
            to_corner = corner - centroid
            corner_dist = np.linalg.norm(to_corner)
            
            if corner_dist < 1e-6:
                continue
            
            corner_dir = to_corner / corner_dist
            
            # 找到边界在该方向上的最远点
            max_boundary_dist = 0
            for bp in boundary:
                to_bp = bp - centroid
                bp_dist = np.linalg.norm(to_bp)
                if bp_dist > 1e-6:
                    # 计算投影
                    proj = np.dot(to_bp, corner_dir)
                    if proj > max_boundary_dist:
                        max_boundary_dist = proj
            
            # 如果角点在该方向上超出边界，需要扩展
            if corner_dist > max_boundary_dist:
                ratio = (corner_dist + base_padding) / max(max_boundary_dist, 1e-6)
                max_expansion_ratio = max(max_expansion_ratio, ratio)
        
        # 应用扩展
        expanded = []
        for point in boundary:
            direction = point - centroid
            distance = np.linalg.norm(direction)
            if distance > 0:
                unit_dir = direction / distance
                # 扩展距离 = 原距离 * 扩展比例 + 额外 padding
                new_dist = distance * max_expansion_ratio + base_padding
                expanded_point = centroid + unit_dir * new_dist
            else:
                expanded_point = point
            expanded.append(expanded_point)
        
        expanded = np.array(expanded)
        
        # 平滑处理
        if self.config.boundary_smooth_sigma > 0 and len(expanded) > 5:
            n_pad = 3
            expanded_closed = np.vstack([
                expanded[-n_pad:],
                expanded,
                expanded[:n_pad]
            ])
            
            smoothed_x = gaussian_filter1d(expanded_closed[:, 0], 
                                          sigma=self.config.boundary_smooth_sigma)
            smoothed_y = gaussian_filter1d(expanded_closed[:, 1], 
                                          sigma=self.config.boundary_smooth_sigma)
            
            expanded = np.column_stack([smoothed_x, smoothed_y])[n_pad:-n_pad]
        
        return expanded
    
    def _compute_adaptive_alpha(self, points: np.ndarray) -> float:
        """自适应计算 alpha 值"""
        distances = pdist(points)
        median_dist = np.median(distances)
        
        if median_dist < 1e-6:
            return 0.01
        
        alpha = 1.0 / (median_dist * self.config.alpha_factor)
        alpha = np.clip(alpha, 0.001, 0.1)
        
        return float(alpha)
    
    def _compute_alpha_shape(self, points: np.ndarray, alpha: float) -> Optional[np.ndarray]:
        """计算 Alpha Shape"""
        try:
            # Delaunay 三角剖分
            tri = Delaunay(points)
            
            # 筛选满足 alpha 条件的边
            edge_counts = defaultdict(int)
            
            for simplex in tri.simplices:
                pts = points[simplex]
                radius = self._circumradius(pts[0], pts[1], pts[2])
                
                if radius < 1.0 / alpha:
                    for i in range(3):
                        edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                        edge_counts[edge] += 1
            
            # 边界边 = 只被一个三角形使用的边
            boundary_edges = [e for e, count in edge_counts.items() if count == 1]
            
            if not boundary_edges:
                return None
            
            # 将边连接成有序多边形
            polygon = self._edges_to_polygon(boundary_edges, points)
            
            return np.array(polygon) if polygon else None
            
        except Exception as e:
            print(f"Alpha Shape 计算失败: {e}")
            return None
    
    def _circumradius(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """计算三角形外接圆半径"""
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)
        
        if area_sq <= 0:
            return np.inf
        
        area = np.sqrt(area_sq)
        
        if area < 1e-10:
            return np.inf
        
        return (a * b * c) / (4 * area)
    
    def _edges_to_polygon(self, edges: List[Tuple[int, int]], points: np.ndarray) -> List:
        """将边列表转换为有序多边形"""
        if not edges:
            return []
        
        # 构建邻接表
        adj = defaultdict(list)
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        # 检查是否为简单多边形（每个顶点恰好有 2 个邻居）
        for node, neighbors in adj.items():
            if len(neighbors) != 2:
                # 非简单多边形，使用凸包
                indices = list(set([e[0] for e in edges] + [e[1] for e in edges]))
                return self._compute_convex_hull(points[indices]).tolist()
        
        # 从任意点开始遍历
        polygon = []
        start = edges[0][0]
        current = start
        prev = None
        visited = set()
        
        max_iterations = len(edges) + 10
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            if current in visited and current == start and len(polygon) > 2:
                break
            
            polygon.append(points[current].tolist())
            visited.add(current)
            
            neighbors = adj[current]
            next_point = None
            for n in neighbors:
                if n != prev:
                    next_point = n
                    break
            
            if next_point is None:
                break
            
            prev = current
            current = next_point
        
        return polygon
    
    def _compute_convex_hull(self, points: np.ndarray) -> np.ndarray:
        """计算凸包"""
        if len(points) < 3:
            return points
        
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            return points[hull.vertices]
        except Exception:
            return points
    
    def _expand_and_smooth_boundary(self, boundary: np.ndarray, image_size: Tuple[int, int] = None) -> np.ndarray:
        """扩展并平滑边界"""
        if len(boundary) < 3:
            return boundary
        
        boundary = np.array(boundary)
        # 根据图片对角线计算实际padding
        if image_size:
            W, H = image_size
            diagonal = np.sqrt(W**2 + H**2)
            padding = diagonal * self.config.boundary_padding_ratio
        else:
            padding = 25  # 降级默认值
        
        # 向外扩展
        centroid = np.mean(boundary, axis=0)
        expanded = []
        for point in boundary:
            direction = point - centroid
            distance = np.linalg.norm(direction)
            if distance > 0:
                unit_dir = direction / distance
                expanded_point = point + unit_dir * padding
            else:
                expanded_point = point
            expanded.append(expanded_point)
        
        expanded = np.array(expanded)
        
        # 平滑处理
        if self.config.boundary_smooth_sigma > 0 and len(expanded) > 5:
            n_pad = 3
            expanded_closed = np.vstack([
                expanded[-n_pad:],
                expanded,
                expanded[:n_pad]
            ])
            
            smoothed_x = gaussian_filter1d(expanded_closed[:, 0], 
                                          sigma=self.config.boundary_smooth_sigma)
            smoothed_y = gaussian_filter1d(expanded_closed[:, 1], 
                                          sigma=self.config.boundary_smooth_sigma)
            
            expanded = np.column_stack([smoothed_x, smoothed_y])[n_pad:-n_pad]
        
        return expanded
    
    def _generate_capsule_shape(self, p1: np.ndarray, p2: np.ndarray, image_size: Tuple[int, int] = None) -> np.ndarray:
        """生成胶囊形状（两点情况）"""
        if image_size:
            W, H = image_size
            diagonal = np.sqrt(W**2 + H**2)
            padding = diagonal * self.config.boundary_padding_ratio
        else:
            padding = 25
        
        direction = p2 - p1
        length = np.linalg.norm(direction)
        unit_dir = direction / length if length > 0 else np.array([1, 0])
        
        points = []
        # 第一端的半圆
        for angle in np.linspace(np.pi / 2, 3 * np.pi / 2, 8):
            offset = np.array([np.cos(angle), np.sin(angle)]) * padding
            rotated = self._rotate_point(offset, np.arctan2(unit_dir[1], unit_dir[0]))
            points.append(p1 + rotated)
        # 第二端的半圆
        for angle in np.linspace(-np.pi / 2, np.pi / 2, 8):
            offset = np.array([np.cos(angle), np.sin(angle)]) * padding
            rotated = self._rotate_point(offset, np.arctan2(unit_dir[1], unit_dir[0]))
            points.append(p2 + rotated)
        
        return np.array(points)
    
    def _generate_circle_shape(self, center: np.ndarray, image_size: Tuple[int, int] = None) -> np.ndarray:
        """生成圆形（单点情况）"""
        if image_size:
            W, H = image_size
            diagonal = np.sqrt(W**2 + H**2)
            radius = diagonal * self.config.boundary_padding_ratio
        else:
            radius = 25
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        points = []
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    def _rotate_point(self, point: np.ndarray, angle: float) -> np.ndarray:
        """旋转点"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return rotation_matrix @ point
    
    def _handle_single_object(self, 
                             objects: List[Dict], 
                             image_size: Tuple[int, int]) -> ZoneClusteringResult:
        """处理单个物品的情况"""
        obj = objects[0]
        bbox = obj['bbox_xyxy']
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        zone_type, zone_config = self._infer_singleton_type(obj)
        boundary = self._compute_zone_boundary([obj], image_size)
        
        # 优先使用 name 字段，如果没有则使用 class 字段，用于 partition_zone 分类
        item_name = obj.get('name') or obj.get('class', 'unknown')
        partition_zone = NAME_TO_PARTITION.get(item_name, 'unknown')
        
        zone = ClusterZone(
            id=0,
            zone_type=zone_config['name'],
            objects=[obj],
            object_indices=[0],
            center=center,
            boundary=boundary,
            color=zone_config['color'],
            icon=zone_config['icon'],
            note="",
            category_distribution={partition_zone: 1.0},
            is_singleton=True,
            source='isolated'
        )
        
        return ZoneClusteringResult(
            zones=[zone],
            params={'n_objects': 1}
        )


# ============================================================
#                   可视化渲染器
# ============================================================

class ZoneVisualizer:
    """区域可视化渲染器"""
    
    def __init__(self, font_path: Optional[str] = None, image_size: Optional[Tuple[int, int]] = None):
        self.font_path = font_path
        self.image_size = image_size
        
        # 根据图片尺寸计算自适应参数
        if image_size:
            W, H = image_size
            diagonal = np.sqrt(W**2 + H**2)
            # 字体大小：对角线的2.5%
            self.font_size = int(diagonal * 0.025)
            # 圆圈半径：对角线的0.8%
            self.marker_radius = int(diagonal * 0.008)
            # 线条粗细：对角线的0.15%
            self.line_thickness = max(1, int(diagonal * 0.0015))
            # 标签Y偏移：对角线的1.5%
            self.label_offset_y = int(diagonal * 0.015)
        else:
            # 降级默认值
            self.font_size = 32
            self.marker_radius = 8
            self.line_thickness = 3
            self.label_offset_y = 30
    
    def render(self, image: np.ndarray, zones: List[ClusterZone]) -> np.ndarray:
        """渲染所有区域"""
        result = image.copy()
        
        # 如果初始化时没有提供image_size，根据图片计算
        if not self.image_size:
            H, W = image.shape[:2]
            self.image_size = (W, H)
            diagonal = np.sqrt(W**2 + H**2)
            self.font_size = int(diagonal * 0.025)
            self.marker_radius = int(diagonal * 0.008)
            self.line_thickness = max(1, int(diagonal * 0.0015))
            self.label_offset_y = int(diagonal * 0.015)
        
        # Step 1: 绘制所有区域填充
        for zone in zones:
            result = self._draw_zone_fill(result, zone)
        
        # Step 2: 绘制边界线
        for zone in zones:
            result = self._draw_zone_boundary(result, zone)
        
        # Step 3: 绘制标签
        for zone in zones:
            result = self._draw_zone_label(result, zone)
        
        # Step 4: 绘制物品标记
        for zone in zones:
            result = self._draw_object_markers(result, zone)
        
        return result
    
    def _draw_zone_fill(self, image: np.ndarray, zone: ClusterZone) -> np.ndarray:
        """绘制半透明区域填充"""
        overlay = image.copy()
        pts = zone.boundary.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], zone.color)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        return image
    
    def _draw_zone_boundary(self, image: np.ndarray, zone: ClusterZone) -> np.ndarray:
        """绘制区域边界线"""
        pts = zone.boundary.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=zone.color, 
                     thickness=self.line_thickness, lineType=cv2.LINE_AA)
        return image
    
    def _draw_zone_label(self, image: np.ndarray, zone: ClusterZone) -> np.ndarray:
        """绘制区域标签"""
        centroid = np.mean(zone.boundary, axis=0)
        label_pos = (int(centroid[0]), int(centroid[1]) - self.label_offset_y)
        
        label_text = f"{zone.icon} {zone.zone_type}"
        if zone.note:
            label_text += f" {zone.note}"
        
        if self.font_path and Path(self.font_path).exists():
            image = self._draw_chinese_text(image, label_text, label_pos, zone.color)
        else:
            # 降级：使用 OpenCV（自适应字体大小）
            font_scale = self.font_size / 30.0  # 转换为OpenCV的scale
            thickness = max(1, self.line_thickness)
            cv2.putText(image, zone.zone_type, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, zone.color, thickness)
        
        return image
    
    def _draw_chinese_text(self, 
                          image: np.ndarray, 
                          text: str, 
                          position: Tuple[int, int],
                          color: Tuple[int, int, int]) -> np.ndarray:
        """使用 PIL 绘制中文"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            font = ImageFont.truetype(self.font_path, self.font_size)
            
            x, y = position
            # 描边（根据字体大小自适应）
            stroke_width = max(1, self.font_size // 16)
            for dx, dy in [(-stroke_width, -stroke_width), (-stroke_width, stroke_width), 
                          (stroke_width, -stroke_width), (stroke_width, stroke_width),
                          (-stroke_width, 0), (stroke_width, 0), (0, -stroke_width), (0, stroke_width)]:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
            # 正文
            draw.text((x, y), text, font=font, fill=color[::-1])
            
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"中文绘制失败: {e}")
            return image
    
    def _draw_object_markers(self, image: np.ndarray, zone: ClusterZone) -> np.ndarray:
        """绘制物品标记点"""
        for obj in zone.objects:
            bbox = obj['bbox_xyxy']
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            # 自适应圆圈大小
            inner_radius = self.marker_radius
            outer_radius = int(self.marker_radius * 1.2)
            outer_thickness = max(1, self.marker_radius // 5)
            cv2.circle(image, (cx, cy), inner_radius, zone.color, -1)
            cv2.circle(image, (cx, cy), outer_radius, (255, 255, 255), outer_thickness)
        return image


# ============================================================
#                   便捷函数
# ============================================================

def cluster_zones(objects: List[Dict], 
                 image_size: Tuple[int, int],
                 config: Optional[ZoneClusteringConfig] = None) -> ZoneClusteringResult:
    """
    便捷函数：执行区域聚类
    
    Args:
        objects: 物品列表
        image_size: 图像尺寸 (width, height)
        config: 聚类配置
    
    Returns:
        ZoneClusteringResult: 聚类结果
    """
    engine = ZoneClusteringEngine(config)
    return engine.cluster(objects, image_size)


def visualize_zones(image: np.ndarray,
                   zones: List[ClusterZone],
                   font_path: Optional[str] = None) -> np.ndarray:
    """
    便捷函数：可视化区域
    
    Args:
        image: 输入图像
        zones: 区域列表
        font_path: 中文字体路径
    
    Returns:
        可视化结果图像
    """
    H, W = image.shape[:2]
    visualizer = ZoneVisualizer(font_path, image_size=(W, H))
    return visualizer.render(image, zones)
