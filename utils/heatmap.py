# utils/heatmap.py


import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.ndimage import gaussian_filter


class MessinessHeatmapGenerator:
    """桌面混乱热力图生成器"""
    
    # ==================== 扩散参数配置 ====================
    # 调整这些参数可以控制热力图的扩散范围和平滑程度
    
    # density 层：物品密度热力
    DENSITY_SPREAD_RADIUS_SAM3 = 2.4    # SAM3 面积计算的扩散半径乘数
    DENSITY_SPREAD_RADIUS_BBOX = 1.6    # bbox 对角线计算的扩散半径乘数
    DENSITY_SIGMA_RATIO = 1.0           # sigma 相对于 spread_radius 的比例
    DENSITY_PAD_RATIO = 2.0             # 计算区域 pad 相对于 spread_radius 的比例
    
    # sam3_geometry 层：SAM3 几何特征热力
    SAM3_GEO_SPREAD_RADIUS = 2.4        # 扩散半径乘数
    SAM3_GEO_SIGMA_RATIO = 0.9          # sigma 比例
    SAM3_GEO_PAD_RATIO = 2.5            # pad 比例
    
    # overlap 层：重叠区域热力
    OVERLAP_SPREAD_RADIUS = 2.0         # 扩散半径乘数
    OVERLAP_SIGMA_RATIO = 0.9           # sigma 比例
    OVERLAP_PAD_RATIO = 2.5             # pad 比例
    
    # semantic 层：语义（脏乱物品）热力
    SEMANTIC_SPREAD_RADIUS = 2.8        # 扩散半径乘数（脏乱物品扩散更广）
    SEMANTIC_SIGMA_RATIO = 1.0          # sigma 比例
    SEMANTIC_PAD_RATIO = 2.5            # pad 比例
    
    # ==================== 物品权重配置 ====================
    
    # 物品类别权重：不同类型的物品对"混乱感"的贡献不同
    # 与 analysis.py 中的 super_classes 保持同步
    CATEGORY_WEIGHTS = {
        'garbageAndDirty': 2.0,       # 垃圾与脏乱物品最影响整洁度
        'powerAndCable': 1.3,         # 电源线缆容易显乱
        'learning': 1.2,              # 学习物品分散时显乱
        'foodAndDrink': 1.0,          # 饮食相关
        'smallStuff': 0.9,            # 小物件
        'entertainment': 0.8,         # 娱乐物品
        'medicine': 0.8,              # 医药物品
        'electronics': 0.6,           # 电子设备通常较规整
        None: 1.0                     # 默认权重
    }
    
    # 特殊物品额外权重
    SPECIAL_ITEM_WEIGHTS = {
        'trash': 2.5,
        'packaging': 2.0,
        'disorderedbook': 1.8,
        'scatteredpaper': 1.8,
        'scatteredpen': 1.6,
        'disorderedcable': 1.8,
        'powersupplycable': 1.4,
    }
    
    def __init__(self, 
                 gaussian_sigma_ratio: float = 0.03,  # 减小模糊程度，更精细
                 density_weight: float = 0.30,
                 variance_weight: float = 0.15,
                 convexity_weight: float = 0.15,
                 overlap_weight: float = 0.30,
                 semantic_weight: float = 0.10):
        """
        初始化热力图生成器
        
        Args:
            gaussian_sigma_ratio: 高斯模糊 sigma 相对于图像对角线的比例
            density_weight: 密度因子权重
            variance_weight: 轮廓方差因子权重
            convexity_weight: 凸包率因子权重
            overlap_weight: 重叠因子权重
            semantic_weight: 语义因子权重
        """
        self.gaussian_sigma_ratio = gaussian_sigma_ratio
        self.weights = {
            'density': density_weight,
            'variance': variance_weight,
            'convexity': convexity_weight,
            'overlap': overlap_weight,
            'semantic': semantic_weight
        }
        
    def load_data(self, 
                  refined_json_path: str, 
                  analysis_json_path: str) -> Tuple[Dict, Dict]:
        """加载 refined_json 和 analysis_output 数据"""
        with open(refined_json_path, 'r', encoding='utf-8') as f:
            refined_data = json.load(f)
        with open(analysis_json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        return refined_data, analysis_data
    
    def _get_item_weight(self, obj: Dict) -> float:
        """计算单个物品的混乱权重"""
        name = obj.get('name', obj.get('class', ''))
        category = obj.get('class', None)
        
        # 基础权重来自类别
        base_weight = self.CATEGORY_WEIGHTS.get(category, 1.0)
        
        # 特殊物品额外权重
        special_weight = self.SPECIAL_ITEM_WEIGHTS.get(name, 1.0)
        
        # 属性加成
        attr_multiplier = 1.0
        if obj.get('is_messy_related', False):
            attr_multiplier *= 1.3
        if obj.get('is_dirty_related', False):
            attr_multiplier *= 1.5
            
        return base_weight * special_weight * attr_multiplier
    
    def _compute_density_layer(self, 
                                objects: List[Dict], 
                                refined_objects: List[Dict],
                                H: int, W: int) -> np.ndarray:
        """
        计算密度层：以 bbox 中心为热源，圆形向外扩散
        同时考虑 YOLO bbox 尺寸和 SAM3 几何信息
        """
        density = np.zeros((H, W), dtype=np.float32)
        
        for obj in objects:
            bbox = obj.get('bbox', obj.get('bbox_xyxy', []))
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            weight = self._get_item_weight(obj)
            obj_w, obj_h = x2 - x1, y2 - y1
            area = obj_w * obj_h
            area_norm = np.sqrt(area) / np.sqrt(H * W)
            heat_value = weight * (1 + area_norm)
            
            # 计算中心点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 扩散半径：基于物体尺寸
            # SAM3 对象可能有更精细的尺寸信息
            sam3_geo = obj.get('sam3_geometry', {})
            if sam3_geo:
                sam3_area = sam3_geo.get('mask_area', area)
                # 用 SAM3 面积估算等效半径
                equiv_radius = np.sqrt(sam3_area / np.pi)
                spread_radius = equiv_radius * self.DENSITY_SPREAD_RADIUS_SAM3
            else:
                # 用 bbox 对角线的一半作为基础半径
                spread_radius = np.sqrt(obj_w**2 + obj_h**2) / 2 * self.DENSITY_SPREAD_RADIUS_BBOX
            
            # 计算区域范围
            pad = int(spread_radius * self.DENSITY_PAD_RATIO)
            calc_x1 = max(0, int(center_x) - pad)
            calc_y1 = max(0, int(center_y) - pad)
            calc_x2 = min(W, int(center_x) + pad)
            calc_y2 = min(H, int(center_y) + pad)
            
            if calc_x2 <= calc_x1 or calc_y2 <= calc_y1:
                continue
            
            # 生成坐标网格，计算到中心的距离
            y_coords, x_coords = np.ogrid[calc_y1:calc_y2, calc_x1:calc_x2]
            dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
            
            # 高斯圆形扩散
            sigma = spread_radius * self.DENSITY_SIGMA_RATIO
            gaussian_heat = np.exp(-dist_sq / (2 * sigma**2))
            
            # 叠加到密度层
            density[calc_y1:calc_y2, calc_x1:calc_x2] += gaussian_heat * heat_value
            
        return density
            
        return density
    
    def _compute_sam3_geometry_layer(self,
                                      refined_objects: List[Dict],
                                      H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算 SAM3 几何特征层：以中心点为热源，圆形扩散
        - variance_layer: normalized_contour_variance 贡献
        - convexity_layer: (1 - convexity_ratio) 贡献
        """
        variance_layer = np.zeros((H, W), dtype=np.float32)
        convexity_layer = np.zeros((H, W), dtype=np.float32)
        
        for obj in refined_objects:
            sam3 = obj.get('sam3', None)
            if sam3 is None:
                continue
            
            # 获取几何特征
            contour_var = sam3.get('normalized_contour_variance', 0)
            convexity = sam3.get('convexity_ratio', 1.0)
            bbox = obj.get('bbox_xyxy', [])
            
            # 方差贡献：归一化到 [0, 1]
            var_score = min(contour_var / 0.3, 1.0)
            # 凸包率贡献：1 - ratio，越低越乱
            non_convex_score = 1.0 - convexity
            
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            obj_w, obj_h = x2 - x1, y2 - y1
            
            # 中心点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 用 SAM3 面积估算扩散半径
            sam3_area = sam3.get('mask_area', obj_w * obj_h)
            spread_radius = np.sqrt(sam3_area / np.pi) * self.SAM3_GEO_SPREAD_RADIUS
            
            # 计算区域
            pad = int(spread_radius * self.SAM3_GEO_PAD_RATIO)
            calc_x1, calc_y1 = max(0, int(center_x) - pad), max(0, int(center_y) - pad)
            calc_x2, calc_y2 = min(W, int(center_x) + pad), min(H, int(center_y) + pad)
            
            if calc_x2 <= calc_x1 or calc_y2 <= calc_y1:
                continue
            
            # 圆形扩散
            y_coords, x_coords = np.ogrid[calc_y1:calc_y2, calc_x1:calc_x2]
            dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
            sigma = spread_radius * self.SAM3_GEO_SIGMA_RATIO
            gaussian = np.exp(-dist_sq / (2 * sigma**2))
            
            variance_layer[calc_y1:calc_y2, calc_x1:calc_x2] += gaussian * var_score
            convexity_layer[calc_y1:calc_y2, calc_x1:calc_x2] += gaussian * non_convex_score
                
        return variance_layer, convexity_layer
    
    def _compute_overlap_layer(self,
                                objects: List[Dict],
                                H: int, W: int) -> np.ndarray:
        """
        计算重叠层：以中心点为热源，圆形扩散
        多个热源重叠的区域热度自然叠加
        """
        overlap_layer = np.zeros((H, W), dtype=np.float32)
        
        for obj in objects:
            bbox = obj.get('bbox', obj.get('bbox_xyxy', []))
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            obj_w, obj_h = x2 - x1, y2 - y1
            
            # 中心点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 扩散半径：用 SAM3 面积或 bbox 面积
            sam3_geo = obj.get('sam3_geometry', {})
            if sam3_geo and sam3_geo.get('mask_area'):
                area = sam3_geo['mask_area']
            else:
                area = obj_w * obj_h
            spread_radius = np.sqrt(area / np.pi) * self.OVERLAP_SPREAD_RADIUS
            
            # 计算区域
            pad = int(spread_radius * self.OVERLAP_PAD_RATIO)
            calc_x1, calc_y1 = max(0, int(center_x) - pad), max(0, int(center_y) - pad)
            calc_x2, calc_y2 = min(W, int(center_x) + pad), min(H, int(center_y) + pad)
            
            if calc_x2 <= calc_x1 or calc_y2 <= calc_y1:
                continue
            
            # 圆形扩散
            y_coords, x_coords = np.ogrid[calc_y1:calc_y2, calc_x1:calc_x2]
            dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
            sigma = spread_radius * self.OVERLAP_SIGMA_RATIO
            gaussian = np.exp(-dist_sq / (2 * sigma**2))
            
            overlap_layer[calc_y1:calc_y2, calc_x1:calc_x2] += gaussian
        
        # 只保留重叠部分（值 > 1 的区域）
        overlap_layer = np.maximum(overlap_layer - 1, 0)
        
        return overlap_layer
    
    def _compute_semantic_layer(self,
                                 objects: List[Dict],
                                 H: int, W: int) -> np.ndarray:
        """
        计算语义层：以中心点为热源，圆形扩散
        脏乱物品扩散范围更大
        """
        semantic_layer = np.zeros((H, W), dtype=np.float32)
        
        for obj in objects:
            bbox = obj.get('bbox', obj.get('bbox_xyxy', []))
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
            
            score = 0.0
            if obj.get('is_messy_related', False):
                score += 0.4
            if obj.get('is_dirty_related', False):
                score += 0.6
            if obj.get('is_special_related', False):
                score += 0.2
                
            if score > 0:
                obj_w, obj_h = x2 - x1, y2 - y1
                
                # 中心点
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 脏乱物品扩散范围更大
                sam3_geo = obj.get('sam3_geometry', {})
                if sam3_geo and sam3_geo.get('mask_area'):
                    area = sam3_geo['mask_area']
                else:
                    area = obj_w * obj_h
                spread_radius = np.sqrt(area / np.pi) * self.SEMANTIC_SPREAD_RADIUS
                
                # 计算区域
                pad = int(spread_radius * self.SEMANTIC_PAD_RATIO)
                calc_x1, calc_y1 = max(0, int(center_x) - pad), max(0, int(center_y) - pad)
                calc_x2, calc_y2 = min(W, int(center_x) + pad), min(H, int(center_y) + pad)
                
                if calc_x2 <= calc_x1 or calc_y2 <= calc_y1:
                    continue
                
                # 圆形扩散
                y_coords, x_coords = np.ogrid[calc_y1:calc_y2, calc_x1:calc_x2]
                dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
                sigma = spread_radius * self.SEMANTIC_SIGMA_RATIO
                gaussian = np.exp(-dist_sq / (2 * sigma**2))
                
                semantic_layer[calc_y1:calc_y2, calc_x1:calc_x2] += gaussian * score
                
        return semantic_layer
    
    def generate_heatmap(self,
                         refined_json_path: str,
                         analysis_json_path: str,
                         image_path: Optional[str] = None) -> Dict:
        """
        生成混乱热力图
        
        Args:
            refined_json_path: refined_json 文件路径
            analysis_json_path: analysis_output JSON 文件路径
            image_path: 原始图像路径（可选，用于叠加显示）
            
        Returns:
            Dict: {
                'heatmap': 归一化热力矩阵 (H, W)，值域 [0, 1],
                'heatmap_colored': 彩色热力图 (H, W, 3) BGR,
                'overlay': 叠加原图的结果 (H, W, 3) BGR（如果提供了 image_path）,
                'layers': 各层热力图（用于调试/分析）,
                'hot_regions': 高热区域列表
            }
        """
        # 加载数据
        refined_data, analysis_data = self.load_data(refined_json_path, analysis_json_path)
        
        # 优先从原图获取精确尺寸，避免错位
        if image_path is None:
            image_path = refined_data.get('image', '')
        
        if image_path and Path(image_path).exists():
            original_img = cv2.imread(image_path)
            if original_img is not None:
                H, W = original_img.shape[:2]
            else:
                img_size = analysis_data.get('metadata', {}).get('img_size', [1080, 1920])
                H, W = img_size[0], img_size[1]
        else:
            img_size = analysis_data.get('metadata', {}).get('img_size', [1080, 1920])
            H, W = img_size[0], img_size[1]
            original_img = None
        
        # 从 analysis_data 获取 objects_clean（包含更多属性信息）
        objects_clean = analysis_data.get('objects_clean', [])
        
        # 从 refined_data 获取原始对象（包含 SAM3 数据）
        refined_objects = refined_data.get('objects', [])
        
        # 计算各层
        density_layer = self._compute_density_layer(objects_clean, refined_objects, H, W)
        variance_layer, convexity_layer = self._compute_sam3_geometry_layer(refined_objects, H, W)
        overlap_layer = self._compute_overlap_layer(objects_clean, H, W)
        semantic_layer = self._compute_semantic_layer(objects_clean, H, W)
        
        # 归一化各层到 [0, 1]
        def safe_normalize(layer):
            max_val = layer.max()
            if max_val > 0:
                return layer / max_val
            return layer
        
        density_norm = safe_normalize(density_layer)
        variance_norm = safe_normalize(variance_layer)
        convexity_norm = safe_normalize(convexity_layer)
        overlap_norm = safe_normalize(overlap_layer)
        semantic_norm = safe_normalize(semantic_layer)
        
        # 加权融合
        heatmap = (
            self.weights['density'] * density_norm +
            self.weights['variance'] * variance_norm +
            self.weights['convexity'] * convexity_norm +
            self.weights['overlap'] * overlap_norm +
            self.weights['semantic'] * semantic_norm
        )
        
        # 轻微的全局平滑，让边缘更自然但保留检测框结构
        diagonal = np.sqrt(H**2 + W**2)
        sigma = diagonal * self.gaussian_sigma_ratio
        # 控制模糊范围：不要太大以保留结构
        sigma = max(8, min(sigma, 25))
        heatmap_smooth = gaussian_filter(heatmap, sigma=sigma)
        
        # 最终归一化
        heatmap_final = safe_normalize(heatmap_smooth)
        
        # 轻微的对比度增强
        heatmap_final = np.power(heatmap_final, 0.85)
        
        # 生成彩色热力图（蓝 → 青 → 绿 → 黄 → 红）
        heatmap_colored = self._apply_colormap(heatmap_final)
        
        # 提取高热区域
        hot_regions = self._extract_hot_regions(heatmap_final, objects_clean, H, W)
        
        result = {
            'heatmap': heatmap_final,
            'heatmap_colored': heatmap_colored,
            'layers': {
                'density': density_norm,
                'variance': variance_norm,
                'convexity': convexity_norm,
                'overlap': overlap_norm,
                'semantic': semantic_norm
            },
            'hot_regions': hot_regions,
            'image_size': (H, W)
        }
        
        # 生成叠加图（使用之前已加载的原图）
        if original_img is not None:
            overlay = self._overlay_heatmap(original_img, heatmap_final)
            result['overlay'] = overlay
            result['original_image'] = original_img
        elif image_path and Path(image_path).exists():
            original_img = cv2.imread(image_path)
            if original_img is not None:
                overlay = self._overlay_heatmap(original_img, heatmap_final)
                result['overlay'] = overlay
                result['original_image'] = original_img
                
        return result
    
    def _apply_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        应用自定义色谱：低值透明蓝 → 高值不透明红
        """
        # 使用 OpenCV 的 JET colormap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return colored
    
    def _overlay_heatmap(self, 
                         image: np.ndarray, 
                         heatmap: np.ndarray,
                         alpha_base: float = 0.25,
                         alpha_scale: float = 0.55,
                         heat_threshold: float = 0.05) -> np.ndarray:
        """
        将热力图叠加到原图上
        低热区域更透明，高热区域更不透明
        极低热度区域完全透明，保持原图清晰
        """
        img_h, img_w = image.shape[:2]
        H, W = heatmap.shape
        
        # 如果尺寸不匹配，使用高质量插值调整热力图大小
        if H != img_h or W != img_w:
            heatmap = cv2.resize(heatmap, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        
        # 生成彩色热力图
        heatmap_colored = self._apply_colormap(heatmap)
        
        # 计算动态 alpha：基于热力值
        # 低于阈值的区域完全透明
        alpha = np.where(
            heatmap > heat_threshold,
            alpha_base + alpha_scale * heatmap,
            0.0
        )
        alpha = np.clip(alpha, 0, 0.85)
        alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
        
        # Alpha 混合
        overlay = (1 - alpha_3ch) * image.astype(np.float32) + alpha_3ch * heatmap_colored.astype(np.float32)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay
    
    def _extract_hot_regions(self,
                              heatmap: np.ndarray,
                              objects: List[Dict],
                              H: int, W: int,
                              threshold: float = 0.5,
                              top_k: int = 5) -> List[Dict]:
        """
        提取高热区域及其关联的物品
        """
        hot_regions = []
        
        # 二值化找高热区域
        binary = (heatmap > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # 过滤过小区域
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 计算区域内的平均热度
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 1, -1)
            avg_heat = np.mean(heatmap[mask == 1])
            
            # 找出该区域包含的物品
            contained_items = []
            for obj in objects:
                center = obj.get('center', [0, 0])
                cx, cy = int(center[0]), int(center[1])
                if 0 <= cx < W and 0 <= cy < H and mask[cy, cx] == 1:
                    contained_items.append(obj.get('name', 'unknown'))
            
            # 判断问题类型
            problem_type = self._classify_problem(contained_items, avg_heat, objects)
            
            hot_regions.append({
                'bbox': [x, y, x + w, y + h],
                'center': [x + w // 2, y + h // 2],
                'area': area,
                'avg_heat': float(avg_heat),
                'contained_items': contained_items,
                'problem_type': problem_type,
                'contour': cnt.squeeze().tolist() if cnt.size > 0 else []
            })
        
        # 按热度排序，取 Top-K
        hot_regions.sort(key=lambda r: r['avg_heat'], reverse=True)
        return hot_regions[:top_k]
    
    def _classify_problem(self, 
                          items: List[str], 
                          heat: float,
                          all_objects: List[Dict]) -> str:
        """
        根据区域内物品分类问题类型
        """
        if not items:
            return 'empty_but_hot'
        
        # 检查是否有垃圾/脏物
        dirty_items = {'trash', 'packaging', 'emptybottle'}
        if any(item in dirty_items for item in items):
            return 'hygiene_risk'
        
        # 检查是否物品过多（密集）
        if len(items) >= 3:
            return 'overcrowded'
        
        # 检查是否有散乱物品
        scattered_items = {'scatteredpaper', 'scatteredpen', 'disorderedbook', 'disorderedcable'}
        if any(item in scattered_items for item in items):
            return 'scattered'
        
        # 检查线缆问题
        cable_items = {'powersupplycable', 'tidycable', 'disorderedcable'}
        if any(item in cable_items for item in items):
            return 'cable_mess'
        
        return 'general_mess'


def generate_messiness_heatmap(refined_json_path: str,
                                analysis_json_path: str,
                                output_dir: Optional[str] = None,
                                save_layers: bool = False) -> Dict:
    """
    便捷函数：生成混乱热力图并可选保存
    
    Args:
        refined_json_path: refined_json 文件路径
        analysis_json_path: analysis_output JSON 文件路径
        output_dir: 输出目录（可选）
        save_layers: 是否保存各层热力图
        
    Returns:
        热力图生成结果
    """
    generator = MessinessHeatmapGenerator()
    result = generator.generate_heatmap(refined_json_path, analysis_json_path)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存彩色热力图
        cv2.imwrite(str(output_path / 'heatmap_colored.png'), result['heatmap_colored'])
        
        # 保存叠加图
        if 'overlay' in result:
            cv2.imwrite(str(output_path / 'heatmap_overlay.png'), result['overlay'])
        
        # 保存各层（可选）
        if save_layers:
            layers_dir = output_path / 'layers'
            layers_dir.mkdir(exist_ok=True)
            for name, layer in result['layers'].items():
                layer_colored = generator._apply_colormap(layer)
                cv2.imwrite(str(layers_dir / f'{name}_layer.png'), layer_colored)
        
        # 保存热区信息
        hot_regions_path = output_path / 'hot_regions.json'
        with open(hot_regions_path, 'w', encoding='utf-8') as f:
            json.dump(result['hot_regions'], f, ensure_ascii=False, indent=2)
            
        print(f"Heatmap saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    # 示例用法
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    refined_path = project_root / "data/refined_json/current_refined.json"
    analysis_path = project_root / "analysis_output/last_result.json"
    output_dir = project_root / "analysis_output/heatmap"
    
    if refined_path.exists() and analysis_path.exists():
        result = generate_messiness_heatmap(
            str(refined_path),
            str(analysis_path),
            str(output_dir),
            save_layers=True
        )
        print(f"Found {len(result['hot_regions'])} hot regions")
        for i, region in enumerate(result['hot_regions']):
            print(f"  Region {i+1}: heat={region['avg_heat']:.2f}, type={region['problem_type']}, items={region['contained_items']}")
    else:
        print("JSON files not found. Please run the analysis pipeline first.")
