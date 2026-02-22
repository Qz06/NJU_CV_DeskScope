import copy
import json
import os
from datetime import datetime
from pathlib import Path
from PIL import Image

import numpy as np

messy_things = {
    "pen", "eraser", "ruler", "scissors", "sticky notes", "paper clip",
    "stapler", "cable", "charger", "mouse", "earphones", "headphones",
    "mobile phone", "tablet", "keys", "wallet", "glasses", "watch",
    "package", "snack box", "carton", "box", "plastic bottle", "can",
    "disposable cup", "bottle cap", "straw", "paper", "plastic bag",
    "toy", "remote controller", "gamepad", "power bank"
}

learning_things = {
    "pen", "eraser", "ruler", "scissors", "sticky notes",
    "paper clip", "stapler", "book", "document", "paper",
    "computer", "laptop", "monitor", "keyboard", "mouse",
    "calculator", "lamp", "printer", "pencil case"
}

dirty_things = {
    "snack leftovers", "food wrapper", "waste paperball",
    "waste package", "tissue", "plastic bag", "broken items",
    "debris", "chopsticks", "food container", "instant noodles",
    "takeout", "bowl", "plastic bottle", "can", "disposable cup",
    "banana", "apple", "orange", "carton", "snack box"
}

special_things = {
    "book",            # 可能正在学习 / 复习
    "document",        # 可能与工作或学习任务有关
    "studentcard" if False else "card",  # 身份/权限信息
    "mask",            # 可能与健康或防护意识有关
    "ointment",        # 可能存在身体不适
    "backpack",        # 学生/通勤/外出状态
    "handbag",
    "umbrella",        # 天气/出行状态
    "power bank",      # 长时间外出或电量焦虑
    "clock",           # 时间管理意识
    "calculator",     # 学习/考试相关
    "gamepad",         # 游戏/娱乐偏好
    "speaker"          # 娱乐/媒体使用习惯
}



super_classes = {

    'learning': {
        'book', 'document', 'paper',
        'pen', 'eraser', 'ruler', 'scissors',
        'sticky notes', 'paper clip', 'stapler',
        'calculator', 'pencil case'
    },

    'electronics': {
        'computer', 'laptop', 'monitor', 'keyboard', 'mouse',
        'tablet', 'mobile phone', 'printer',
        'headphones', 'earphones', 'speaker'
    },

    'powerAndCable': {
        'charger', 'cable', 'socket', 'power bank'
    },

    'foodAndDrink': {
        'cup', 'thermos cup', 'plastic cup',
        'bottle', 'plastic bottle', 'can', 'disposable cup',
        'bowl', 'waterpot', 'squeeze bottle',
        'apple', 'banana', 'orange',
        'instant noodles', 'takeout',
        'snack box', 'food container', 'carton'
    },

    'garbageAndDirty': {
        'snack leftovers', 'food wrapper',
        'waste paperball', 'waste package',
        'tissue', 'plastic bag',
        'bottle cap', 'straw',
        'broken items', 'debris'
    },

    'smallStuff': {
        'keys', 'wallet', 'glasses', 'watch',
        'card', 'comb', 'mirror',
        'mouse pad', 'phone holder'
    },

    'entertainment': {
        'toy', 'remote controller', 'gamepad'
    },

    'medicine': {
        'ointment', 'mask'
    }
}


partition_zones = {

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





name2super = {}
for super_name, sub_set in super_classes.items():
    for x in sub_set:
        name2super[x] = super_name


def to_py(o):
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [to_py(x) for x in o.tolist()]
    if isinstance(o, set):
        return [to_py(x) for x in o]
    if isinstance(o, list):
        return [to_py(x) for x in o]
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    return o


def pairwise_iou_areas(boxes):  # 计算重叠面积和总面积
    N = boxes.shape[0]
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    inter = inter_w * inter_h
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter, areas


def distribution_entropy(boxes, areas, H_img, W_img, grid_size=(4, 4)):
    GY, GX = grid_size
    cell_h = H_img / GY
    cell_w = W_img / GX
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
    ix = np.clip((centers_x // cell_w).astype(int), 0, GX - 1)
    iy = np.clip((centers_y // cell_h).astype(int), 0, GY - 1)
    idx = iy * GX + ix
    pk_raw = np.zeros(GX * GY, dtype=float)
    for i, a in enumerate(areas):
        pk_raw[idx[i]] += a
    total = pk_raw.sum() + 1e-9
    pk = pk_raw / total
    pk_nonzero = pk[pk > 0]
    H = -np.sum(pk_nonzero * np.log(pk_nonzero))
    return H / np.log(GX * GY)


def get_pos_variance(groups):  # 计算同类物品的分散程度（位置方差）
    variance = []
    for name, posList in groups.items():
        if len(posList) == 0:
            continue
        posList = np.array(posList)
        var_x = np.mean((posList[:, 0] - np.mean(posList[:, 0])) ** 2)
        var_y = np.mean((posList[:, 1] - np.mean(posList[:, 1])) ** 2)
        variance.append((var_x + var_y) / 2)
    if not variance:
        return np.nan
    return np.mean(variance)



def analyse_from_json(json_path, output_path=None):
    """
    从 data/refined_json 中的 JSON 文件读取检测结果并进行分析。
    
    Args:
        json_path: refined_json 文件的路径
        output_path: 输出分析结果 JSON 文件的路径，默认为 analysis_output/last_result.json
    
    Returns:
        分析结果字典
    """
    # 加载 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 从 JSON 中提取图像尺寸
    image_path = data.get('image', '')
    if os.path.exists(image_path):
        img = Image.open(image_path)
        W, H = img.size
    else:
        # 如果图像不存在，尝试从 bbox 推断尺寸
        H, W = 5712, 4284  # 默认值，可根据实际情况调整
    
    objects_list = data.get('objects', [])
    
    tidy_score, learning_score, hygiene_score = 100, 50, 100
    
    desk_size = 0
    dx1, dy1, dx2, dy2 = 0, 0, 0, 0
    total_count, messy_count, learning_count, dirty_count = 0, 0, 0, 0
    
    union_mask = np.zeros((H, W), dtype=np.uint8)
    
    groups = {k: [] for k in super_classes.keys()}
    
    objects_raw = []
    objects_clean = []
    dirty_objs = []
    
    # 收集非 desk 物体的 bbox 用于后续计算
    non_desk_boxes = []
    
    # SAM3 几何特征收集
    sam3_contour_variances = []  # 收集所有 SAM3 对象的 normalized_contour_variance
    sam3_convexity_ratios = []   # 收集所有 SAM3 对象的 convexity_ratio
    sam3_objects_detail = []     # 保存 SAM3 分析的详细信息
    
    for obj in objects_list:
        name = obj['class']
        xyxy = obj['bbox_xyxy']
        confidence = obj.get('confidence', 1.0)
        
        if name == 'desk':
            dx1, dy1, dx2, dy2 = xyxy
            desk_size = max((dx2 - dx1) * (dy2 - dy1), desk_size)
        else:
            total_count += 1
            
            x1, y1, x2, y2 = xyxy
            center_pos = [(x1 + x2) / 2, (y1 + y2) / 2]
            
            objects_raw.append({
                'name': name,
                'bbox_xyxy': xyxy,
                'confidence': confidence
            })
            
            cur_obj_clean_node = {
                'name': name,
                'class': None,
                'area': (x2 - x1) * (y2 - y1),
                'center': center_pos,
                'bbox': xyxy,
                'is_learning_related': False,
                'is_messy_related': False,
                'is_dirty_related': False,
                'is_special_related': False
            }
            
            # 更新 union_mask
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            ix1, iy1 = max(0, ix1), max(0, iy1)
            ix2, iy2 = min(W, ix2), min(H, iy2)
            if ix1 < ix2 and iy1 < iy2:
                union_mask[iy1:iy2, ix1:ix2] = 1
            
            non_desk_boxes.append(xyxy)
            
            if name in name2super:
                groups[name2super[name]].append(np.array(center_pos))
                cur_obj_clean_node['class'] = name2super[name]
            
            if name in messy_things:
                messy_count += 1
                cur_obj_clean_node['is_messy_related'] = True
            if name in learning_things:
                learning_count += 1
                cur_obj_clean_node['is_learning_related'] = True
            if name in dirty_things:
                dirty_count += 1
                cur_obj_clean_node['is_dirty_related'] = True
                dirty_objs.append(name)
            if name in special_things:
                cur_obj_clean_node['is_special_related'] = True
            
            # 提取 SAM3 几何特征（如果存在）
            geometry_required = obj.get('geometry_required', False)
            sam3_data = obj.get('sam3', None)
            
            if geometry_required and sam3_data is not None:
                contour_var = sam3_data.get('normalized_contour_variance', None)
                convexity = sam3_data.get('convexity_ratio', None)
                
                sam3_obj_info = {
                    'name': name,
                    'normalized_contour_variance': contour_var,
                    'convexity_ratio': convexity,
                    'mask_area': sam3_data.get('mask_area', 0),
                    'aspect_ratio': sam3_data.get('aspect_ratio', 0)
                }
                sam3_objects_detail.append(sam3_obj_info)
                
                # 添加到整体特征列表
                if contour_var is not None:
                    sam3_contour_variances.append(contour_var)
                if convexity is not None:
                    sam3_convexity_ratios.append(convexity)
                
                # 将 SAM3 信息也添加到 clean 对象中
                cur_obj_clean_node['sam3_geometry'] = sam3_obj_info
            
            objects_clean.append(cur_obj_clean_node)
    
    covered_area = np.sum(union_mask)
    coverage_rate = covered_area / desk_size if desk_size != 0 else covered_area / (H * W)
    
    # 转换为 numpy 数组进行计算
    boxes = np.array(non_desk_boxes) if non_desk_boxes else np.array([]).reshape(0, 4)
    
    if boxes.size > 0:
        inter, area = pairwise_iou_areas(boxes)
        sum_pair_inter = np.sum(np.triu(inter, k=1))
        sum_areas = np.sum(area)
        overlap_ratio_pairwise = sum_pair_inter / (sum_areas + 1e-9)
        distribution_entropy_score = distribution_entropy(boxes, area, H, W)
    else:
        overlap_ratio_pairwise = 0.0
        distribution_entropy_score = 0.0
    
    posVar = get_pos_variance(groups)
    # 归一化方差
    if np.isnan(posVar):
        posVar = 0.0
    else:
        scale_area = desk_size if desk_size > 0 else H * W
        if scale_area <= 0:
            posVar = 0.0
        else:
            posVar = posVar / scale_area
            posVar = min(posVar, 1.0)
    
    total_count = max(1, total_count)
    messy_rate, learning_rate, dirty_rate = messy_count / total_count, learning_count / total_count, dirty_count / total_count
    
    # ========== SAM3 几何特征分析 ==========
    # 计算 SAM3 对象的平均轮廓方差（越大越杂乱）
    if sam3_contour_variances:
        avg_contour_variance = np.mean(sam3_contour_variances)
        max_contour_variance = np.max(sam3_contour_variances)
    else:
        avg_contour_variance = 0.0
        max_contour_variance = 0.0
    
    # 计算 SAM3 对象的平均凸包率（越小越杂乱）
    # convexity_ratio 范围是 (0, 1]，1 表示完美凸包
    # 我们用 (1 - convexity_ratio) 作为杂乱度指标
    if sam3_convexity_ratios:
        avg_convexity_ratio = np.mean(sam3_convexity_ratios)
        min_convexity_ratio = np.min(sam3_convexity_ratios)
        # 杂乱度指标：越低的凸包率越杂乱
        avg_non_convexity = 1.0 - avg_convexity_ratio
    else:
        avg_convexity_ratio = 1.0
        min_convexity_ratio = 1.0
        avg_non_convexity = 0.0
    
    # SAM3 几何杂乱度综合得分
    # normalized_contour_variance 典型范围约 0.05~0.3，归一化到 0~1
    contour_var_normalized = min(avg_contour_variance / 0.3, 1.0)
    # avg_non_convexity 已经在 0~1 范围内
    
    # SAM3 几何特征对杂乱度的贡献（综合两个指标）
    sam3_messiness_score = 0.6 * contour_var_normalized + 0.4 * avg_non_convexity
    
    print(f'coverage_rate:{coverage_rate}')
    print(f'messy_rate:{messy_rate}')
    print(f'overlap_ratio_pairwise:{overlap_ratio_pairwise}')
    print(f'posVar:{posVar}')
    print(f'distribution_entropy:{distribution_entropy_score}')
    print(f'dirty_count:{dirty_count}')
    print(f'--- SAM3 Geometry Analysis ---')
    print(f'sam3_objects_count: {len(sam3_objects_detail)}')
    print(f'avg_contour_variance: {avg_contour_variance:.4f}')
    print(f'avg_convexity_ratio: {avg_convexity_ratio:.4f}')
    print(f'sam3_messiness_score: {sam3_messiness_score:.4f}')
    
    # 更新后的 tidy_score 计算
    # 原始权重: coverage(10) + messy_rate(20) + overlap(20) + posVar(30) + entropy(15) = 95
    # 新增 SAM3 几何特征权重: 15
    # 重新分配权重使总和保持合理
    tidy_score -= (
        8 * coverage_rate +           # 覆盖率（略降）
        15 * messy_rate +              # 杂乱物品比例（略降）
        15 * overlap_ratio_pairwise +  # 重叠率（略降）
        25 * posVar +                  # 位置分散度（略降）
        12 * distribution_entropy_score +  # 分布熵（略降）
        15 * sam3_messiness_score      # 新增：SAM3 几何杂乱度
    )
    
    learning_score += 100 * learning_rate - 0.3 * (100 - tidy_score) - 15 * dirty_rate
    hygiene_score -= 10 * dirty_count + 45 * dirty_rate
    
    # 结构化信息生成：
    
    scores = {
        'tidy': tidy_score,
        'learning': learning_score,
        'hygiene': hygiene_score
    }
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'desk_area': desk_size,
        'img_size': [H, W],
        'source_json': str(json_path)
    }
    
    spatial_relations = {
        'proximity_pairs': [],
        'crowded_regions': [],
        'crowded_center': {},
        'overlap_ratio': overlap_ratio_pairwise,
        'position_distribution_variance': posVar
    }
    
    # 如果没有检测到 desk，使用整个图像作为桌面区域
    if desk_size == 0 or (dx2 - dx1) <= 0 or (dy2 - dy1) <= 0:
        dx1, dy1, dx2, dy2 = 0, 0, W, H
        desk_size = W * H
    
    desklen_normal = np.sqrt(desk_size) if desk_size > 0 else np.sqrt(H * W)
    proximity_threshold = 0.2  # 近邻阈值
    proximity_pairs = []
    desk_diag = np.sqrt((dx2 - dx1) ** 2 + (dy2 - dy1) ** 2)
    if desk_diag > 0:
        for i, obj1 in enumerate(objects_clean):
            for j, obj2 in enumerate(objects_clean):
                if i >= j:
                    continue
                center1 = np.array(obj1['center'])
                center2 = np.array(obj2['center'])
                distance = np.linalg.norm(center1 - center2)
                if distance / desk_diag <= proximity_threshold:
                    proximity_pairs.append(
                        [obj1['name'], obj2['name'], distance / desk_diag])
    spatial_relations['proximity_pairs'] = proximity_pairs
    
    desk_center_x = (dx1 + dx2) / 2
    desk_center_y = (dy1 + dy2) / 2
    region_threshold_x = (dx2 - dx1) / 3 if (dx2 - dx1) > 0 else W / 3
    region_threshold_y = (dy2 - dy1) / 3 if (dy2 - dy1) > 0 else H / 3
    regions = {
        'center': [],
        'left': [],
        'right': [],
        'top': [],
        'bottom': [],
        'top_left': [],
        'top_right': [],
        'bottom_left': [],
        'bottom_right': []
    }
    for obj in objects_clean:
        center_x, center_y = obj['center']
        if dx1 <= center_x <= dx2 and dy1 <= center_y <= dy2:
            if abs(center_x - desk_center_x) <= region_threshold_x and abs(
                    center_y - desk_center_y) <= region_threshold_y:
                regions['center'].append(obj)
            elif center_x < desk_center_x - region_threshold_x:
                if center_y < desk_center_y - region_threshold_y:
                    regions['top_left'].append(obj)
                elif center_y > desk_center_y + region_threshold_y:
                    regions['bottom_left'].append(obj)
                else:
                    regions['left'].append(obj)
            elif center_x > desk_center_x + region_threshold_x:
                if center_y < desk_center_y - region_threshold_y:
                    regions['top_right'].append(obj)
                elif center_y > desk_center_y + region_threshold_y:
                    regions['bottom_right'].append(obj)
                else:
                    regions['right'].append(obj)
            else:
                if center_y < desk_center_y - region_threshold_y:
                    regions['top'].append(obj)
                elif center_y > desk_center_y + region_threshold_y:
                    regions['bottom'].append(obj)
    for region_name, items in regions.items():
        spatial_relations['crowded_regions'].append({
            'region': region_name,
            'item_count': len(items),
            'items': [item['name'] for item in items]
        })
    
    if objects_clean:
        centers = np.array([obj['center'] for obj in objects_clean])
        crowded_center = centers.mean(axis=0)
        radius = desklen_normal / 5
        item_count = sum(
            np.linalg.norm(center - crowded_center) <= radius
            for center in centers
        )
        spatial_relations['crowded_center'] = {
            'center': crowded_center.tolist(),
            'item_count': item_count
        }
    else:
        spatial_relations['crowded_center'] = {
            'center': [0, 0],
            'item_count': 0
        }
    
    statistics = {
        'coverage_rate': coverage_rate,
        'messy_rate': messy_rate,
        'learning_rate': learning_rate,
        'dirty_rate': dirty_rate,
        'object_count': total_count,
        # SAM3 几何特征统计
        'sam3_analysis': {
            'objects_analyzed': len(sam3_objects_detail),
            'avg_contour_variance': float(avg_contour_variance),
            'max_contour_variance': float(max_contour_variance),
            'avg_convexity_ratio': float(avg_convexity_ratio),
            'min_convexity_ratio': float(min_convexity_ratio),
            'geometry_messiness_score': float(sam3_messiness_score),
            'objects_detail': sam3_objects_detail
        }
    }
    
    semantic_groups = copy.deepcopy(super_classes)
    semantic_groups['electronics'] = {'keyboard', 'mouse', 'laptop', 'monitor', 'tablet', 'pad', 'watch', 'headphone',
                                      'phone', 'razor', 'earphone'}
    
    # 更新 situation_summary，包含 SAM3 分析信息
    sam3_summary = ""
    if sam3_objects_detail:
        sam3_summary = f"通过精细分割分析了{len(sam3_objects_detail)}个物体的几何特征，" \
                       f"平均轮廓方差为{avg_contour_variance:.3f}，平均凸包率为{avg_convexity_ratio:.3f}。"
    
    situation_summary = {
        'general': f'桌面上共有{total_count}个物品，覆盖率为{coverage_rate:.2f}，杂乱率为{messy_rate:.2f}，学习相关物品比例为{learning_rate:.2f}，不卫生物品比例为{dirty_rate:.2f}。',
        'cleanliness': f'发现{dirty_count}个可能的卫生风险（{dirty_objs}）',
        'geometry_analysis': sam3_summary
    }
    
    output_to_json = {
        'metadata': metadata,
        'objects_raw': objects_raw,
        'objects_clean': objects_clean,
        'scores': scores,
        'spatial_relations': spatial_relations,
        'statistics': statistics,
        'semantic_groups': semantic_groups,
        'situation_summary': situation_summary
    }
    
    # 确定输出路径
    if output_path is None:
        os.makedirs("analysis_output", exist_ok=True)
        output_path = "analysis_output/last_result.json"
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(to_py(output_to_json), f, ensure_ascii=False, indent=2)
    
    print(f"Analysis result saved to: {output_path}")
    
    # 将 objects_clean 按类别分组
    items_by_category = {}
    for obj in objects_clean:
        category = obj['class']
        if category not in items_by_category:
            items_by_category[category] = []
        items_by_category[category].append(obj)
    
    return {
        'tidy': tidy_score,
        'learning': learning_score,
        'hygiene': hygiene_score,
        'items': items_by_category
    }


# 便捷函数：直接处理 data/refined_json 目录下的文件
def analyse_refined_json(json_filename, output_dir="analysis_output"):
    """
    分析 data/refined_json 目录下的 JSON 文件。
    
    Args:
        json_filename: JSON 文件名（如 'desk19_refined.json'）
        output_dir: 输出目录
    
    Returns:
        分析结果字典
    """
    # 获取项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # utils -> project_root
    
    json_path = project_root / "data" / "refined_json" / json_filename
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # 生成输出文件名
    base_name = Path(json_filename).stem
    output_path = project_root / output_dir / f"{base_name}_analysis.json"
    
    return analyse_from_json(str(json_path), str(output_path))


if __name__ == "__main__":
    # 示例用法
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        result = analyse_refined_json(json_file)
    else:
        # 默认处理 desk19_refined.json
        result = analyse_refined_json("desk19_refined.json")
    
    print(f"\nScores: tidy={result['tidy']:.2f}, learning={result['learning']:.2f}, hygiene={result['hygiene']:.2f}")
