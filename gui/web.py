import streamlit as st
import cv2
import numpy as np
import json
import sys
import os
import time
from pathlib import Path
from PIL import Image

# --- 1. 环境与路径配置 ---
# 确保项目根目录在 sys.path 中，以便导入 pipeline 和 utils
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
try:
    from pipeline.run_yolo import run_yolo
    from pipeline.run_sam3 import run_sam3
    from pipeline.refine_geometry import analyze_mask
    from utils.analysis import analyse_from_json
    from utils.suggest import generate_suggestion
    from utils.visualize import draw_yolo_boxes, draw_detection_with_sam3
    from utils.heatmap import MessinessHeatmapGenerator
    from utils.clustering import ZoneClusteringEngine, ZoneVisualizer, ZoneClusteringConfig
    
    # SAM3 依赖
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    st.error(f"模块导入失败，请检查环境配置: {e}")
    st.stop()

# --- 2. 页面配置与样式 ---
st.set_page_config(
    page_title="DeskScope 智能桌面分析",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    """注入自定义 CSS 以提升设计感"""
    st.markdown("""
    <style>
        /* 全局字体与背景 - 明亮清新风格 */
        .stApp {
            background-color: #f8f9fc; /* 极淡的灰蓝色背景 */
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            color: #333333;
        }
        
        /* 侧边栏背景 */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #f0f2f6;
        }
        
        /* 标题样式 */
        h1, h2, h3 {
            color: #2c3e50;
            font-weight: 700;
        }
        
        h1 {
            background: linear-gradient(120deg, #2980b9, #6dd5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* 卡片式容器 - 纯白悬浮感 */
        .metric-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05); /* 柔和阴影 */
            transition: all 0.3s ease;
            border: 1px solid #f0f0f0;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 36px;
            font-weight: 800;
            margin: 10px 0;
            font-family: 'Segoe UI', sans-serif;
        }
        .metric-label {
            font-size: 14px;
            color: #95a5a6;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            font-weight: 600;
        }
        
        /* 颜色定义 - 调整为适合浅色背景的鲜艳色 */
        .color-primary { color: #3498db; }
        .color-success { color: #2ecc71; }
        .color-warning { color: #f1c40f; }
        .color-danger { color: #e74c3c; }
        
        /* 建议卡片 */
        .suggestion-box {
            background-color: #ffffff;
            border-left: 5px solid #3498db;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            line-height: 1.8;
            color: #444;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03);
            font-size: 16px;
        }
        
        /* TODO 列表样式 */
        .todo-item {
            background-color: #ffffff;
            margin: 10px 0;
            padding: 15px 20px;
            border-radius: 10px;
            border-left: 5px solid #ccc;
            display: flex;
            align-items: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
            transition: transform 0.2s;
            border: 1px solid #f5f5f5;
        }
        .todo-item:hover {
            transform: translateX(5px);
        }
        .todo-high { border-left-color: #ff6b6b; background-color: #fff5f5; }
        .todo-mid { border-left-color: #feca57; background-color: #fffdf0; }
        .todo-low { border-left-color: #1dd1a1; background-color: #f0fff4; }
        
        .todo-tag {
            font-size: 11px;
            padding: 4px 10px;
            border-radius: 20px;
            margin-right: 15px;
            font-weight: 700;
            color: #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .tag-high { background: linear-gradient(135deg, #ff6b6b, #ee5253); }
        .tag-mid { background: linear-gradient(135deg, #feca57, #ff9f43); }
        .tag-low { background: linear-gradient(135deg, #1dd1a1, #10ac84); }

        /* 图片容器 */
        .img-container {
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 4px solid #fff;
        }
        
        /* 去除 Streamlit 默认的 padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* 隐藏 Streamlit页脚 */
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 3. 辅助函数 ---

def load_image(image_path):
    """加载并调整图片大小"""
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        st.error(f"无法加载图片: {e}")
        return None

def save_uploaded_file(uploaded_file):
    """保存上传的文件到临时目录"""
    temp_dir = project_root / "data" / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)

def get_local_images():
    """获取 data/images 下的本地图片列表"""
    img_dir = project_root / "data" / "images"
    if not img_dir.exists():
        return []
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return [f.name for f in img_dir.iterdir() if f.suffix.lower() in extensions]

def parse_todos(suggestion_text):
    """从 LLM 建议中解析 TODO 列表"""
    todos = []
    lines = suggestion_text.split('\n')
    for line in lines:
        line = line.strip()
        priority = "一般"
        if '【高优先级】' in line:
            priority = "High"
            content = line.replace('【高优先级】', '').strip()
        elif '【中优先级】' in line:
            priority = "Medium"
            content = line.replace('【中优先级】', '').strip()
        elif '【低优先级】' in line:
            priority = "Low"
            content = line.replace('【低优先级】', '').strip()
        else:
            continue
            
        # 清理内容
        content = content.lstrip('0123456789. 、：:')
        if content:
            todos.append({"priority": priority, "content": content})
    return todos

# --- 4. 核心分析逻辑 ---

def run_full_analysis(image_path):
    """执行完整的分析 pipeline"""
    status_container = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Step 1: YOLO
        status_container.info("[1/6] 正在运行目标检测...")
        yolo_model_path = project_root / "models/yolo/best.pt"
        yolo_result = run_yolo(image_path, str(yolo_model_path), show_result=False)
        progress_bar.progress(20)
        
        # Step 2: SAM3
        status_container.info("[2/6] 正在进行图像精细分割...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam3_model = build_sam3_image_model(device=device)
        sam3_processor = Sam3Processor(sam3_model)
        
        pil_image = Image.open(image_path).convert("RGB")
        total_objs = len(yolo_result["objects"])
        
        # 为每个物体运行 SAM3
        for i, obj in enumerate(yolo_result["objects"]):
            status_container.info(f"[2/6] 正在分析物体 ({i+1}/{total_objs}): {obj['class']}...")
            mask = run_sam3(sam3_processor, pil_image, obj["bbox_xyxy"])
            geo = analyze_mask(mask, image_path=image_path)
            obj["sam3"] = geo
            
        progress_bar.progress(50)
        
        # 保存结果
        refined_json_path = project_root / "data/refined_json/web_refined.json"
        refined_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(refined_json_path, "w", encoding="utf-8") as f:
            json.dump(yolo_result, f, indent=2, ensure_ascii=False)
            
        # Step 3: Analysis
        status_container.info("[3/6] 正在计算多维特征...")
        # 直接输出到 last_result.json，以便 utils.suggest 模块能够读取到包含 metadata 的完整数据
        analysis_output_path = project_root / "analysis_output/last_result.json"
        
        # 获取基础评分
        base_scores = analyse_from_json(str(refined_json_path), str(analysis_output_path))
        
        # 读取完整详细数据以获取统计信息 (如 item_count)
        try:
            with open(analysis_output_path, "r", encoding="utf-8") as f:
                full_data = json.load(f)
            
            # 合并数据：将 full_data 中的 statistics 和 scores 字段合并到返回结果中
            analysis_result = base_scores.copy()
            if "statistics" in full_data:
                analysis_result.update(full_data["statistics"]) # 获取 object_count, messy_rate 等
            if "scores" in full_data:
                # 注意不要覆盖 base_scores 中的核心评分，只补充缺失的如 entropy
                for k, v in full_data["scores"].items():
                    if k not in analysis_result:
                        analysis_result[k] = v
        except Exception as e:
            print(f"读取完整分析结果失败: {e}")
            analysis_result = base_scores
            
        progress_bar.progress(65)
        
        # Step 4: Heatmap
        status_container.info("[4/6] 正在生成混乱热力图...")
        heatmap_gen = MessinessHeatmapGenerator()
        heatmap_res = heatmap_gen.generate_heatmap(str(refined_json_path), str(analysis_output_path), image_path)
        progress_bar.progress(80)
        
        # Step 5: Clustering (AR Zones)
        status_container.info("[5/6] 正在规划桌面功能分区...")
        clustering_config = ZoneClusteringConfig()
        clustering_engine = ZoneClusteringEngine(clustering_config)
        
        # 读取图片尺寸用于归一化
        img_h, img_w = np.array(pil_image).shape[:2]
        
        clustering_res = clustering_engine.cluster(yolo_result['objects'], (img_w, img_h))
        
        # 渲染聚类图
        font_path = project_root / "assets/fonts/SIMYOU.TTF"
        visualizer = ZoneVisualizer(font_path=str(font_path) if font_path.exists() else None)
        zone_img_bgr = visualizer.render(np.array(pil_image)[:, :, ::-1].copy(), clustering_res.zones)
        zone_img_rgb = cv2.cvtColor(zone_img_bgr, cv2.COLOR_BGR2RGB)
        progress_bar.progress(90)
        
        # Step 6: LLM Suggestion
        status_container.info("[6/6] AI 顾问正在生成优化建议...")
        try:
            # 由于 Step 3 已经生成了完整的 last_result.json，这里直接调用即可
            suggestion = generate_suggestion()
        except Exception as e:
            suggestion = f"无法生成建议: {e}"
            
        progress_bar.progress(100)
        status_container.success("分析完成")
        time.sleep(1)
        status_container.empty()
        progress_bar.empty()
        
        # 绘制检测结果图
        detection_img = draw_detection_with_sam3(np.array(pil_image)[:, :, ::-1], yolo_result["objects"])
        detection_img_rgb = cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB)
        
        # 热力图 Overlay
        if 'overlay' in heatmap_res:
            heatmap_img_rgb = cv2.cvtColor(heatmap_res['overlay'], cv2.COLOR_BGR2RGB)
        else:
            heatmap_img_rgb = detection_img_rgb # Fallback
            
        return {
            "scores": analysis_result,
            "suggestion": suggestion,
            "images": {
                "original": pil_image,
                "detection": Image.fromarray(detection_img_rgb),
                "heatmap": Image.fromarray(heatmap_img_rgb),
                "zones": Image.fromarray(zone_img_rgb)
            },
            "clustering": clustering_res,
            "hot_regions": heatmap_res.get('hot_regions', [])
        }
        
    except Exception as e:
        status_container.error(f"分析过程中发生错误: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# --- 5. 主界面逻辑 ---

def render_sidebar():
    with st.sidebar:
        st.title("DeskScope")
        st.markdown("---")
        
        st.subheader("1. 图像源")
        source_type = st.radio("选择模式", ["本地示例图片", "上传新图片"])
        
        file_path = None
        
        if source_type == "本地示例图片":
            local_imgs = get_local_images()
            if local_imgs:
                selected_img = st.selectbox("选择图片", local_imgs, index=0 if local_imgs else None)
                if selected_img:
                    file_path = str(project_root / "data/images" / selected_img)
            else:
                st.warning("data/images 目录下没有找到图片。")
                
        else:
            uploaded_file = st.file_uploader("上传图片 (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                file_path = save_uploaded_file(uploaded_file)
        
        st.markdown("---")
        st.subheader("2. 预览")
        if file_path and os.path.exists(file_path):
            img = load_image(file_path)
            st.image(img, caption="当前图片", use_container_width=True)
            
            st.markdown("---")
            run_btn = st.button("开始智能分析", type="primary", use_container_width=True)
            return file_path, run_btn
            
        return None, False

def render_metrics(scores):
    """渲染评分卡片"""
    cols = st.columns(3)
    
    # 整洁度 Tidy Score
    tidy = scores.get('tidy', 0)
    tidy_color = "color-success" if tidy >= 60 else "color-warning" if tidy >= 40 else "color-danger"
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tidy Score</div>
            <div class="metric-value {tidy_color}">{tidy:.1f}</div>
            <div style="font-size:12px; color:#888;">桌面整体整洁度</div>
        </div>
        """, unsafe_allow_html=True)
        
    # 学习氛围 Learning Score
    learning = scores.get('learning', 0)
    learn_color = "color-primary"
    
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Learning Vibe</div>
            <div class="metric-value {learn_color}">{learning:.1f}</div>
            <div style="font-size:12px; color:#888;">学习/工作专注度</div>
        </div>
        """, unsafe_allow_html=True)
        
    # 卫生分 Hygiene Score
    hygiene = scores.get('hygiene', 0)
    hyg_color = "color-success" if hygiene >= 80 else "color-danger"
    
    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Hygiene Score</div>
            <div class="metric-value {hyg_color}">{hygiene:.1f}</div>
            <div style="font-size:12px; color:#888;">环境卫生状况</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    inject_custom_css()
    
    # 初始化 Session State
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # 侧边栏
    file_path, run_clicked = render_sidebar()
    
    # 标题区
    st.title("DeskScope 桌面智能分析系统")
    st.markdown("基于 **YOLO + SAM3 + LLM** 的多模态桌面状态评估与整理助手")
    
    # 分析触发逻辑
    if run_clicked and file_path:
        with st.spinner("正在启动分析引擎..."):
            results = run_full_analysis(file_path)
            if results:
                st.session_state.analysis_results = results
                st.toast("分析已完成")
    
    # 结果展示区
    res = st.session_state.analysis_results
    
    if res:
        st.divider()
        
        # 1. 评分区
        render_metrics(res['scores'])
        
        st.write("") # Spacer
        
        # 2. 核心可视化 (Tab View)
        st.subheader("视觉分析概览")
        tab1, tab2, tab3 = st.tabs(["检测与语义", "混乱热力图", "功能分区规划"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(res['images']['detection'], use_container_width=True, caption="YOLO 检测 + SAM3 几何分割")
            with col2:
                st.markdown("#### 检测详情")
                scores = res['scores']
                # 优先使用 object_count (来自 statistics)，如果没有则尝试 item_count，最后默认为 0
                count = scores.get('object_count', scores.get('item_count', 0))
                st.info(f"共检测到 **{count}** 个物品")
                st.write(f"- 杂物比例: `{scores.get('messy_rate', 0)*100:.1f}%`")
                st.write(f"- 覆盖率: `{scores.get('coverage_rate', 0)*100:.1f}%`")
                st.write(f"- 分布熵: `{scores.get('entropy', 0):.2f}` (越低越整齐)")
                
        with tab2:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(res['images']['heatmap'], use_container_width=True, caption="Messiness Heatmap (Red = Messy)")
            with col2:
                st.markdown("#### 高热度区域 (Hot Regions)")
                hot_regions = res.get('hot_regions', [])
                if hot_regions:
                    for i, region in enumerate(hot_regions[:4]):
                        problem = region.get('problem_type', '未知')
                        items = ", ".join(region.get('contained_items', [])[:3])
                        heat = region.get('avg_heat', 0)
                        st.markdown(f"""
                        <div style="padding:15px; background:#fff5f5; margin-bottom:10px; border-radius:8px; border-left:4px solid #ff6b6b; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                            <strong style="color: #c0392b;">区域 {i+1}: {problem}</strong><br>
                            <small style="color:#555;">包含: {items}...</small><br>
                            <small style="color:#e74c3c; font-weight:bold;">热度: {heat:.2f}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("未检测到显著的混乱热点区域。")

        with tab3:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(res['images']['zones'], use_container_width=True, caption="Functional Zones Clustering")
            with col2:
                st.markdown("#### 智能分区结果")
                zones = res['clustering'].zones
                for z in zones:
                    st.markdown(f"""
                    <div style="padding:15px; background:#f0f8ff; margin-bottom:10px; border-radius:8px; border-left:4px solid #3498db; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                        <span style="font-size:1.2em;">{z.icon}</span> <strong style="color: #2980b9;">{z.zone_type}</strong><br>
                        <small style="color:#555;">包含 {len(z.objects)} 个物品</small>
                    </div>
                    """, unsafe_allow_html=True)

        st.divider()

        # 3. 建议与 TODO
        row = st.columns([1.5, 1])
        
        with row[0]:
            st.subheader("AI 整理顾问建议")
            # 使用 st.info 容器，它原生支持 Markdown 渲染，且样式美观
            with st.container():
                st.info(res['suggestion'])
            
        with row[1]:
            st.subheader("智能行动清单 (TODO)")
            todos = parse_todos(res['suggestion'])
            
            if todos:
                for todo in todos:
                    p = todo['priority']
                    content = todo['content']
                    
                    # 使用 Streamlit 原生组件以完美支持 Markdown 渲染
                    # 映射关系: High -> Error (红), Medium -> Warning (黄), Low -> Success (绿)
                    if p == "High":
                        st.error(f"**[高优]** {content}", icon="🔥")
                    elif p == "Medium":
                        st.warning(f"**[建议]** {content}", icon="⚡")
                    else:
                        st.success(f"**[可选]** {content}", icon="✅")
            else:
                st.info("AI 正在分析细节，主要清单请参考左侧建议全文。")

    else:
        # 空状态欢迎页
        st.info("👈 请在左侧选择一张图片并点击「开始智能分析」")
        
        # 简单的 Gallery 展示
        st.markdown("#### 功能演示")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**1. 物体检测与分割**")
            st.caption("识别桌面物品，精确勾勒轮廓")
            st.empty() # Placeholder for demo img
        with c2:
            st.markdown("**2. 混乱热力图**")
            st.caption("红外热成像般的混乱度可视化")
        with c3:
            st.markdown("**3. 全自动整理建议**")
            st.caption("LLM 生成贴心的整理计划")

if __name__ == "__main__":
    main()
