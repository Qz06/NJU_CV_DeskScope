# gui/app.py

import dearpygui.dearpygui as dpg
import numpy as np
import cv2
import json
import os
from pathlib import Path
from threading import Thread

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DeskScopeApp:
    def __init__(self):
        self.width = 1500
        self.height = 900
        
        # 视觉风格定义 (Rational & Gentle)
        self.colors = {
            "bg": (28, 30, 34),             # 深灰背景
            "card_bg": (38, 40, 45),        # 卡片背景
            "card_border": (55, 58, 65),    # 卡片边框
            "primary": (70, 130, 180),      # 柔和蓝 (Steel Blue)
            "primary_hover": (90, 150, 200),
            "primary_active": (50, 110, 160),
            "secondary": (50, 54, 60),      # 次级按钮/背景
            "text_main": (230, 230, 235),
            "text_sub": (150, 155, 165),
            "success": (100, 180, 120),     # 柔和绿
            "warning": (220, 160, 80),      # 柔和橙
            "danger": (200, 90, 90),        # 柔和红
            "accent": (220, 140, 60)
        }

        # 状态变量
        self.current_image_path = None
        self.original_image = None
        self.analyzed_image = None
        self.heatmap_image = None
        self.ar_image = None
        self.refined_json_data = None
        self.analysis_json_data = None
        self.llm_suggestion = ""
        self.todo_list = []
        
        # 是否正在处理
        self.is_processing = False
        
    def setup(self):
        """初始化 DearPyGUI 上下文和窗口"""
        dpg.create_context()
        # 注意：viewport title 不支持中文，使用英文避免乱码
        # 设置固定大小，禁止调整窗口大小
        dpg.create_viewport(
            title="DeskScope - Desktop Analyzer", 
            width=self.width, 
            height=self.height,
            resizable=False,  # 禁止调整大小
            min_width=self.width,
            min_height=self.height
        )
        
        # 注册并绑定中文字体
        self._setup_fonts()
        
        # 创建主题
        self._create_themes()
        
        # 创建主窗口
        self._create_main_window()
        
        # 设置并显示
        dpg.setup_dearpygui()
        dpg.show_viewport()
    
    def _setup_fonts(self):
        """设置中文字体"""
        font_path = PROJECT_ROOT / "assets" / "fonts" / "SIMYOU.TTF"
        
        if font_path.exists():
            with dpg.font_registry():
                # 1. 普通正文字体 (20px)
                with dpg.font(str(font_path), 20) as self.chinese_font:
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                
                # 2. 标题字体 (32px)
                with dpg.font(str(font_path), 32) as self.header_font:
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                
                # 3. 评分大字体 (28px) - 调整大小以适应卡片
                with dpg.font(str(font_path), 28) as self.score_font:
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
            
            # 绑定为全局默认字体
            dpg.bind_font(self.chinese_font)
            print(f"中文字体加载成功: {font_path}")
        else:
            print(f"字体文件不存在: {font_path}")
            print("  将使用默认字体，中文可能无法正常显示")
            self.chinese_font = None
            self.header_font = None
            self.score_font = None
        
    def _create_themes(self):
        """创建现代暗色主题 (Rational & Gentle)"""
        with dpg.theme() as self.global_theme:
            with dpg.theme_component(dpg.mvAll):
                # 基础布局样式 - 增加间距和圆角
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 20, 20)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 12, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 12, 12)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 8, 8)
                
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 10)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 10)
                dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 8)
                
                # 颜色定义
                c = self.colors
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, c["bg"])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, c["card_bg"])
                dpg.add_theme_color(dpg.mvThemeCol_Border, c["card_border"])
                dpg.add_theme_color(dpg.mvThemeCol_Text, c["text_main"])
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, c["bg"])
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, c["card_bg"])
                
                # Tab 样式
                dpg.add_theme_color(dpg.mvThemeCol_Tab, c["secondary"])
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, c["primary_hover"])
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, c["primary"])
                
                # 按钮样式
                dpg.add_theme_color(dpg.mvThemeCol_Button, c["secondary"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, c["primary_hover"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, c["primary_active"])
                
                # 输入框/Frame
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (20, 22, 25))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (30, 32, 35))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (40, 42, 45))
                
                # 滚动条
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, c["bg"])
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, c["secondary"])
                
        dpg.bind_theme(self.global_theme)
        
        # 强调按钮主题 (用于"开始分析")
        with dpg.theme() as self.primary_btn_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, self.colors["primary"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, self.colors["primary_hover"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, self.colors["primary_active"])
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))

        # 评分卡片主题 - 更明显的卡片感
        with dpg.theme() as self.score_card_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (45, 48, 55))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (70, 75, 85))
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)

        # 建议文本框主题 - 去除边框，像文档一样
        with dpg.theme() as self.suggestion_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (35, 37, 42))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (50, 53, 60))

        
    def _create_main_window(self):
        """创建主窗口布局"""
        with dpg.window(tag="main_window", label="DeskScope", no_title_bar=True, 
                       no_move=True, no_resize=True, no_collapse=True):
            
            # --- 顶部 Header ---
            with dpg.group(horizontal=True):
                # 标题
                t = dpg.add_text("DeskScope 桌面分析", color=self.colors["primary"])
                if self.header_font:
                    dpg.bind_item_font(t, self.header_font)
                
                dpg.add_spacer(width=30)
                
                # 状态栏 (更简洁)
                with dpg.group(horizontal=True):
                    dpg.add_text("状态:", color=self.colors["text_sub"])
                    dpg.add_text("就绪", tag="status_text", color=self.colors["success"])
                    
                    dpg.add_spacer(width=20)
                    dpg.add_text("当前文件:", color=self.colors["text_sub"])
                    dpg.add_text("未选择", tag="current_file_text", color=self.colors["text_main"])

            dpg.add_spacer(height=15)
            
            # --- 主体内容区 ---
            with dpg.group(horizontal=True):
                # 左侧：图片展示区
                self._create_image_panel()
                
                dpg.add_spacer(width=15)
                
                # 右侧：控制与结果区
                self._create_result_panel()
    
    def _create_image_panel(self):
        """创建左侧图片展示面板"""
        # 调整宽度以适应 1500px 窗口
        with dpg.child_window(width=960, height=820, border=False):
            # Tab 切换不同图片视图
            with dpg.tab_bar(tag="image_tabs"):
                # Tab 1: 分析结果图
                with dpg.tab(label="  检测结果  "):
                    dpg.add_spacer(height=10)
                    with dpg.child_window(height=740, border=False):
                        # 创建纹理注册表
                        with dpg.texture_registry():
                            # 默认空白图片
                            default_data = np.zeros((600, 900, 4), dtype=np.float32)
                            default_data[:, :, 3] = 1.0  # Alpha
                            dpg.add_dynamic_texture(900, 600, default_data.flatten().tolist(), 
                                                   tag="texture_analyzed")
                        
                        # 图片容器 - 居中显示
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=20)
                            dpg.add_image("texture_analyzed", tag="image_analyzed")
                        
                        dpg.add_spacer(height=15)
                        dpg.add_text("选择图片并点击 [开始分析] 查看检测结果", tag="analyzed_hint",
                                    color=self.colors["text_sub"], indent=30)
                
                # Tab 2: 热力图
                with dpg.tab(label="  混乱热力图  "):
                    dpg.add_spacer(height=10)
                    with dpg.child_window(height=740, border=False):
                        with dpg.texture_registry():
                            default_data = np.zeros((600, 900, 4), dtype=np.float32)
                            default_data[:, :, 3] = 1.0
                            dpg.add_dynamic_texture(900, 600, default_data.flatten().tolist(),
                                                   tag="texture_heatmap")
                        
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=20)
                            dpg.add_image("texture_heatmap", tag="image_heatmap")
                            
                        dpg.add_spacer(height=15)
                        dpg.add_text("点击 [开始分析] 生成混乱热力图", tag="heatmap_hint",
                                    color=self.colors["text_sub"], indent=30)
                        # 热区信息展示
                        dpg.add_spacer(height=5)
                        dpg.add_text("", tag="heatmap_info", color=self.colors["danger"], wrap=920, indent=30)
                
                # Tab 3: 桌面功能分区
                with dpg.tab(label="  功能分区  "):
                    dpg.add_spacer(height=10)
                    with dpg.child_window(height=740, border=False):
                        with dpg.texture_registry():
                            default_data = np.zeros((600, 900, 4), dtype=np.float32)
                            default_data[:, :, 3] = 1.0
                            dpg.add_dynamic_texture(900, 600, default_data.flatten().tolist(),
                                                   tag="texture_ar")
                        
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=20)
                            dpg.add_image("texture_ar", tag="image_ar")
                            
                        dpg.add_spacer(height=15)
                        dpg.add_text("点击 [开始分析] 生成桌面功能分区", tag="ar_hint",
                                    color=self.colors["text_sub"], indent=30)
                        # 分区信息展示
                        dpg.add_spacer(height=5)
                        dpg.add_text("", tag="ar_info", color=self.colors["success"], wrap=920, indent=30)

    def _create_result_panel(self):
        """创建右侧分析结果面板"""
        with dpg.child_window(width=480, height=820, border=False):
            
            # --- 1. 控制区 (Card) ---
            with dpg.child_window(height=80, border=True):
                dpg.bind_item_theme(dpg.last_item(), self.score_card_theme)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=5)
                    btn_file = dpg.add_button(label="选择图片", callback=self._open_file_dialog, width=140, height=45)
                    
                    dpg.add_spacer(width=10)
                    
                    btn_run = dpg.add_button(label="开始智能分析", callback=self._start_analysis, width=280, height=45, tag="btn_analyze")
                    dpg.bind_item_theme(btn_run, self.primary_btn_theme)
            
            dpg.add_spacer(height=15)
            
            # --- 2. 评分展示 (Card) ---
            dpg.add_text("智能评分", color=self.colors["primary"])
            
            with dpg.group(horizontal=True):
                # 整洁度
                with dpg.child_window(width=145, height=100, border=True, no_scrollbar=True):
                    dpg.bind_item_theme(dpg.last_item(), self.score_card_theme)
                    # 利用默认 ItemSpacing (12px) 自动布局，不再额外加 Spacer
                    dpg.add_text("整洁度", color=self.colors["primary"], indent=45)
                    t = dpg.add_text("--", tag="score_tidy", color=self.colors["text_main"], indent=45)
                    if self.score_font: dpg.bind_item_font(t, self.score_font)
                
                # 学习氛围
                with dpg.child_window(width=145, height=100, border=True, no_scrollbar=True):
                    dpg.bind_item_theme(dpg.last_item(), self.score_card_theme)
                    dpg.add_text("学习氛围", color=self.colors["success"], indent=40)
                    t = dpg.add_text("--", tag="score_learning", color=self.colors["text_main"], indent=45)
                    if self.score_font: dpg.bind_item_font(t, self.score_font)
                
                # 卫生状况
                with dpg.child_window(width=145, height=100, border=True, no_scrollbar=True):
                    dpg.bind_item_theme(dpg.last_item(), self.score_card_theme)
                    dpg.add_text("卫生状况", color=self.colors["warning"], indent=40)
                    t = dpg.add_text("--", tag="score_hygiene", color=self.colors["text_main"], indent=45)
                    if self.score_font: dpg.bind_item_font(t, self.score_font)
            
            dpg.add_spacer(height=10)
            
            # --- 3. AI 建议 (Card) ---
            dpg.add_text("AI 优化建议", color=self.colors["accent"])
            
            # 稍微减小高度以平衡整体布局
            with dpg.child_window(height=200, border=True):
                dpg.bind_item_theme(dpg.last_item(), self.suggestion_theme)
                dpg.add_text("点击 [开始分析] 获取 AI 建议...", tag="llm_suggestion_text", 
                            wrap=440, color=self.colors["text_main"])
            
            dpg.add_spacer(height=10)
            
            # --- 4. TODO 清单 (Card) ---
            dpg.add_text("行动清单", color=self.colors["success"])
            with dpg.child_window(height=160, border=True, tag="todo_container"):
                dpg.bind_item_theme(dpg.last_item(), self.suggestion_theme)
                dpg.add_text("分析完成后将显示待办事项...", tag="todo_placeholder",
                            color=self.colors["text_sub"], indent=10)

    def _open_file_dialog(self):
        """打开文件选择对话框"""
        with dpg.file_dialog(directory_selector=False, show=True, 
                            callback=self._file_selected_callback,
                            width=700, height=500,
                            default_path=str(PROJECT_ROOT / "data" / "images")):
            # 使用通配符模式匹配所有图片格式（不区分大小写）
            dpg.add_file_extension("Image Files (*.jpg *.jpeg *.png *.bmp){.jpg,.JPG,.jpeg,.JPEG,.png,.PNG,.bmp,.BMP}", color=self.colors["success"])
            dpg.add_file_extension("All Files (*.*){.*}", color=self.colors["text_sub"])
    
    def _file_selected_callback(self, sender, app_data):
        """文件选择回调"""
        if app_data and 'file_path_name' in app_data:
            file_path = app_data['file_path_name']
            self.current_image_path = file_path
            
            # 更新状态
            dpg.set_value("current_file_text", f"{Path(file_path).name}")
            dpg.set_value("status_text", "图片已选择，点击 [开始分析]")
            
            # 加载并预览图片
            self._load_and_preview_image(file_path)
    
    def _load_and_preview_image(self, image_path):
        """加载并预览原始图片"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                dpg.set_value("status_text", "无法加载图片")
                return
            
            self.original_image = img
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 调整大小以适应显示区域
            display_img = self._resize_for_display(img_rgb, max_width=900, max_height=600)
            
            # 更新纹理
            try:
                self._update_texture("texture_analyzed", display_img)
                dpg.set_value("analyzed_hint", "原始图片 - 点击[开始分析]查看检测结果")
            except Exception as tex_err:
                print(f"纹理更新错误: {tex_err}")
                import traceback
                traceback.print_exc()
                dpg.set_value("status_text", f"显示图片失败")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            dpg.set_value("status_text", f"加载失败: {str(e)}")
    
    def _resize_for_display(self, img, max_width=900, max_height=600):
        """调整图片大小以适应显示区域"""
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return img
    
    def _update_texture(self, texture_tag, img_rgb):
        """更新 DearPyGUI 纹理"""
        h, w = img_rgb.shape[:2]
        
        # 转换为 RGBA float32
        if len(img_rgb.shape) == 2:
            img_rgba = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGBA)
        elif img_rgb.shape[2] == 3:
            img_rgba = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2RGBA)
        else:
            img_rgba = img_rgb
        
        img_float = img_rgba.astype(np.float32) / 255.0
        flat_data = img_float.flatten().tolist()
        
        image_tag = texture_tag.replace("texture_", "image_")
        
        # 检查纹理是否存在且尺寸是否匹配
        if dpg.does_item_exist(texture_tag):
            # 获取现有纹理尺寸
            old_w = dpg.get_item_width(texture_tag)
            old_h = dpg.get_item_height(texture_tag)
            
            if old_w == w and old_h == h:
                # 尺寸相同，直接更新数据
                dpg.set_value(texture_tag, flat_data)
                return
            else:
                # 尺寸不同，需要删除重建
                if dpg.does_item_exist(image_tag):
                    parent = dpg.get_item_parent(image_tag)
                    dpg.delete_item(image_tag)
                dpg.delete_item(texture_tag)
        
        # 创建新纹理
        with dpg.texture_registry():
            dpg.add_dynamic_texture(w, h, flat_data, tag=texture_tag)
        
        # 重新创建 image 组件
        if not dpg.does_item_exist(image_tag):
            hint_tag = texture_tag.replace("texture_", "") + "_hint"
            if "analyzed" in texture_tag:
                hint_tag = "analyzed_hint"
            
            if dpg.does_item_exist(hint_tag):
                parent = dpg.get_item_parent(hint_tag)
                dpg.add_image(texture_tag, tag=image_tag, parent=parent, before=hint_tag)
            else:
                # 找不到 hint，直接创建
                dpg.add_image(texture_tag, tag=image_tag)
    
    def _start_analysis(self):
        """开始分析流程"""
        if self.is_processing:
            dpg.set_value("status_text", "正在处理中，请稍候...")
            return
        
        if not self.current_image_path:
            dpg.set_value("status_text", "请先选择图片")
            dpg.configure_item("status_text", color=self.colors["warning"])
            return
        
        # 在新线程中运行分析
        self.is_processing = True
        dpg.set_value("status_text", "正在分析中...")
        dpg.configure_item("status_text", color=self.colors["primary"])
        dpg.configure_item("btn_analyze", enabled=False)
        
        thread = Thread(target=self._run_analysis_pipeline)
        thread.start()
    
    def _run_analysis_pipeline(self):
        """运行完整分析流程（在后台线程）- 与 main.py 流程一致"""
        try:
            from PIL import Image
            import torch
            from pipeline.run_yolo import run_yolo
            from pipeline.run_sam3 import run_sam3
            from pipeline.refine_geometry import analyze_mask
            from utils.analysis import analyse_from_json
            from utils.suggest import generate_suggestion
            from utils.visualize import draw_yolo_boxes, draw_detection_with_sam3
            
            # Step 1: YOLO 粗检测
            dpg.set_value("status_text", "[1/7] 正在进行 YOLO 粗检测...")
            yolo_model_path = PROJECT_ROOT / "models/yolo/best.pt"
            # GUI 模式下不弹窗
            result = run_yolo(self.current_image_path, str(yolo_model_path), show_result=False)
            self.refined_json_data = result
            
            # 先显示 YOLO 粗检测结果
            yolo_img = draw_yolo_boxes(self.original_image, result["objects"])
            yolo_img_rgb = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)
            display_img = self._resize_for_display(yolo_img_rgb)
            self._update_texture("texture_analyzed", display_img)
            dpg.set_value("analyzed_hint", f"YOLO 检测到 {len(result['objects'])} 个物体，正在进行精细分析...")
            
            # Step 2: SAM3 精细分析（对所有检测到的物体）
            dpg.set_value("status_text", "[2/7] 正在进行 SAM3 精细分割...")
            
            total_objects = len(result["objects"])
            if total_objects > 0:
                try:
                    from sam3.model_builder import build_sam3_image_model
                    from sam3.model.sam3_image_processor import Sam3Processor
                    
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    sam3_model = build_sam3_image_model(device=device)
                    sam3_processor = Sam3Processor(sam3_model)
                    
                    pil_image = Image.open(self.current_image_path).convert("RGB")
                    
                    # 对所有检测到的物体进行 SAM3 处理
                    for idx, obj in enumerate(result["objects"], 1):
                        dpg.set_value("status_text", f"[2/7] SAM3 分析中 ({idx}/{total_objects})...")
                        
                        mask = run_sam3(
                            sam3_processor,
                            pil_image,
                            obj["bbox_xyxy"]
                        )
                        
                        geo = analyze_mask(mask, image_path=self.current_image_path)
                        obj["sam3"] = geo
                    
                    print(f"SAM3 completed for {total_objects} objects")
                except ImportError as e:
                    print(f"SAM3 模块未安装，跳过精细分析: {e}")
                except Exception as e:
                    print(f"SAM3 分析出错，继续后续流程: {e}")
            
            # 保存完整检测结果（包含 SAM3）
            refined_json_path = PROJECT_ROOT / "data/refined_json/current_refined.json"
            refined_json_path.parent.mkdir(exist_ok=True, parents=True)
            with open(refined_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Step 3: 绘制完整检测结果（YOLO + SAM3）
            dpg.set_value("status_text", "[3/7] 正在绘制检测结果...")
            analyzed_img = draw_detection_with_sam3(self.original_image, result["objects"])
            analyzed_img_rgb = cv2.cvtColor(analyzed_img, cv2.COLOR_BGR2RGB)
            display_img = self._resize_for_display(analyzed_img_rgb)
            self._update_texture("texture_analyzed", display_img)
            dpg.set_value("analyzed_hint", f"检测完成: {len(result['objects'])} 个物体")
            
            # Step 4: Analysis 深度分析
            dpg.set_value("status_text", "[4/7] 正在进行深度分析...")
            analysis_output_path = PROJECT_ROOT / "analysis_output/last_result.json"
            analysis_result = analyse_from_json(str(refined_json_path), str(analysis_output_path))
            
            # 更新评分
            dpg.set_value("score_tidy", f"{analysis_result['tidy']:.1f}")
            dpg.set_value("score_learning", f"{analysis_result['learning']:.1f}")
            dpg.set_value("score_hygiene", f"{analysis_result['hygiene']:.1f}")
            
            # 保存评分到独立文件
            scores_data = {
                'tidy': float(analysis_result['tidy']),
                'learning': float(analysis_result['learning']),
                'hygiene': float(analysis_result['hygiene']),
                'timestamp': analysis_result.get('timestamp', '')
            }
            scores_output_path = PROJECT_ROOT / "analysis_output/scores.json"
            with open(scores_output_path, 'w', encoding='utf-8') as f:
                json.dump(scores_data, f, ensure_ascii=False, indent=2)
            print(f"Scores saved to {scores_output_path}")
            
            # Step 5: 生成热力图
            dpg.set_value("status_text", "[5/7] 正在生成热力图...")
            self._generate_heatmap(str(refined_json_path), str(analysis_output_path))
            
            # Step 6: 生成桌面功能分区
            dpg.set_value("status_text", "[6/7] 正在生成桌面功能分区...")
            self._generate_zone_clustering(str(refined_json_path))
            
            # Step 7: LLM 建议
            dpg.set_value("status_text", "[7/7] 正在生成 AI 建议...")
            suggestion = generate_suggestion()
            self.llm_suggestion = suggestion
            dpg.set_value("llm_suggestion_text", suggestion)
            
            # 保存 AI 建议到文件
            suggestion_output_path = PROJECT_ROOT / "analysis_output/ai_suggestion.txt"
            with open(suggestion_output_path, 'w', encoding='utf-8') as f:
                f.write(suggestion)
            print(f"AI suggestion saved to {suggestion_output_path}")
            
            # 解析 TODO 清单
            todos = self._parse_and_display_todos(suggestion)
            
            # 保存 TODO 清单到文件
            todos_output_path = PROJECT_ROOT / "analysis_output/todos.json"
            with open(todos_output_path, 'w', encoding='utf-8') as f:
                json.dump(todos, f, ensure_ascii=False, indent=2)
            print(f"TODO list saved to {todos_output_path}")
            
            # 完成
            dpg.set_value("status_text", "分析完成！")
            dpg.configure_item("status_text", color=self.colors["success"])
            
        except Exception as e:
            dpg.set_value("status_text", f"分析失败: {str(e)}")
            dpg.configure_item("status_text", color=self.colors["danger"])
            import traceback
            traceback.print_exc()
        
        finally:
            self.is_processing = False
            dpg.configure_item("btn_analyze", enabled=True)
    
    def _parse_and_display_todos(self, suggestion):
        """从 LLM 建议中解析 TODO 清单，返回 TODO 列表"""
        # 清除旧的 TODO
        if dpg.does_item_exist("todo_placeholder"):
            dpg.delete_item("todo_placeholder")
        
        # 清除旧的 TODO 项
        for child in dpg.get_item_children("todo_container", 1) or []:
            dpg.delete_item(child)
        
        # 简单解析：查找包含【高优先级】【中优先级】【低优先级】的行
        todos = []
        lines = suggestion.split('\n')
        
        for line in lines:
            line = line.strip()
            if '【高优先级】' in line or '【中优先级】' in line or '【低优先级】' in line:
                # 提取优先级和内容
                if '【高优先级】' in line:
                    priority = "高优先级"
                    p_color = self.colors["danger"]
                    content = line.replace('【高优先级】', '').strip()
                elif '【中优先级】' in line:
                    priority = "中优先级"
                    p_color = self.colors["warning"]
                    content = line.replace('【中优先级】', '').strip()
                else:
                    priority = "低优先级"
                    p_color = self.colors["success"]
                    content = line.replace('【低优先级】', '').strip()
                
                # 清理内容
                content = content.lstrip('0123456789. 、：:')
                if content:
                    todos.append({
                        "priority": priority,
                        "content": content,
                        "completed": False
                    })
        
        # 如果没有找到标准格式，尝试查找数字列表
        if not todos:
            for line in lines:
                line = line.strip()
                if line and line[0].isdigit() and ('。' in line or '，' in line):
                    todos.append({
                        "priority": "一般",
                        "content": line,
                        "completed": False
                    })
        
        # 显示 TODO
        if todos:
            for i, todo in enumerate(todos[:10]):  # 最多显示 10 条
                priority = todo["priority"]
                content = todo["content"]
                
                # 根据优先级选择颜色
                if priority == "高优先级":
                    p_color = self.colors["danger"]
                elif priority == "中优先级":
                    p_color = self.colors["warning"]
                elif priority == "低优先级":
                    p_color = self.colors["success"]
                else:
                    p_color = self.colors["primary"]
                
                with dpg.group(horizontal=True, parent="todo_container"):
                    dpg.add_checkbox(tag=f"todo_check_{i}")
                    dpg.add_text(f"[{priority}]", color=p_color)
                    dpg.add_text(content[:80] + ("..." if len(content) > 80 else ""), 
                                wrap=280, color=self.colors["text_main"])
        else:
            dpg.add_text("未能解析出具体 TODO 项，请查看 AI 建议详情", 
                        parent="todo_container", color=self.colors["warning"])
        
        return todos
    
    def _generate_heatmap(self, refined_json_path, analysis_json_path):
        """生成混乱热力图"""
        try:
            from utils.heatmap import MessinessHeatmapGenerator
            
            generator = MessinessHeatmapGenerator()
            result = generator.generate_heatmap(
                refined_json_path, 
                analysis_json_path,
                self.current_image_path
            )
            
            # 保存热力图数据供 AR 使用
            self.heatmap_data = result
            
            # 获取叠加图
            if 'overlay' in result:
                overlay_bgr = result['overlay']
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                display_img = self._resize_for_display(overlay_rgb)
                self._update_texture("texture_heatmap", display_img)
                
                # 更新热区信息
                hot_regions = result.get('hot_regions', [])
                if hot_regions:
                    info_text = f"检测到 {len(hot_regions)} 个高混乱区域:\n"
                    for i, region in enumerate(hot_regions[:3]):
                        items_str = ', '.join(region.get('contained_items', [])[:3])
                        problem_names = {
                            'hygiene_risk': '卫生风险',
                            'overcrowded': '物品过密',
                            'scattered': '物品分散',
                            'cable_mess': '线缆杂乱',
                            'general_mess': '一般杂乱'
                        }
                        problem = problem_names.get(region.get('problem_type', ''), '杂乱')
                        info_text += f"  {i+1}. {problem} (热度:{region.get('avg_heat', 0):.2f}) - {items_str}\n"
                    dpg.set_value("heatmap_info", info_text)
                    dpg.set_value("heatmap_hint", f"混乱热力图 - 红色区域需要整理")
                else:
                    dpg.set_value("heatmap_hint", "热力图生成完成 - 桌面较为整洁")
                    dpg.set_value("heatmap_info", "")
            else:
                dpg.set_value("heatmap_hint", "热力图生成完成（无叠加图）")
                
            # 保存热力图
            heatmap_output_dir = PROJECT_ROOT / "analysis_output/heatmap"
            heatmap_output_dir.mkdir(parents=True, exist_ok=True)
            
            if 'heatmap_colored' in result:
                cv2.imwrite(str(heatmap_output_dir / 'heatmap_colored.png'), result['heatmap_colored'])
            if 'overlay' in result:
                cv2.imwrite(str(heatmap_output_dir / 'heatmap_overlay.png'), result['overlay'])
            
            # 保存热区 JSON
            hot_regions_path = heatmap_output_dir / 'hot_regions.json'
            with open(hot_regions_path, 'w', encoding='utf-8') as f:
                json.dump(result.get('hot_regions', []), f, ensure_ascii=False, indent=2)
                
            print(f"Heatmap saved to {heatmap_output_dir}")
            
        except Exception as e:
            print(f"生成热力图失败: {e}")
            import traceback
            traceback.print_exc()
            dpg.set_value("heatmap_hint", f"热力图生成失败: {str(e)[:50]}")
            self.heatmap_data = None
    
    def _generate_zone_clustering(self, refined_json_path):
        """生成桌面功能分区（基于 DBSCAN + K-Means 聚类）"""
        try:
            from utils.clustering import ZoneClusteringEngine, ZoneVisualizer, ZoneClusteringConfig
            
            # 加载检测结果
            with open(refined_json_path, 'r', encoding='utf-8') as f:
                refined_data = json.load(f)
            
            # 获取图像尺寸
            H, W = self.original_image.shape[:2]
            
            # 执行聚类
            config = ZoneClusteringConfig()
            engine = ZoneClusteringEngine(config)
            result = engine.cluster(refined_data['objects'], (W, H))
            
            # 保存聚类结果供后续使用
            self.clustering_result = result
            
            # 可视化
            font_path = PROJECT_ROOT / "assets" / "fonts" / "SIMYOU.TTF"
            visualizer = ZoneVisualizer(
                font_path=str(font_path) if font_path.exists() else None
            )
            rendered = visualizer.render(self.original_image.copy(), result.zones)
            
            # 显示结果
            rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
            display_img = self._resize_for_display(rendered_rgb)
            self._update_texture("texture_ar", display_img)
            
            # 更新分区信息
            zones = result.zones
            
            if zones:
                info_text = f"识别到 {len(zones)} 个功能区域:\n"
                for zone in zones:
                    item_count = len(zone.objects)
                    items_preview = ', '.join([obj['class'] for obj in zone.objects[:3]])
                    if len(zone.objects) > 3:
                        items_preview += '...'
                    info_text += f"  {zone.icon} {zone.zone_type} ({item_count}件): {items_preview}\n"
                
                # 显示聚类参数信息
                info_text += f"\n聚类参数: eps={result.params.get('eps', 0):.3f}, "
                info_text += f"位置权重={config.position_weight:.0%}"
                
                dpg.set_value("ar_info", info_text)
                dpg.set_value("ar_hint", "桌面功能分区 - 基于位置和语义聚类")
            else:
                dpg.set_value("ar_hint", "分区完成 - 未检测到明显的功能区域")
                dpg.set_value("ar_info", "")
            
            # 保存结果
            clustering_output_dir = PROJECT_ROOT / "analysis_output/clustering"
            clustering_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存可视化图片
            cv2.imwrite(str(clustering_output_dir / 'zone_clustering.png'), rendered)
            
            # 保存 JSON 数据
            def to_python_type(obj):
                """将 numpy 类型转换为 Python 原生类型"""
                import numpy as np
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: to_python_type(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [to_python_type(x) for x in obj]
                return obj
            
            clustering_data = {
                'zones': [
                    {
                        'id': int(z.id),
                        'zone_type': z.zone_type,
                        'icon': z.icon,
                        'center': [float(x) for x in z.center],
                        'object_count': len(z.objects),
                        'objects': [obj.get('name', obj.get('class', '')) for obj in z.objects],
                        'category_distribution': to_python_type(z.category_distribution),
                        'is_singleton': z.is_singleton,
                        'source': z.source
                    }
                    for z in zones
                ],
                'params': to_python_type(result.params)
            }
            
            with open(clustering_output_dir / 'zones.json', 'w', encoding='utf-8') as f:
                json.dump(clustering_data, f, ensure_ascii=False, indent=2)
            
            print(f"Zone clustering saved to {clustering_output_dir}")
            
        except Exception as e:
            print(f"生成桌面功能分区失败: {e}")
            import traceback
            traceback.print_exc()
            dpg.set_value("ar_hint", f"分区生成失败: {str(e)[:50]}")
    
    def run(self):
        """运行主循环"""
        dpg.set_primary_window("main_window", True)
        dpg.start_dearpygui()
        dpg.destroy_context()


def run_app():
    """启动应用的便捷函数"""
    app = DeskScopeApp()
    app.setup()
    app.run()


if __name__ == "__main__":
    run_app()
