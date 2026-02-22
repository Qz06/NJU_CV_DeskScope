# pipeline/main.py
import json
from pathlib import Path
from PIL import Image
import argparse

from .run_yolo import run_yolo
from .run_sam3 import run_sam3
from .refine_geometry import analyze_mask

# ---- SAM3 初始化（示意）----
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import torch

# ---- Analysis & Suggest ----
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.analysis import analyse_from_json
from utils.suggest import generate_suggestion


def _parse_todos_from_suggestion(suggestion):
    """从 AI 建议中解析 TODO 清单"""
    todos = []
    lines = suggestion.split('\n')
    
    for line in lines:
        line = line.strip()
        if '【高优先级】' in line or '【中优先级】' in line or '【低优先级】' in line:
            # 提取优先级和内容
            if '【高优先级】' in line:
                priority = "高优先级"
                content = line.replace('【高优先级】', '').strip()
            elif '【中优先级】' in line:
                priority = "中优先级"
                content = line.replace('【中优先级】', '').strip()
            else:
                priority = "低优先级"
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
    
    return todos


def run_pipeline(image_path=None):
    """运行完整的分析流程（命令行模式）"""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    if image_path is None:
        image_path = PROJECT_ROOT / "data/images/desk3.jpg"
    else:
        image_path = Path(image_path)

    yolo_model_path = PROJECT_ROOT / "models/yolo/best.pt"

    # 1. YOLO
    print("=" * 50)
    print("[Step 1] Running YOLO detection...")
    result = run_yolo(str(image_path), str(yolo_model_path))

    # 2. SAM3 初始化
    print("=" * 50)
    print("[Step 2] Initializing SAM3 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam3_model = build_sam3_image_model(device=device)
    sam3_processor = Sam3Processor(sam3_model)

    image = Image.open(image_path).convert("RGB")

    # 3. 对所有检测到的物体做 SAM3 精细分割
    print("=" * 50)
    print("[Step 3] Running SAM3 refinement for ALL detected objects...")
    total_objects = len(result["objects"])
    for idx, obj in enumerate(result["objects"], 1):
        print(f"  Processing object {idx}/{total_objects}: {obj['class_name']}")
        
        mask = run_sam3(
            sam3_processor,
            image,
            obj["bbox_xyxy"]
        )

        geo = analyze_mask(mask, image_path=str(image_path))
        obj["sam3"] = geo

    # 4. 保存原始检测结果 JSON
    print("=" * 50)
    print("[Step 4] Saving refined detection result...")
    refined_json_path = PROJECT_ROOT / "data/refined_json/desk19_refined.json"
    refined_json_path.parent.mkdir(exist_ok=True, parents=True)

    with open(refined_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Refined JSON saved to: {refined_json_path}")

    # 5. 调用 Analysis 生成分析结果
    print("=" * 50)
    print("[Step 5] Running analysis on detection result...")
    analysis_output_path = PROJECT_ROOT / "analysis_output/last_result.json"
    analysis_result = analyse_from_json(str(refined_json_path), str(analysis_output_path))
    
    print(f"\n[Analysis Scores]")
    print(f"  - Tidy Score:    {analysis_result['tidy']:.2f}")
    print(f"  - Learning Score: {analysis_result['learning']:.2f}")
    print(f"  - Hygiene Score:  {analysis_result['hygiene']:.2f}")
    
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
    print(f"Scores saved to: {scores_output_path}")

    # 6. 调用 LLM 生成建议
    print("=" * 50)
    print("[Step 6] Generating suggestions via LLM...")
    suggestion = generate_suggestion()
    
    print("\n" + "=" * 50)
    print("【LLM 生成的桌面分析与建议】")
    print("=" * 50)
    print(suggestion)

    # 7. 保存 AI 建议到文件
    suggestion_output_path = PROJECT_ROOT / "analysis_output/ai_suggestion.txt"
    with open(suggestion_output_path, "w", encoding="utf-8") as f:
        f.write(suggestion)
    print(f"\nAI suggestion saved to: {suggestion_output_path}")
    
    # 8. 解析并保存 TODO 清单
    print("=" * 50)
    print("[Step 7] Parsing TODO list...")
    todos = _parse_todos_from_suggestion(suggestion)
    
    if todos:
        print(f"\n[TODO List] ({len(todos)} items)")
        for i, todo in enumerate(todos, 1):
            print(f"  {i}. [{todo['priority']}] {todo['content'][:60]}...")
    
    todos_output_path = PROJECT_ROOT / "analysis_output/todos.json"
    with open(todos_output_path, 'w', encoding='utf-8') as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)
    print(f"TODO list saved to: {todos_output_path}")

    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("=" * 50)

    return {
        "refined_json": str(refined_json_path),
        "analysis_json": str(analysis_output_path),
        "suggestion": suggestion,
        "scores": analysis_result
    }


def run_gui():
    """启动 GUI 模式"""
    from gui.app import run_app
    run_app()


def main():
    """主入口：支持命令行参数选择模式"""
    parser = argparse.ArgumentParser(description="DeskScope - 桌面分析工具")
    parser.add_argument("--gui", action="store_true", help="启动 GUI 模式")
    parser.add_argument("--image", type=str, help="指定图片路径（命令行模式）")
    
    args = parser.parse_args()
    
    if args.gui:
        run_gui()
    else:
        run_pipeline(args.image)


if __name__ == "__main__":
    main()
