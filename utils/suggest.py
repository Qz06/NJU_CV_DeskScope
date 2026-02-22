import os
import json
from openai import OpenAI

client = OpenAI(api_key='YOUR_API_KEY',
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", )


def load_analysis():  # 读取 analysis.py 输出的 last_result.json
    path = "analysis_output/last_result.json"
    if not os.path.exists(path):
        raise FileNotFoundError("analysis_output/last_result.json 不存在，请先运行 analysis()")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_hot_regions():
    path = "analysis_output/heatmap/hot_regions.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def build_prompt(data, hot_regions):
    meta = data["metadata"]
    objects = data["objects_clean"]
    scores = data["scores"]
    stats = data["statistics"]
    relations = data["spatial_relations"]
    classes = data['semantic_groups']
    summary = data['situation_summary']

    # 格式化 hot_regions 信息
    hot_regions_text = ""
    if hot_regions:
        for i, region in enumerate(hot_regions):
            items = ", ".join(region.get("contained_items", []))
            heat = region.get("avg_heat", 0)
            problem = region.get("problem_type", "unknown")
            hot_regions_text += f"- 区域 {i+1}: 问题类型={problem}, 平均热度={heat:.2f}, 包含物品=[{items}]\n"
    else:
        hot_regions_text = "无显著高热度拥挤区域。"

    prompt = f"""
    你是一位细致、体贴、善于观察生活细节的桌面整理与生活方式顾问。  
以下是系统从用户桌面图像中分析得到的结构化信息。  
所有信息均来自检测模型，请你在此基础上进行严谨、贴心、口语化、真实可信的分析和建议。

请严格按照以下结构输出你的回答：

================【输入数据】================

【元信息（metadata）】
{meta}

【桌面物体列表（objects_clean）】
（每个物体含：名字、类别、是否学习相关 / 杂乱相关 / 不卫生相关 / 特殊状态相关）
{objects}

【评分（scores）】
{scores}

【统计信息（statistics）】
（覆盖率、杂乱率、不卫生率、总数量等）
{stats}

【空间关系（spatial_relations）】
（近邻对、重叠率、区域拥挤情况、物体聚集中心）
{relations}

【热力图高风险区域（hot_regions）】
（基于热力图识别出的过度拥挤或混乱区域）
{hot_regions_text}

【物品类别（semantic_groups）】
（可能出现的物品分类）
{classes}

【情况小结（situation_summary）】
（桌面情况的一句话概括）
{summary}

================【任务要求】================

你必须按照下面4个部分输出内容：

------------------------------------------------------------
【第一部分：桌面总体分析】
------------------------------------------------------------
1. 先给一个整体性的评价（语言口语化，例如“你的桌面整体来说偏乱，但还没到不可收拾的程度”）。
2. 然后分点说明：
   - 物品数量情况  
   - 杂乱情况：结合 messy_rate、overlap_ratio、position_distribution_variance  
   - 覆盖率：结合 coverage_rate  
   - 卫生情况：dirty_rate + 不卫生物品列表  
   - 学习氛围：基于 learning_rate 和学习相关物体  
   - 空间结构：根据 crowded_regions、crowded_center 以及 hot_regions 做局部分析

要求分析要细致、有洞察力，不要泛泛而谈。

------------------------------------------------------------
【第二部分：桌面风险与整理建议】
------------------------------------------------------------
分成两个小节：

### （A）当前桌面存在的潜在风险（非常重要）
需要你结合【空间关系 + objects_clean + hot_regions】进行具体分析，包括：
- 是否有水杯/饮料靠近电子设备（特别是笔记本、键盘、插座）  
- 某个区域是否特别拥挤（参考 hot_regions 中的具体物品组合）
- 哪些物体重叠严重或堆叠风险高  
- 是否有容易掉落、绊线、压到的东西  
- 是否有垃圾长时间遗留（如 trash、packaging）  

要求非常具体，例如：
“笔记本右侧有一团纸巾和零食包装，容易在写字时滑落”
“区域 1 中堆积了大量线缆和杂物，可能存在散热或缠绕风险”

### （B）可执行的整理建议（按紧急程度排序）
请你给出从“最值得优先处理”到“可以稍后处理”的排序，例如：

1. 【高优先级】先把 XX 移开，以免 XX  
2. 【中优先级】建议把 YY 收纳到一个区域  
3. 【低优先级】如果有空，可以整理 ZZ 区域  

建议必须切实可行、简单、直接。

------------------------------------------------------------
【第三部分：用户状态推测】
------------------------------------------------------------
你需要根据：

### 1. metadata 中的时间（用于判断季节）,只精确到月, 不要精确到日
例如：
- 12 月 → 冬季 → 注意保暖  
- 6 月 → 夏季 → 注意补水  
- 深夜时间 → 可能熬夜  

### 2. objects_clean 中 is_special_related 的物体
这些物体强烈反映用户状态，你必须重点推断，例如：

- usingbook / scatteredpaper / notebook / scatteredpen  
  → 用户可能在赶作业/备考，有压力或情绪紧张  
- fruit / bread  
  → 可能有易腐食物忘吃  
- toy / gamecontroller / comic  
  → 用户处于放松、娱乐状态  
- medicine / vitamin  
  → 用户可能在康复期或身体不适  
- studentcard  
  → 用户可能刚回到宿舍 / 或明天需要去学校，提醒用户记得带校园卡 

你需要把这些特征组织成一段“非常贴近生活、自然但有逻辑”的用户状态推测。

注意：推测要谨慎但温柔，不要太绝对，比如多用“可能、或许、看起来像是……”。

------------------------------------------------------------
【第四部分：对用户状态的贴心建议】
------------------------------------------------------------
根据第三部分的推测，给用户一些真诚、温和、实际的建议。

例如如果用户在备考：
- “建议你今晚早点休息，明天效率会更高”

如果桌上有水果：
- “水果容易坏，记得今天吃掉，不然明天就浪费了”

如果出现 medicine：
- “最近身体可能不太舒服，别太累，多喝水”

要求语言要像朋友一样，而不是机器命令式。

------------------------------------------------------------
【风格要求】
- 全中文，不要出现英文类名或英文变量名  
- 内容必须基于输入数据，不要胡编不存在的物体  
- 语气温柔、口语化、贴心、有生活气息  
- 分点叙述，逻辑清晰  
- 输出不能太短，需要有深度  

------------------------------------------------------------

请根据上述要求开始你的输出。

    """

    return prompt


def generate_suggestion():
    data = load_analysis()
    hot_regions = load_hot_regions()
    prompt = build_prompt(data, hot_regions)

    response = client.chat.completions.create(
        model='qwen3-max',
        messages=[
            {"role": "system", "content": "你是一个严格、专业、不会胡编的桌面整理与学习效率专家。"},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
