# calculate_valid_combinations.py
# -*- coding: utf-8 -*-
"""
这个脚本用于计算在应用所有约束之后，
archetypes.py 理论上可以生成的有效角色画像组合的总数。
"""
import yaml
from pathlib import Path
from itertools import product
import os # Import os for path manipulation

# --- 从 archetypes.py 复制的常量和函数 ---
# 确保路径相对于此脚本的父目录 (项目根目录)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TOPIC_SUBTOPIC_MAP = {
    "学业与成长": ["学业压力", "考试焦虑", "成绩困扰", "学习方法", "学习动力", "厌学情绪", "注意力不集中", "作业问题", "升学规划", "时间管理"],
    "人际与社交": ["同学关系", "交友困惑", "校园霸凌", "被孤立", "沟通技巧", "师生关系", "网络交友", "人际边界", "信任问题", "社交恐惧"],
    "家庭与亲子": ["亲子沟通", "家庭压力", "父母期望", "家庭矛盾", "家庭暴力", "缺乏理解", "隐私空间"],
    "情绪与内在状态": ["创伤事件复述与应激", "抑郁低落与绝望", "焦虑烦躁与失控", "孤独感与无价值感", "自伤与自杀念头", "冲动与愤怒管理", "网络/手机成瘾", "躯体化症状"],
    "自我认知与价值观": ["自信心与自我价值", "外貌焦虑", "性格困扰", "理想与现实冲突", "个人成长哲学", "生命意义与道德思辨", "青春期萌动"],
    "互动模式与对话探索": ["AI身份试探", "测试系统边界", "幽默/游戏式互动", "哲学式辩论", "碎片化/回避式表达", "寻求角色扮演"],
    "家长视角与求助": ["孩子社交问题", "孩子情绪行为异常", "孩子学业求助", "亲子沟通方法", "如何引导与帮助孩子", "担忧孩子网络使用"]
}
OCCUPATIONS_LIST = ["小学生", "初中生", "高中生", "老师", "家长"]

def load_yaml_file(file_path: Path, required_keys: list):
    # Adjust file_path to be relative to the project root
    full_path = PROJECT_ROOT / file_path
    if not full_path.exists():
        raise FileNotFoundError(f"Required YAML file not found: {full_path}")
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            raise ValueError(f"Invalid format in {full_path}. Expected a list of dictionaries.")
        for i, item in enumerate(data):
            if not all(key in item for key in required_keys):
                raise ValueError(f"Item at index {i} in '{full_path}' is missing one of required keys: {required_keys}.")
            item["constraints"] = item.get("constraints", {}) # Ensure constraints key exists
        print(f"Successfully loaded {len(data)} items from '{full_path}'.")
        return data
    except (yaml.YAMLError, ValueError) as e:
        print(f"Error processing YAML file '{full_path}': {e}")
        raise

def is_combination_valid(occupation: str, topic: str, config_item: dict) -> bool:
    """通用约束检查函数。"""
    constraints = config_item.get("constraints", {})
    if not isinstance(constraints, dict): return True # No constraints means valid
    if (allowed := constraints.get("allowed_occupations")) and occupation not in allowed: return False
    if (forbidden := constraints.get("forbidden_occupations")) and occupation in forbidden: return False
    if (allowed := constraints.get("allowed_topics")) and topic not in allowed: return False
    if (forbidden := constraints.get("forbidden_topics")) and topic in forbidden: return False
    return True
# --- 结束复制的部分 ---

def calculate_all_valid_combinations():
    """
    计算所有有效的画像组合数量。
    """
    try:
        core_drives = load_yaml_file(Path("core_drives.yaml"), ["name", "description"])
        reaction_patterns = load_yaml_file(Path("reaction_patterns.yaml"), ["name", "description"])
    except Exception as e:
        print(f"Error loading YAML files: {e}")
        return 0

    valid_combination_count = 0
    
    # 构建所有主题 -> 子主题的扁平列表，用于迭代
    all_topic_subtopic_pairs = []
    for topic, subtopics_list in TOPIC_SUBTOPIC_MAP.items():
        for subtopic in subtopics_list:
            all_topic_subtopic_pairs.append({"topic": topic, "subtopic": subtopic})

    # 遍历所有可能的组合
    # product creates an iterator of Cartesian product of input iterables.
    # OCCUPATIONS_LIST
    # all_topic_subtopic_pairs (each item is a dict with "topic" and "subtopic")
    # core_drives
    # reaction_patterns
    
    print(f"Starting combination check...")
    print(f"Occupations: {len(OCCUPATIONS_LIST)}")
    print(f"Topic-Subtopic Pairs: {len(all_topic_subtopic_pairs)}")
    print(f"Core Drives: {len(core_drives)}")
    print(f"Reaction Patterns: {len(reaction_patterns)}")

    total_possible_unconstrained = len(OCCUPATIONS_LIST) * len(all_topic_subtopic_pairs) * len(core_drives) * len(reaction_patterns)
    print(f"Total possible unconstrained combinations (Occupation x SubTopic x Drive x Reaction): {total_possible_unconstrained}")

    processed_combinations = 0
    for occ in OCCUPATIONS_LIST:
        for ts_pair in all_topic_subtopic_pairs:
            topic = ts_pair["topic"]
            # subtopic = ts_pair["subtopic"] # subtopic is part of the unique combo, but not used in validation directly
            for drive in core_drives:
                for rp in reaction_patterns:
                    processed_combinations += 1
                    if processed_combinations % 1000 == 0:
                        print(f"Processed {processed_combinations}/{total_possible_unconstrained} potential combinations...")
                        
                    # 约束检查
                    # 1. 核心驱动对于 (职业, 主题) 是否有效
                    drive_valid = is_combination_valid(occ, topic, drive)
                    # 2. 反应模式对于 (职业, 主题) 是否有效
                    rp_valid = is_combination_valid(occ, topic, rp)

                    if drive_valid and rp_valid:
                        valid_combination_count += 1
    
    print(f"Finished processing. Total processed combinations: {processed_combinations}")
    return valid_combination_count

if __name__ == "__main__":
    print("Calculating the total number of valid unique persona combinations...")
    count = calculate_all_valid_combinations()
    print(f"\n理论上可生成的有效角色画像组合总数为: {count}") 