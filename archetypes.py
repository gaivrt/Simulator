# archetypes_v3_simplified.py
# -*- coding: utf-8 -*-
"""
本脚本是角色画像生成器的第三版（精简版），专注于通过"策略性组合"方法，
自动化生成高度多样化、真实且具有内在逻辑一致性的K12心理咨询模拟用户角色卡（Personas）。

核心生成哲学：三维正交与策略增强
本版本采用"核心驱动 x 困扰情境 x 反应模式"三维正交组合的生成逻辑，以最大化多样性。
- 核心驱动 (Core Drive): 定义角色的"灵魂"，即其行为的内在动机。
- 困扰情境 (Situation): 定义角色面临的外部问题。
- 反应模式 (Reaction Pattern): 定义角色在压力下的外在行为表现。

同时，通过"策略性增强"配置，可以强制生成特定数量的稀有但关键的"边缘案例"，
以确保训练出的AI模型具有足够的鲁棒性来应对真实世界的复杂情况。

功能亮点:
1.  **模块化配置**: 所有生成维度（核心驱动、反应模式）均通过外部YAML文件配置，易于扩展和维护。
2.  **策略性组合生成**: 以组合爆炸（Combinatorial）为基础，极大化生成角色的多样性，并通过策略性配置（Strategic Configs）来确保关键案例的覆盖。
3.  **深度角色构建**: 指导LLM为每个角色创造独特的背景故事（解释核心驱动的来源），确保每个角色都有"历史"和"灵魂"。
4.  **强大的LLM兼容性与健壮性**: 支持OpenAI和Gemini，并包含连接性检查、并行处理和详细的错误处理。

如何使用:
1.  确保 `core_drives.yaml`, `reaction_patterns.yaml`, `system_prompt.txt` 文件与本脚本在同一目录下。
2.  在 `.env` 文件中配置您的LLM API密钥。
3.  在脚本末尾的 `if __name__ == "__main__":` 部分，配置 `TOTAL_PROFILES_TO_GENERATE` 和 `STRATEGIC_CONFIGS`。
4.  运行脚本: `python archetypes_v3_simplified.py`
5.  生成的角色卡将保存在 `generated_character_profiles_v3.json` 文件中。
"""
import json
import os
import yaml
import threading
import sys
import random
import uuid
import re  # 引入正则表达式库用于清洗
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import traceback # Ensure traceback is imported

# --- LLM Specific Imports ---
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai library not found. OpenAI provider will not work.")

try:
     import google.generativeai as genai
     GEMINI_AVAILABLE = True
except ImportError:
     GEMINI_AVAILABLE = False
     print("Warning: google-generativeai library not found. Gemini provider will not work.")
# ---------------------------

load_dotenv()

# --- LLM Config ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4-turbo")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_ARCHETYPE_MODEL", "gemini-1.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 1.2))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", 0.95))
OPENAI_FREQUENCY_PENALTY = float(os.getenv("OPENAI_FREQUENCY_PENALTY", 0.3))
OPENAI_PRESENCE_PENALTY = float(os.getenv("OPENAI_PRESENCE_PENALTY", 0.3))
# ---------------------------------

# --- 预设的理想字段 ---
DESIRED_FIELDS = [
    "Gender", "Age", "Occupation", "Topic", "Subtopic", "Situation",
    "Event Time", "Event Location", "Event Participants", "Event Description",
    "Emotional Experience Words", "Coped Strategies and Effects", "Goals and Expectations",
    "persona_id"  # persona_id 是在脚本中生成的，也应保留
]
# --------------------

# --- 全局配置与锁 ---
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))
shared_list_lock = threading.Lock()
# --------------------

# --- 资产加载 ---
# K12
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
"""
# 高校
TOPIC_SUBTOPIC_MAP = {
    "学业与成长": ["学业压力", "考试焦虑", "成绩困扰", "学习方法", "学习习惯", "厌学情绪", "升学规划", "作业问题", "时间管理"],
    "人际与社交": ["同学关系", "交友困惑", "校园霸凌", "人际边界", "师生关系", "网络交友"],
    "恋爱与情感": ["择偶标准", "亲密关系", "恋爱困扰", "分手失恋", "情感表达", "恋爱观念", "暗恋/单恋"],
    "职业与规划": ["就业焦虑", "职业发展", "实习困扰", "职场人际", "工作压力", "职业规划"],
    "家庭与亲子": ["亲子沟通", "家庭压力", "家庭关系", "家庭暴力"],
    "情绪与行为": ["行为习惯", "焦虑情绪", "抑郁与低落", "孤独感", "愤怒与烦躁", "自伤与自杀念头", "网络/手机依赖", "创伤应激"],
    "自我认知": ["自我价值感", "外貌焦虑", "生命意义探索", "性格困扰", "自信心"]
}
OCCUPATIONS_LIST = ["小学生", "初中生", "高中生", "大学生", "老师", "家长", "职场人士"]
"""

def load_yaml_file(file_path: Path, required_keys: list):
    if not file_path.exists():
        raise FileNotFoundError(f"Required YAML file not found: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            raise ValueError(f"Invalid format in {file_path}. Expected a list of dictionaries.")
        for i, item in enumerate(data):
            if not all(key in item for key in required_keys):
                raise ValueError(f"Item at index {i} in '{file_path}' is missing one of required keys: {required_keys}.")
            item["constraints"] = item.get("constraints", {})
        print(f"Successfully loaded {len(data)} items from '{file_path}'.")
        return data
    except (yaml.YAMLError, ValueError) as e:
        print(f"Error processing YAML file '{file_path}': {e}")
        raise

def load_text_file(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Required text file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

try:
    CORE_DRIVES = load_yaml_file(Path("core_drives.yaml"), ["name", "description"])
    # CORE_DRIVES = load_yaml_file(Path("high_core_drives.yaml"), ["name", "description"])
    REACTION_PATTERNS = load_yaml_file(Path("reaction_patterns.yaml"), ["name", "description"])
    # REACTION_PATTERNS = load_yaml_file(Path("high_reaction_patterns.yaml"), ["name", "description"])
    SYSTEM_PROMPT_TEMPLATE = load_text_file(Path("system_prompt.txt"))
    # SYSTEM_PROMPT_TEMPLATE = load_text_file(Path("high_system_prompt.txt"))
except (FileNotFoundError, ValueError) as e:
    print(f"Fatal error loading configuration files: {e}. Exiting.")
    sys.exit(1)

# --- 约束校验函数 ---
def is_combination_valid(occupation: str, topic: str, config_item: dict) -> bool:
    """通用约束检查函数。"""
    constraints = config_item.get("constraints", {})
    if not isinstance(constraints, dict): return True
    if (allowed := constraints.get("allowed_occupations")) and occupation not in allowed: return False
    if (forbidden := constraints.get("forbidden_occupations")) and occupation in forbidden: return False
    if (allowed := constraints.get("allowed_topics")) and topic not in allowed: return False
    if (forbidden := constraints.get("forbidden_topics")) and topic in forbidden: return False
    return True

# --- LLM Client & Helper ---
openai_client = None
if LLM_PROVIDER == "openai":
     if OPENAI_AVAILABLE and OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None)
     else:
         print("Error: OpenAI selected but library or API Key/Base URL missing.")
elif LLM_PROVIDER == "gemini":
     if GEMINI_AVAILABLE and GEMINI_API_KEY:
         genai.configure(api_key=GEMINI_API_KEY)
     else:
          print("Error: Gemini selected but library or API Key missing.")
else:
    print(f"Warning: Unsupported LLM_PROVIDER: {LLM_PROVIDER}. LLM calls will fail.")

def check_llm_connectivity():
    print("\n--- Checking LLM Connectivity ---")
    if LLM_PROVIDER == "openai":
        if openai_client:
            try:
                openai_client.models.list()
                print("OpenAI connection successful.")
                return True
            except Exception as e:
                print(f"Error: OpenAI connection failed. Details: {e}")
                return False
        else: return False
    elif LLM_PROVIDER == "gemini":
        if GEMINI_AVAILABLE and GEMINI_API_KEY and GEMINI_MODEL_NAME:
            try:
                model = genai.GenerativeModel(GEMINI_MODEL_NAME)
                model.count_tokens("test connectivity")
                print("Gemini connection successful.")
                return True
            except Exception as e:
                print(f"Error: Gemini connection failed. Details: {e}")
                return False
        else: return False
    return False

def call_llm_api(system_prompt_formatted, model_override=None):
    """
    调用LLM API并处理返回的JSON。
    新增了对返回结果的清洗，以处理潜在的Markdown格式。
    """
    def _clean_json_string(s: str) -> str:
        """
        从可能包含Markdown代码块的字符串中提取纯净的JSON部分。
        """
        # 使用正则表达式查找被 ```json ... ``` 包裹的内容
        match = re.search(r"```json\s*([\s\S]*?)\s*```", s)
        if match:
            # 如果找到，返回第一个捕获组的内容，并去除首尾空白
            return match.group(1).strip()
        
        # 如果没有找到Markdown块，尝试直接去除首尾的空白和换行符
        # 这可以处理一些简单的不规范格式，比如开头或结尾有换行
        return s.strip()

    raw_output = "N/A"
    try:
        if LLM_PROVIDER == "openai" and openai_client:
            model = model_override or OPENAI_DEFAULT_MODEL
            completion = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt_formatted}],
                response_format={"type": "json_object"},
                temperature=LLM_TEMPERATURE, top_p=LLM_TOP_P,
                frequency_penalty=OPENAI_FREQUENCY_PENALTY,
                presence_penalty=OPENAI_PRESENCE_PENALTY
            )
            raw_output = completion.choices[0].message.content
            # OpenAI在指定json_object时通常返回纯净JSON，但清洗一下更安全
            cleaned_output = _clean_json_string(raw_output)
            return json.loads(cleaned_output)

        elif LLM_PROVIDER == "gemini" and GEMINI_AVAILABLE:
            model = genai.GenerativeModel(model_override or GEMINI_MODEL_NAME)
            response = model.generate_content(
                system_prompt_formatted,
                generation_config=genai.types.GenerationConfig(
                    temperature=LLM_TEMPERATURE, top_p=LLM_TOP_P,
                    response_mime_type="application/json",
                )
            )
            if response.candidates and response.candidates[0].content.parts:
                raw_output = response.candidates[0].content.parts[0].text
                # **关键修复：在解析前清洗Gemini的输出**
                cleaned_output = _clean_json_string(raw_output)
                return json.loads(cleaned_output)
            return None
        return None
    except json.JSONDecodeError as e:
        # 提供更详细的错误日志
        print(f"LLM API Error: Failed to decode JSON. Error: {e}")
        print(f"----- LLM Raw Output Start -----\n{raw_output}\n----- LLM Raw Output End -----")
        return None
    except Exception as e:
        print(f"LLM API Error: {e}")
        return None

# --- 核心生成逻辑 ---
def process_generation_task(task):
    """处理单个生成任务，调用LLM并返回结果。"""
    try:
        # DEBUG: Print task details
        # print(f"DEBUG: Starting process_generation_task with task: {task}")
        
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            target_core_drive_name=task["core_drive"]["name"],
            target_core_drive_description=task["core_drive"]["description"],
            target_topic=task["topic"],
            target_subtopic=task["subtopic"],
            target_reaction_pattern_name=task["reaction_pattern"]["name"],
            target_reaction_pattern_description=task["reaction_pattern"]["description"],
            target_occupation=task["occupation"]
        )
        
        # DEBUG: Print formatted prompt
        # print(f"DEBUG: System prompt formatted: {system_prompt[:200]}...") # Print first 200 chars
        
        profile_json = call_llm_api(system_prompt)
        
        # DEBUG: Print LLM response
        # if profile_json is not None:
        #     print(f"DEBUG: LLM call returned type: {type(profile_json)}, content: {str(profile_json)[:200]}...")
        # else:
        #     print("DEBUG: LLM call returned None")

        if profile_json and isinstance(profile_json, dict):
            profile_json['persona_id'] = str(uuid.uuid4())
            
            # --- 新增字段过滤功能 ---
            filtered_profile = {key: profile_json[key] for key in DESIRED_FIELDS if key in profile_json}
            # 检查是否有未包含在 DESIRED_FIELDS 中的原始 profile_json 字段 (可选的警告)
            # unexpected_fields = [key for key in profile_json if key not in DESIRED_FIELDS]
            # if unexpected_fields:
            #     print(f"  Warning: Profile for {task.get('core_drive', {}).get('name', 'UnknownDrive')} generated unexpected fields: {unexpected_fields}. These were removed.")
            return filtered_profile
            # --- 结束新增 ---
        else:
            print(f"  Warning: LLM call failed or returned invalid/non-dict data for a task (type: {type(profile_json)}). Skipping.")
            return None
    except Exception as e:
        print(f"  Error processing task (Exception Type: {type(e)}): {e}. Skipping.")
        print("  Traceback:")
        traceback.print_exc() # This will print the full stack trace
        return None

def generate_profiles(num_profiles: int, strategic_configs: list = []):
    """主生成函数，采用策略性组合生成模式。"""
    generation_plan = []
    remaining_profiles = num_profiles

    # 1. 应用策略性增强配置
    print("\n--- Applying Strategic Configurations ---")
    if strategic_configs:
        for config in strategic_configs:
            count = config.get("count", 0)
            if count <= 0: continue
            
            possible_drives = [cd for cd in CORE_DRIVES if not config.get("core_drive") or cd['name'] == config.get("core_drive")]
            possible_occupations = [occ for occ in OCCUPATIONS_LIST if not config.get("occupation") or occ == config.get("occupation")]
            possible_topics = [t for t in TOPIC_SUBTOPIC_MAP.keys() if not config.get("topic") or t == config.get("topic")]
            possible_rps = [rp for rp in REACTION_PATTERNS if not config.get("reaction_pattern") or rp['name'] == config.get("reaction_pattern")]
            
            if not all([possible_drives, possible_occupations, possible_topics, possible_rps]):
                print(f"Warning: A strategic config could not find matching items. Config: {config}")
                continue

            for _ in range(count):
                if remaining_profiles <= 0: break
                
                drive = random.choice(possible_drives)
                occupation = random.choice(possible_occupations)
                topic = random.choice(possible_topics)
                subtopic = random.choice(TOPIC_SUBTOPIC_MAP[topic])
                rp = random.choice(possible_rps)

                if is_combination_valid(occupation, topic, drive) and is_combination_valid(occupation, topic, rp):
                    generation_plan.append({
                        "occupation": occupation, "topic": topic, "subtopic": subtopic,
                        "core_drive": drive, "reaction_pattern": rp
                    })
                    remaining_profiles -= 1
    
    print(f"Strategically planned {num_profiles - remaining_profiles} profiles. {remaining_profiles} remaining for combinatorial generation.")

    # 2. 构建组合模式的生成计划 (为剩余数量)
    if remaining_profiles > 0:
        print(f"\n--- Building Plan for remaining {remaining_profiles} Profiles (Combinatorial Mode) ---")
        all_possible_combos = list(product(OCCUPATIONS_LIST, TOPIC_SUBTOPIC_MAP.items(), CORE_DRIVES, REACTION_PATTERNS))
        valid_combos = []
        for occ, (top, subs), drive, rp in all_possible_combos:
            if is_combination_valid(occ, top, drive) and is_combination_valid(occ, top, rp):
                for sub in subs:
                    valid_combos.append({
                        "occupation": occ, "topic": top, "subtopic": sub,
                        "core_drive": drive, "reaction_pattern": rp
                    })
        
        if not valid_combos:
            print("Warning: No valid combinations found in combinatorial mode. Cannot generate remaining profiles.")
        else:
            num_to_sample = min(remaining_profiles, len(valid_combos))
            generation_plan.extend(random.sample(valid_combos, num_to_sample))
            if num_to_sample < remaining_profiles:
                print(f"Warning: Only found {num_to_sample} unique valid combinations. Total generation will be less than requested.")

    if not generation_plan:
        print("Generation plan is empty. No profiles will be generated.")
        return []

    # 3. 执行生成
    print(f"\n--- Executing Generation Plan: {len(generation_plan)} profiles to create ---")
    all_profiles = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_generation_task, task): task for task in generation_plan}
        with tqdm(total=len(futures), desc="Generating Profiles") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_profiles.append(result)
                pbar.update(1)

    print(f"\n--- Generation Complete: Successfully generated {len(all_profiles)} profiles. ---")
    return all_profiles

# --- 主执行部分 ---
if __name__ == "__main__":
    if not check_llm_connectivity():
        sys.exit(1)
        
    # --- 配置 ---
    TOTAL_PROFILES_TO_GENERATE = 20 # 要生成的总画像数
    
    # 策略性增强配置 (可选，可为空列表 [])
    STRATEGIC_CONFIGS = [
        {
            "reaction_pattern": "绝望求死与危机边缘型",
            "count": 1 # 强制生成1个危机边缘型画像
        },
        # 您可以在此处添加更多策略性配置
    ]
    # --- 结束配置 ---

    print(f"--- Running Archetype Generator ---")
    print(f"Mode: Strategic Combinatorial, Target Profiles: {TOTAL_PROFILES_TO_GENERATE}")

    generated_profiles = generate_profiles(
        num_profiles=TOTAL_PROFILES_TO_GENERATE,
        strategic_configs=STRATEGIC_CONFIGS
    )

    # --- 保存结果 ---
    if generated_profiles:
        output_filename = "generated_character_profiles_v3.json"
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(generated_profiles, f, ensure_ascii=False, indent=2)
            print(f"\nAll {len(generated_profiles)} profiles saved to: {output_filename}")
        except IOError as e:
            print(f"\nError saving file '{output_filename}': {e}.")
    else:
        print("\nNo character profiles were generated in this run.")

    print(f"\nLLM Provider used was: {LLM_PROVIDER}")

