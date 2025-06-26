import json
import os
import re
import jieba
import jieba.posseg as pseg
from collections import Counter
import sys # Added for llm_utils path

# --- Attempt to import LLM utilities ---
LLM_UTILS_AVAILABLE = False
call_llm_api = None
check_llm_connectivity = None
# Assuming llm_utils.py is in the same directory (tools)
try:
    # Temporarily add the current directory to sys.path to find llm_utils
    # This is a common pattern if the script is run directly from its directory
    # and llm_utils is a sibling module not installed as a package.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from llm_utils import call_llm_api as llm_call_func, check_llm_connectivity as llm_check_func
    call_llm_api = llm_call_func
    check_llm_connectivity = llm_check_func
    LLM_UTILS_AVAILABLE = True
    print("Successfully imported llm_utils.")
except ImportError as e:
    print(f"Warning: Failed to import llm_utils. LLM-based filtering will be disabled. Error: {e}")
except Exception as e: # Catch any other exception during import
    print(f"Warning: An unexpected error occurred while importing llm_utils. LLM-based filtering will be disabled. Error: {e}")
finally:
    # Clean up sys.path if it was modified
    if 'current_dir' in locals() and current_dir in sys.path and sys.path[0] == current_dir:
        sys.path.pop(0)
# ---------------------------------------

# --- Attempt to import tqdm for progress bar ---
TQDM_AVAILABLE = False
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
    print("Successfully imported tqdm for progress bars.")
except ImportError:
    print("Warning: tqdm library not found. Progress bar for LLM filtering will not be shown.")
# ---------------------------------------------

# --- Parallel Processing Import ---
from concurrent.futures import ThreadPoolExecutor
# --------------------------------

# --- Text Preprocessing Functions ---

def step1_text_cleaning(text: str) -> str:
    """
    文本清洗 - 移除对聚类有害的噪声 (As per user's guide)
    """
    if not isinstance(text, str):
        print(f"警告: step1_text_cleaning 输入不是字符串，已跳过: {type(text)}")
        return "" # Return empty string or handle as appropriate

    # 1. 统一标点符号（重要！）
    # 原因：避免"焦虑。"和"焦虑！"被认为是不同的
    punctuation_map = {
        '！': '!', '？': '?', '，': ',', '。': '.',
        '；': ';', '：': ':', '"': '"', '"': '"',
        ''': "'", ''': "'", '（': '(', '）': ')',
        '【': '[', '】': ']', '、': ','
    }
    for cn, en in punctuation_map.items():
        text = text.replace(cn, en)

    # 2. 移除特殊字符和数字（除非是重要的）
    # 保留：时间表达（3天、2周）、程度表达（第1次、10分）
    text = re.sub(r'[^\w\\s\\u4e00-\\u9fff！？，。；：""\'\'（）【】]', ' ', text) # User's exact regex from guide

    # 3. 统一空格
    text = re.sub(r'\\s+', ' ', text).strip()

    # 4. 处理重复字符（很很很难过 → 很难过）
    text = re.sub(r'(.)\\1{2,}', r'\\1', text)

    return text

def step2_emotion_normalization(text: str) -> str:
    """
    情感强度归一化
    """
    if not isinstance(text, str):
        print(f"警告: step2_emotion_normalization 输入不是字符串，已跳过: {type(text)}")
        return ""
    intensity_levels = {
        '极度': '非常', '极其': '非常', '特别特别': '非常',
        '相当相当': '非常', '十分十分': '非常',
        '特别': '非常', '相当': '非常', '十分': '非常',
        '格外': '非常', '异常': '非常', '超级': '非常',
        '比较': '有点', '还算': '有点', '稍微': '有点',
        '多少': '有点', '多多少少': '有点',
        '一点点': '一点', '一丁点': '一点', '些许': '一点'
    }
    for original, normalized in intensity_levels.items():
        text = text.replace(original, normalized)
    text = re.sub(r'很{2,}', '非常', text)
    text = re.sub(r'太{2,}', '非常', text)
    return text

# 全局加载停用词表和心理学词汇
PSYCHOLOGY_TERMS_GLOBAL = [
    '焦虑症', '抑郁症', '强迫症', '恐惧症',
    '情绪调节', '认知偏差', '行为模式', '应对策略',
    '人际关系', '自我认知', '情感表达', '压力源'
]
for term in PSYCHOLOGY_TERMS_GLOBAL:
    jieba.add_word(term, freq=1000, tag='n')

# 从文件加载停用词
STOPWORDS_ZH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stopwords_zh.txt")
GLOBAL_STOPWORDS = set()
if os.path.exists(STOPWORDS_ZH_PATH):
    with open(STOPWORDS_ZH_PATH, 'r', encoding='utf-8') as f:
        GLOBAL_STOPWORDS = {line.strip() for line in f if line.strip()}
    print(f"已从 {STOPWORDS_ZH_PATH} 加载 {len(GLOBAL_STOPWORDS)} 个停用词。")
else:
    print(f"警告: 停用词文件 {STOPWORDS_ZH_PATH} 未找到。将使用空停用词列表。")


def step3_smart_segmentation(text: str) -> list:
    """
    智能分词与词性筛选
    """
    if not isinstance(text, str):
        print(f"警告: step3_smart_segmentation 输入不是字符串，已跳过: {type(text)}")
        return []
    words = pseg.cut(text)
    keep_pos = {'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vd', 'vn', 'a', 'ad', 'an', 'd', 'i', 'l'}
    valuable_words = []
    for word, pos in words:
        if (pos in keep_pos and len(word) > 1) or \
           (word in PSYCHOLOGY_TERMS_GLOBAL) or \
           (len(word) > 3):
            valuable_words.append(word)
    return valuable_words

def step4_optimized_stopwords(words: list) -> list:
    """
    优化停用词处理
    """
    if not isinstance(words, list):
        print(f"警告: step4_optimized_stopwords 输入不是列表，已跳过: {type(words)}")
        return []

    psychology_stopwords = {
        '感觉', '觉得', '认为', '以为', '好像', '似乎', '可能',
        '应该', '或许', '大概', '估计', '可能性', '据说',
        '听说', '看起来', '看上去', '看似', '貌似'
    }
    emotion_core_words = {
        '焦虑', '抑郁', '紧张', '害怕', '恐惧', '愤怒',
        '伤心', '难过', '开心', '高兴', '兴奋', '平静',
        '孤独', '寂寞', '委屈', '愧疚', '羞耻', '骄傲',
        '困惑', '迷茫', '无助', '绝望', '希望', '期待'
    }
    behavior_words = {
        '逃避', '面对', '沟通', '交流', '表达', '隐藏',
        '压抑', '释放', '控制', '失控', '坚持', '放弃',
        '接受', '拒绝', '支持', '反对', '理解', '误解'
    }
    
    # 使用全局加载的停用词，并合并心理学特有停用词
    # all_stopwords = GLOBAL_STOPWORDS | basic_stopwords | psychology_stopwords # basic_stopwords is removed as GLOBAL_STOPWORDS should cover it
    all_stopwords = GLOBAL_STOPWORDS | psychology_stopwords
    keep_words = emotion_core_words | behavior_words
    
    filtered_words = [word for word in words if word not in all_stopwords or word in keep_words]
    return filtered_words

def step5_synonym_normalization(words: list) -> list:
    """
    同义词归一化
    """
    if not isinstance(words, list):
        print(f"警告: step5_synonym_normalization 输入不是列表，已跳过: {type(words)}")
        return []
    emotion_synonyms = {
        '紧张': '焦虑', '不安': '焦虑', '担心': '焦虑',
        '忧虑': '焦虑', '恐慌': '焦虑', '惊慌': '焦虑',
        '难过': '抑郁', '伤心': '抑郁', '悲伤': '抑郁',
        '沮丧': '抑郁', '低落': '抑郁', '消沉': '抑郁',
        '生气': '愤怒', '愤慨': '愤怒', '恼火': '愤怒',
        '气愤': '愤怒', '暴躁': '愤怒', '烦躁': '愤怒',
        '躲避': '逃避', '回避': '逃避', '避开': '逃避',
        '沟通': '交流', '对话': '交流', '谈话': '交流'
    }
    degree_synonyms = {
        '特别': '非常', '相当': '非常', '十分': '非常',
        '颇为': '比较', '还算': '比较', '稍微': '有点'
    }
    all_synonyms = {**emotion_synonyms, **degree_synonyms}
    normalized_words = [all_synonyms.get(word, word) for word in words]
    return normalized_words

def handle_negation(words: list) -> list:
    """处理否定表达 (As per user's guide)"""
    if not isinstance(words, list):
        print(f"警告: handle_negation 输入不是列表，已跳过: {type(words)}")
        return []
    negation_words = {'不', '没', '无', '非', '未', '别', '莫'}
    processed = []
    i = 0
    while i < len(words):
        if words[i] in negation_words and i + 1 < len(words):
            processed.append(f"neg_{words[i+1]}") # User's original logic
            i += 2
        else:
            processed.append(words[i])
            i += 1
    return processed

def complete_preprocessing_pipeline(text: str) -> str:
    """
    完整的预处理管道
    """
    if not isinstance(text, str) or not text.strip(): # Check for empty or non-string input
        # print(f"警告: complete_preprocessing_pipeline 输入为空或不是字符串，已跳过: '{text}'")
        return "" # Return empty string for empty/invalid input
        
    text = step1_text_cleaning(text)
    text = step2_emotion_normalization(text)
    words = step3_smart_segmentation(text)
    words = step4_optimized_stopwords(words)
    words = step5_synonym_normalization(words)
    words = handle_negation(words)

    seen = set()
    unique_words = []
    for word in words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
    return ' '.join(unique_words)

# --- Original filter_short_conversations function ---
def filter_short_conversations(conversations: list, min_messages: int = 3) -> list:
    """
    Filters out conversations that have fewer than min_messages.
    Also handles conversations with missing or malformed 'messages' field.
    """
    cleaned_conversations = []
    if not isinstance(conversations, list):
        print("错误: filter_short_conversations 的输入必须是一个列表。")
        return []

    for i, conv in enumerate(conversations):
        conv_id = conv.get("conversation_id", f"未知ID_索引_{i}")
        messages = conv.get("messages")

        if not isinstance(messages, list):
            print(f"警告: 对话 '{conv_id}' 没有 'messages' 列表或者 'messages' 不是一个列表。已跳过。")
            continue

        if len(messages) >= min_messages:
            cleaned_conversations.append(conv)
        else:
            print(f"提示: 对话 '{conv_id}' 有 {len(messages)} 条消息 (少于 {min_messages} 条)。已过滤。")
            
    return cleaned_conversations

# --- LLM-based Filtering Functions ---
def _get_conversation_text_for_llm(conversation: dict, max_messages_to_sample: int = 10) -> str:
    """
    Extracts and formats text from a conversation for LLM analysis.
    Tries to get roles (user/assistant) if available.
    """
    if not isinstance(conversation, dict) or "messages" not in conversation or not isinstance(conversation["messages"], list):
        return ""

    text_parts = []
    messages_to_process = conversation["messages"][:max_messages_to_sample]

    for msg in messages_to_process:
        if not isinstance(msg, dict):
            continue
        
        content = msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue

        role = str(msg.get("role", "Participant")).capitalize() # Default to "Participant" if no role
        # Avoid generic system/tool messages if they don't represent actual dialogue
        if role.lower() in ["system", "tool"] and len(content) < 50 : # Heuristic
            continue
        text_parts.append(f"{role}: {content}")
        
    return "\n".join(text_parts)

def analyze_conversation_meaningfulness_llm(conversation_text: str, model_override: str = None):
    """
    Uses LLM to analyze if a conversation text is meaningful for psychological context.
    Returns: A dict like {"is_meaningful": bool, "reason": str} or None if error.
    """
    if not LLM_UTILS_AVAILABLE or not call_llm_api:
        print("LLM utils not available, skipping meaningfulness analysis.")
        return {"is_meaningful": True, "reason": "LLM utils not available, defaulted to meaningful."}

    if not conversation_text.strip():
        return {"is_meaningful": False, "reason": "Empty conversation text."}

    system_prompt = f"""
You are an AI assistant highly specialized in analyzing conversation transcripts from psychological counseling or mental well-being support contexts. Your task is to determine if a given conversation snippet is 'meaningful' or 'meaningless'.

A 'meaningful' conversation:
- Shows a genuine attempt at interaction related to psychological well-being, emotional states, life challenges, or seeking/providing support.
- Contains coherent questions, statements, or expressions of feelings.
- Examples: "I've been feeling very anxious lately.", "Can you help me understand why I react this way?", "User: I'm sad. Assistant: Tell me more."

A 'meaningless' conversation:
- Appears to be a test input (e.g., "test", "hello world" by itself without context).
- Consists of random characters, gibberish, or placeholders.
- Is an automated system message not part of a genuine user-assistant dialogue (e.g., "Session started", "Typing...").
- Is too short and generic to ascertain any psychological relevance (e.g., just "Hi", "Ok").
- Contains primarily code, logs, or non-conversational content.

Analyze the following conversation snippet. Respond ONLY in JSON format with two keys:
1. "is_meaningful": boolean (true if the conversation is meaningful, false otherwise).
2. "reason": a brief string (max 20 words) explaining your decision, especially if you classify it as meaningless.

Conversation Snippet:
---
{conversation_text}
---
"""
    try:
        response = call_llm_api(system_prompt, model_override=model_override, expect_json=True)
        if response and isinstance(response, dict) and "is_meaningful" in response:
            if not isinstance(response.get("is_meaningful"), bool): # Validate type
                 print(f"LLM Warning: 'is_meaningful' is not a boolean. Defaulting to True. Response: {response}")
                 return {"is_meaningful": True, "reason": "LLM response for is_meaningful was not a boolean."}
            return {
                "is_meaningful": response["is_meaningful"],
                "reason": str(response.get("reason", "No reason provided by LLM."))
            }
        else:
            print(f"LLM Warning: Invalid or missing 'is_meaningful' in response. Defaulting to True. Response: {response}")
            return {"is_meaningful": True, "reason": "LLM response malformed or missing key, defaulted to meaningful."}
    except Exception as e:
        print(f"LLM Error: Failed during meaningfulness analysis. Defaulting to True. Error: {e}")
        return {"is_meaningful": True, "reason": f"LLM call failed ({type(e).__name__}), defaulted to meaningful."}

def filter_conversations_by_llm(conversations: list, llm_model: str = None, num_workers: int = 10) -> list:
    """
    Filters conversations based on LLM's judgment of their meaningfulness using parallel processing.
    """
    if not LLM_UTILS_AVAILABLE or not check_llm_connectivity or not call_llm_api :
        print("LLM utils or connectivity not available. Skipping LLM-based filtering.")
        return conversations
    
    # Perform a connectivity check before starting a potentially long batch
    print("Checking LLM connectivity before starting LLM-based filtering...")
    if not check_llm_connectivity(verbose=False): # verbose=False to avoid too much print
        print("LLM connectivity check failed. Skipping LLM-based filtering.")
        return conversations
    print("LLM connectivity OK.")

    # Helper function for parallel execution
    def _process_conv_for_llm(conv_tuple):
        original_conv, conv_id_str = conv_tuple
        conversation_text = _get_conversation_text_for_llm(original_conv)
        if not conversation_text.strip():
            # Return original conversation and judgment directly for empty ones
            return original_conv, conv_id_str, {"is_meaningful": False, "reason": "Empty conversation text (LLM skipped)."}
        llm_judgment = analyze_conversation_meaningfulness_llm(conversation_text, model_override=llm_model)
        return original_conv, conv_id_str, llm_judgment

    meaningful_conversations_final = []
    filtered_out_details = [] # To store details of filtered out conversations
    
    # Prepare tuples of (conversation, conv_id_str) for ThreadPoolExecutor
    convs_with_ids_to_process = []
    for i, conv_item in enumerate(conversations):
        conv_id = conv_item.get("conversation_id", f"未知ID_索引_{i}")
        convs_with_ids_to_process.append((conv_item, conv_id))

    total_conversations_to_process = len(convs_with_ids_to_process)
    print(f"Starting LLM-based meaningfulness filtering for {total_conversations_to_process} conversations with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # executor.map processes items in order they are submitted and returns results in that order
        results_iterator = executor.map(_process_conv_for_llm, convs_with_ids_to_process)
        
        if TQDM_AVAILABLE:
            # tqdm will show progress as results are yielded from the map
            results_with_progress = tqdm(results_iterator, total=total_conversations_to_process, desc="LLM Meaningfulness Filtering")
        else:
            results_with_progress = results_iterator
            print(f"Processing {total_conversations_to_process} conversations with LLM (tqdm progress bar not available). Iteration count will be manual.")

        for i, (original_conv, conv_id_str, llm_judgment) in enumerate(results_with_progress):
            if llm_judgment["is_meaningful"]:
                meaningful_conversations_final.append(original_conv)
            else:
                # Get a snippet for logging, be careful with potentially large content
                snippet = _get_conversation_text_for_llm(original_conv, max_messages_to_sample=1)[:100] + "..."
                filtered_out_details.append({
                    "id": conv_id_str, 
                    "reason": llm_judgment['reason'],
                    "snippet": snippet
                })
            if not TQDM_AVAILABLE and (i + 1) % 50 == 0:
                 print(f"  LLM Filter (manual progress): Processed {i + 1}/{total_conversations_to_process}...")

    if filtered_out_details:
        print(f"\n--- LLM Filtered Out {len(filtered_out_details)} Conversations (Sample Details) ---")
        for detail in filtered_out_details[:5]: # Print details for the first few
            print(f"  ID: {detail['id']}, Reason: {detail['reason']}, Snippet: '{detail['snippet']}'")
        if len(filtered_out_details) > 5:
            print(f"  ... and {len(filtered_out_details) - 5} more filtered by LLM.")
    
    # The function now internally logs details of filtered items.
    # The caller (`if __name__ == '__main__':`) calculates and prints the overall count of filtered items.
    return meaningful_conversations_final

# --- Helpers for standalone testing ---
def _load_json_for_test(filepath: str) -> list:
    """Helper to load JSON data for testing this module."""
    if not os.path.exists(filepath):
        print(f"错误: 测试输入文件 {filepath} 未找到。")
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"加载测试文件 {filepath} 时出错: {e}")
        return []

def _save_json_for_test(filepath: str, data: list):
    """Helper to save JSON data for testing this module."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"测试数据已保存到 {filepath}")
    except Exception as e:
        print(f"保存测试文件 {filepath} 时出错: {e}")

# --- Configuration for standalone execution ---
INPUT_FILE = "DifyLog_0613-中小学_plus.json"
OUTPUT_FILE = "DifyLog_0613-中小学_plus_preprocessed_llmfiltered_v3.json" # New name with LLM filter
MIN_MESSAGE_THRESHOLD = 3
ENABLE_LLM_MEANINGFULNESS_FILTER = True  # Set to True to enable LLM filtering
LLM_FILTER_MODEL_OVERRIDE = None        # e.g., "gemini-1.5-flash" or specific OpenAI model if needed
LLM_FILTER_NUM_WORKERS = 10 # Number of parallel workers for LLM filtering

if __name__ == '__main__':
    print(f"\n从 {INPUT_FILE} 加载对话数据进行预处理...")
    loaded_data = _load_json_for_test(INPUT_FILE)

    if loaded_data:
        initial_conv_count = len(loaded_data)
        print(f"\n原始对话数量: {initial_conv_count}")
        
        # --- Rule-based filtering (Length) ---
        print(f"\n阶段1: 按消息数量过滤 (保留消息数量 >= {MIN_MESSAGE_THRESHOLD} 的对话)...")
        data_after_length_filter = filter_short_conversations(loaded_data, MIN_MESSAGE_THRESHOLD)
        convs_after_length_filter_count = len(data_after_length_filter)
        removed_by_length_rule_count = initial_conv_count - convs_after_length_filter_count
        print(f"规则过滤完成。通过此规则删除对话: {removed_by_length_rule_count}。")
        print(f"剩余对话数量: {convs_after_length_filter_count}")

        # --- LLM-based filtering (Meaningfulness) ---
        data_after_llm_filter = data_after_length_filter # Initialize with previous step's output
        removed_by_llm_filter_count = 0 

        if ENABLE_LLM_MEANINGFULNESS_FILTER:
            if LLM_UTILS_AVAILABLE:
                print(f"\n阶段2: LLM意义判断过滤 (输入对话数量: {len(data_after_length_filter)})...")
                count_before_llm_actual_filter = len(data_after_length_filter)
                data_after_llm_filter = filter_conversations_by_llm(data_after_length_filter, LLM_FILTER_MODEL_OVERRIDE, num_workers=LLM_FILTER_NUM_WORKERS)
                removed_by_llm_filter_count = count_before_llm_actual_filter - len(data_after_llm_filter)
                print(f"LLM意义判断过滤完成。通过LLM判断删除对话: {removed_by_llm_filter_count}。")
                print(f"LLM过滤后剩余对话数量: {len(data_after_llm_filter)}")
            else:
                print("\n警告: LLM工具不可用，跳过LLM意义判断过滤步骤 (阶段2)。")
        else:
            print("\n提示: LLM意义判断过滤已禁用 (跳过阶段2)。")

        # --- Detailed text preprocessing ---
        print(f"\n阶段3: 详细文本预处理 (处理对话数量: {len(data_after_llm_filter)})...")
        processed_conversations_final = []
        if not data_after_llm_filter: # Check if the list is empty before iterating
            print("  没有对话可进行详细文本预处理。")
        else:
            iterable_for_final_processing = data_after_llm_filter
            if TQDM_AVAILABLE:
                iterable_for_final_processing = tqdm(data_after_llm_filter, total=len(data_after_llm_filter), desc="Detailed Text Preprocessing")
            else:
                print(f"  开始详细文本预处理 {len(data_after_llm_filter)} 条对话 (tqdm未启用)") 

            for i, conv in enumerate(iterable_for_final_processing):
                processed_conv = conv.copy() 
                processed_messages = []
                if isinstance(conv.get("messages"), list):
                    for msg_idx, msg in enumerate(conv["messages"]):
                        processed_msg = msg.copy()
                        original_content = msg.get("content", "")
                        
                        if not isinstance(original_content, str):
                            print(f"  警告: 对话 {conv.get('conversation_id', i)}-消息{msg_idx}: 'content' 不是字符串 (类型: {type(original_content)}), 将使用空字符串。原始值: {original_content}")
                            original_content = ""

                        if original_content.strip(): 
                            processed_content = complete_preprocessing_pipeline(original_content)
                            processed_msg["content_processed"] = processed_content
                        else:
                            processed_msg["content_processed"] = "" 
                        processed_messages.append(processed_msg)
                else:
                    print(f"  警告: 对话 {conv.get('conversation_id', i)} 的 'messages' 字段缺失或格式不正确，详细文本预处理跳过。")
                
                processed_conv["messages"] = processed_messages
                processed_conversations_final.append(processed_conv)
                
                if not TQDM_AVAILABLE and (i + 1) % 100 == 0:
                    print(f"    (手动进度)已完成 {i + 1}/{len(data_after_llm_filter)} 条对话的详细文本预处理...")

        print(f"详细文本预处理完成。")

        # --- Filtering Statistics Summary ---
        print("\n--- 过滤统计摘要 ---")
        print(f"加载的原始对话总数: {initial_conv_count}")
        print(f"阶段1 (规则: 消息数量 < {MIN_MESSAGE_THRESHOLD}):")
        print(f"  - 删除数量: {removed_by_length_rule_count}")
        print(f"  - 过滤后剩余: {convs_after_length_filter_count}")
        
        print(f"阶段2 (LLM: 无意义判断, 使用 {LLM_FILTER_NUM_WORKERS} workers):")
        if ENABLE_LLM_MEANINGFULNESS_FILTER:
            if LLM_UTILS_AVAILABLE:
                print(f"  - 基于LLM判断删除数量: {removed_by_llm_filter_count}")
            else:
                print(f"  - 基于LLM判断删除数量: 0 (LLM工具不可用，已跳过)")
        else:
             print(f"  - 基于LLM判断删除数量: 0 (LLM过滤已禁用)")
        print(f"  - LLM过滤后剩余 (进入详细预处理): {len(data_after_llm_filter)}")
        
        print(f"阶段3 (详细文本预处理):")
        print(f"  - 成功处理并最终输出的对话数量: {len(processed_conversations_final)}")
        print("-----------------------\n")
        
        _save_json_for_test(OUTPUT_FILE, processed_conversations_final)
        print(f"\n预处理完成。结果保存在 {OUTPUT_FILE}")
    else:
        print(f"未能加载数据从 {INPUT_FILE}，跳过所有处理。") 