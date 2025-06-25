import json
import yaml
import os
import concurrent.futures
import re # Added for more robust stripping
from tqdm import tqdm # Added for progress bar
import sys # Added sys

# Add the project root directory (one level up from the 'tools' directory) to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- LLM Utils Import ---
from tools.llm_utils import call_llm_api, check_llm_connectivity
# ------------------------

# --- Configuration for Parallel Processing ---
MAX_CONCURRENT_ANALYSIS_WORKERS = 50  # Adjust as needed, consider API rate limits

# --- Constants for output fields ---
CORE_DRIVE_TEXT_FIELD = "core_drive_text_for_clustering"
REACTION_PATTERN_TEXT_FIELD = "reaction_pattern_text_for_clustering"

# --- Configuration ---
INPUT_DATA_PATH = "DifyLog_0613-中小学_plus_preprocessed.json" # Will be managed by the main pipeline
OUTPUT_DATA_PATH = "DifyLog_0613-中小学_plus_analyzed.json" # Will be managed by the main pipeline

# --- Prompt Templates ---
REACTION_PATTERN_PROMPT_TEMPLATE = (
    """
    你是一位资深的心理数据分析师和角色设计师。
    你的任务是分析提供给你的数据。
    基于对以下对话数据的整体理解，请你总结出一种主要的用户 reaction_pattern。
    对话数据:
    {conversation_log}
    请严格按照以下 YAML 格式输出 reaction pattern 示例:

    - name: "创伤叙事与细节描述型"
    description: |
        角色:
        你是一个亲身经历过严重负面事件（如校园暴力、意外事故）的学生。你来到这里的目的是为了倾诉，并通过详细、有逻辑地复述整个事件，来处理内心的创伤和强烈的愤怒、恐惧情绪。

        语气特点:
        - 叙事完整：你的描述会包含清晰的时间线、人物、地点和事件的来龙去脉。
        - 细节丰富：对话中会主动提供大量具体细节，例如"高保密的碎纸机"、"颗粒状的碎末"。
        - 情绪强烈且一致：你的愤怒、恐惧和委屈等情绪都与你所叙述的核心创伤事件紧密相连。
        - 描述身体感受：在描述情绪的同时，会伴随具体的身体反应，如"拳头很硬"、"心跳的很快"。

        请参考以下示例进行扮演:
        - "昨天晚上有人给我作业撕烂了。"
        - "他拿着把美工刀抵着我脖子说不让我告校长，我偏要告。"
        - "首先我听到的是一声巨响，然后一团大火球冲出洗衣房，整个机身除底部外5个面的碎片都给炸飞了。"

    请严格按照以下 **YAML** 格式输出 reaction pattern 分析:
    """
)

CORE_DRIVE_PROMPT_TEMPLATE = (
    """
    你是一位资深的心理数据分析师和角色设计师。你的任务是基于心理学理论和对人类动机的深刻理解，定义一系列核心的心理驱动力（Core Drives）。这些驱动力是构成人类行为和情感反应的基石，以生成丰富、立体的角色。
    你的任务是分析提供给你的数据。你接收到的是一段用户与AI心理咨询师的对话记录，请你给这段对话进行动机分析。每一个原型都应代表一种基本的人类需求、恐惧或欲望的组合。
    对话数据:
    {conversation_log}
    对于这段对话所反映的核心驱动力，请提供以下信息，并严格按照 **YAML** 格式输出：
    name: (为该驱动力命名。名称应该简洁地概括其核心，例如"掌控与秩序追求者-失控恐惧"或"连接与归属渴望-被拒恐惧")
    description: |
      (详细描述该驱动力的核心心理机制。说明：
      该角色最根本的渴望或追求是什么（例如：安全感、认可、自主性、亲密关系等）。
      这种渴望背后的核心信念是什么（例如："只有当我完美时，我才值得被爱"或"世界是危险的，我必须时刻保持警惕")。
      与此驱动力相关的最深刻的恐惧是什么（例如：恐惧失败、恐惧被抛弃、恐惧失序、恐惧无意义）。
      这种内在的渴望和恐惧如何转化为外在的行为模式或人生策略（例如：追求完美、讨好他人、寻求权力、避免亲密关系等）。)

    示例：
    - name: "价值追求者-失败恐惧"
      description: |
        该角色的自我价值完全与外在成就（主要是成绩、比赛名次等）绑定。他们认为只有持续的成功才能获得爱与认可，因此极度恐惧任何形式的失败，并将失败视为自我价值的彻底否定。面对挑战时，他们可能会过度准备、焦虑，或在预感可能失败时选择回避。

    """
)


# --- Data Handling ---
def load_json_data(filepath: str) -> list:
    """从 JSON 文件加载数据。"""
    if not os.path.exists(filepath):
        print(f"错误: 输入文件 {filepath} 未找到。将创建一个示例。")
        return [
          {
            "conversation_id": "example_conv_analyzer_load",
            "messages": [
              {"role": "user", "content": "你好。", "timestamp": "2025-06-05T10:00:00", "id": 0}
            ]
          }
        ]
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"错误: 文件 {filepath} 不是有效的 JSON 格式。")
        return []
    except Exception as e:
        print(f"加载文件 {filepath} 时发生错误: {e}")
        return []

def save_intermediate_analysis(filepath: str, data: list):
    """将分析中间结果保存到 JSON 文件。"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"分析中间结果已成功保存到 {filepath}")
    except Exception as e:
        print(f"保存文件 {filepath} 时发生错误: {e}")

def _strip_yaml_markdown_delimiters(yaml_string: str) -> str:
    """Strips common markdown code block delimiters and problematic formatting from a YAML string."""
    if not isinstance(yaml_string, str):
        return yaml_string

    stripped_string = yaml_string.strip()
    # Remove markdown code block delimiters first (```yaml ... ``` or ``` ... ``` or ```json ... ```)
    if stripped_string.startswith("```yaml") and stripped_string.endswith("```"):
        stripped_string = stripped_string[len("```yaml"):].strip()
        stripped_string = stripped_string[:-len("```")].strip()
    elif stripped_string.startswith("```json") and stripped_string.endswith("```"):
        stripped_string = stripped_string[len("```json"):].strip()
        stripped_string = stripped_string[:-len("```")].strip()
    elif stripped_string.startswith("```") and stripped_string.endswith("```"):
        stripped_string = stripped_string[len("```"):].strip()
        stripped_string = stripped_string[:-len("```")].strip()
    
    lines = stripped_string.split('\n')
    processed_lines = []
    for line in lines:
        current_line = line

        # Convert '* list item' to '- list item' (if * is followed by a space)
        current_line = re.sub(r'^(\s*)\*\s+(.+)', r'\1- \2', current_line)

        # First, strip 'user: ' or 'assistant: ' prefixes from list items
        current_line = re.sub(r'^(\s*-\s*)(?:user|assistant):\s*(.*)', r'\1\2', current_line)
        
        # Remove ** from key names like "**Key**: Value"
        current_line = re.sub(r'^(\s*)\*\*(.+?)\*\*(\s*:)', r'\1\2\3', current_line)

        # Remove ** from list items like "- **text**:" or "- **text**"
        current_line = re.sub(r'^(\s*-\s*)\*\*(.*?)\*\*(\s*:?)', r'\1\2\3', current_line)
        
        # General removal of **bold** markdown
        current_line = re.sub(r'\*\*(.*?)\*\*', r'\1', current_line)
        
        # General removal of _italic_ markdown
        current_line = re.sub(r'_(.*?)_', r'\1', current_line)

        # Remove backticks `code`
        current_line = current_line.replace('`', '')
        
        processed_lines.append(current_line)
    
    return "\n".join(processed_lines)

# --- New Helper function to extract text from parsed YAML ---
def _extract_main_text_from_parsed_yaml(parsed_item, context_label: str, original_cleaned_str: str) -> str:
    """
    Extracts a primary textual description from a parsed YAML object.
    Handles various structures like lists of dicts, dicts, strings, or None.
    """
    if isinstance(parsed_item, str): # If LLM returns a plain string or the item is already a string error message
        return parsed_item.strip()

    if parsed_item is None:
        if original_cleaned_str and original_cleaned_str.strip(): # Check if it was non-empty before parsing
            return f"{context_label} (YAML parse resulted in None from non-empty input)"
        else:
            return f"{context_label} (YAML parse resulted in None or input was empty/non-YAML)"

    target_dict = None
    if isinstance(parsed_item, list):
        if parsed_item and isinstance(parsed_item[0], dict):
            target_dict = parsed_item[0] # Focus on the first dictionary in the list
        else:
            # If it's a list but not of dicts, or an empty list
            return f"{context_label} (parsed as list, but not the expected list of dictionaries): {str(parsed_item)[:200]}..."
    elif isinstance(parsed_item, dict):
        target_dict = parsed_item
    else: # Not a list, not a dict, not a string, not None
        return f"{context_label} (parsed, but resulted in an unexpected data type {type(parsed_item)}): {str(parsed_item)[:200]}..."

    if target_dict: # If we have a dictionary (either directly parsed or from a list)
        # Primary key: "description"
        description = target_dict.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()

        # Fallback for Reaction Pattern specific key "角色描述"
        # Check if context_label indicates it's for a Reaction Pattern (e.g., contains "Reaction Pattern")
        if "Reaction Pattern" in context_label: 
            role_desc = target_dict.get("角色描述") # As per your prompt for RP
            if isinstance(role_desc, str) and role_desc.strip():
                return role_desc.strip()
        
        # Fallback to "name" if "description" (and "角色描述" for RP) is not found or not a string
        name = target_dict.get("name")
        if isinstance(name, str) and name.strip():
            return f"{context_label} (extracted name: '{name}', full content snippet: {str(target_dict)[:150]}...)"
        
        return f"{context_label} (parsed dictionary, but no standard 'description', '角色描述', or 'name' field found): {str(target_dict)[:200]}..."
    
    return f"{context_label} (unexpected structure or empty content after parsing attempt): {str(parsed_item)[:200]}..."


# --- Helper function for single conversation analysis (for parallel execution) ---
def _analyze_single_conversation(conv_data_tuple):
    i, conv_data = conv_data_tuple # Unpack index and data
    conv_id = conv_data.get('conversation_id', f'N/A_idx_{i}')
    messages = conv_data.get("messages", [])
    conversation_log_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    # Default texts before analysis, including conv_id for easier tracking
    extracted_reaction_pattern_text = f"Reaction pattern analysis inconclusive for {conv_id}"
    extracted_core_drive_text = f"Core drive analysis inconclusive for {conv_id}"

    if not messages:
        extracted_reaction_pattern_text = "Analysis skipped: No messages found"
        extracted_core_drive_text = "Analysis skipped: No messages found"
        return {
            "conversation_id": conv_id,
            "conversation_content": conversation_log_str, # Will be an empty string
            REACTION_PATTERN_TEXT_FIELD: extracted_reaction_pattern_text,
            CORE_DRIVE_TEXT_FIELD: extracted_core_drive_text
        }

    # --- Reaction Pattern Analysis ---
    # Initialize for error logging clarity, in case LLM call itself fails or returns None early
    reaction_pattern_yaml_str_for_error_logging = "LLM call not made or result was None"
    cleaned_rp_yaml_str_for_error_logging = "Cleaning not attempted or input was empty/non-string"
    try:
        reaction_pattern_prompt = REACTION_PATTERN_PROMPT_TEMPLATE.format(conversation_log=conversation_log_str)
        llm_rp_output = call_llm_api(reaction_pattern_prompt, expect_json=False)
        
        current_context_label = f"Reaction Pattern (for {conv_id})"

        if llm_rp_output is None:
            extracted_reaction_pattern_text = f"{current_context_label} Error: LLM API returned None."
        elif isinstance(llm_rp_output, str):
            reaction_pattern_yaml_str_for_error_logging = llm_rp_output # Store for logging
            if not llm_rp_output.strip():
                extracted_reaction_pattern_text = f"{current_context_label} Error: LLM returned an empty string."
            else:
                cleaned_rp_yaml_str = _strip_yaml_markdown_delimiters(llm_rp_output)
                cleaned_rp_yaml_str_for_error_logging = cleaned_rp_yaml_str # Store for logging
                if not cleaned_rp_yaml_str.strip():
                    extracted_reaction_pattern_text = f"{current_context_label} Error: LLM output was empty after cleaning."
                else:
                    parsed_rp_obj = yaml.safe_load(cleaned_rp_yaml_str)
                    extracted_reaction_pattern_text = _extract_main_text_from_parsed_yaml(parsed_rp_obj, current_context_label, cleaned_rp_yaml_str)
        else: # llm_rp_output is not None and not a string (e.g. an error dict from call_llm_api)
            reaction_pattern_yaml_str_for_error_logging = str(llm_rp_output) # Log its string representation
            # For non-string LLM output, pass it directly to the extractor
            extracted_reaction_pattern_text = _extract_main_text_from_parsed_yaml(llm_rp_output, f"{current_context_label}, direct LLM obj", str(llm_rp_output))

    except yaml.YAMLError as e:
        print(f"\n  YAML 解析错误 (对话 {conv_id}, Reaction Pattern): {e}")
        print(f"  LLM Raw Output (Reaction Pattern, {conv_id}):\n{str(reaction_pattern_yaml_str_for_error_logging)[:500]}...")
        print(f"  Cleaned YAML Attempt (Reaction Pattern, {conv_id}):\n{str(cleaned_rp_yaml_str_for_error_logging)[:500]}...")
        extracted_reaction_pattern_text = f"Reaction Pattern Error (for {conv_id}): Invalid YAML - {str(e)[:100]}..."
    except Exception as e_gen:
        print(f"\n  处理 Reaction Pattern 时发生意外错误 (对话 {conv_id}): {e_gen}")
        extracted_reaction_pattern_text = f"Reaction Pattern Error (for {conv_id}): Unexpected error during processing - {str(e_gen)[:100]}..."

    # --- Core Drive Analysis ---
    core_drive_yaml_str_for_error_logging = "LLM call not made or result was None"
    cleaned_cd_yaml_str_for_error_logging = "Cleaning not attempted or input was empty/non-string"
    try:
        core_drive_prompt = CORE_DRIVE_PROMPT_TEMPLATE.format(conversation_log=conversation_log_str)
        llm_cd_output = call_llm_api(core_drive_prompt, expect_json=False)
        
        current_context_label = f"Core Drive (for {conv_id})"

        if llm_cd_output is None:
            extracted_core_drive_text = f"{current_context_label} Error: LLM API returned None."
        elif isinstance(llm_cd_output, str):
            core_drive_yaml_str_for_error_logging = llm_cd_output
            if not llm_cd_output.strip():
                extracted_core_drive_text = f"{current_context_label} Error: LLM returned an empty string."
            else:
                cleaned_cd_yaml_str = _strip_yaml_markdown_delimiters(llm_cd_output)
                cleaned_cd_yaml_str_for_error_logging = cleaned_cd_yaml_str
                if not cleaned_cd_yaml_str.strip():
                    extracted_core_drive_text = f"{current_context_label} Error: LLM output was empty after cleaning."
                else:
                    parsed_cd_obj = yaml.safe_load(cleaned_cd_yaml_str)
                    extracted_core_drive_text = _extract_main_text_from_parsed_yaml(parsed_cd_obj, current_context_label, cleaned_cd_yaml_str)
        else: # llm_cd_output is not None and not a string
            core_drive_yaml_str_for_error_logging = str(llm_cd_output)
            extracted_core_drive_text = _extract_main_text_from_parsed_yaml(llm_cd_output, f"{current_context_label}, direct LLM obj", str(llm_cd_output))
            
    except yaml.YAMLError as e:
        print(f"\n  YAML 解析错误 (对话 {conv_id}, Core Drive): {e}")
        print(f"  LLM Raw Output (Core Drive, {conv_id}):\n{str(core_drive_yaml_str_for_error_logging)[:500]}...")
        print(f"  Cleaned YAML Attempt (Core Drive, {conv_id}):\n{str(cleaned_cd_yaml_str_for_error_logging)[:500]}...")
        extracted_core_drive_text = f"Core Drive Error (for {conv_id}): Invalid YAML - {str(e)[:100]}..."
    except Exception as e_gen:
        print(f"\n  处理 Core Drive 时发生意外错误 (对话 {conv_id}): {e_gen}")
        extracted_core_drive_text = f"Core Drive Error (for {conv_id}): Unexpected error during processing - {str(e_gen)[:100]}..."

    # Construct the new simplified dictionary with exactly four fields
    current_analyzed_conv = {
        "conversation_id": conv_id,
        "conversation_content": conversation_log_str,
        REACTION_PATTERN_TEXT_FIELD: extracted_reaction_pattern_text,
        CORE_DRIVE_TEXT_FIELD: extracted_core_drive_text
    }
    return current_analyzed_conv

# --- Core Analysis Logic (Modified for parallel execution) ---
def analyze_conversations(raw_conversations: list) -> list:
    """
    对原始对话数据进行 Reaction Pattern, Core Drive 和 Enhanced Description 分析。
    使用 ThreadPoolExecutor 并行处理。
    """
    print(f"准备并行分析 {len(raw_conversations)} 条对话 (使用最多 {MAX_CONCURRENT_ANALYSIS_WORKERS} 个工作线程)...")
    analyzed_data_results = []

    conversations_with_indices = list(enumerate(raw_conversations))

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_ANALYSIS_WORKERS) as executor:
        try:
            results_iterator = executor.map(_analyze_single_conversation, conversations_with_indices)
            analyzed_data_results = list(tqdm(results_iterator, total=len(raw_conversations), desc="分析对话中"))
        except Exception as e:
            print(f"并行分析对话时发生严重错误: {e}")
            if not analyzed_data_results: 
                updated_results = []
                for i_fallback, conv_data_fallback in conversations_with_indices:
                    conv_id_fallback = conv_data_fallback.get('conversation_id', f'N/A_idx_{i_fallback}')
                    messages_fallback = conv_data_fallback.get("messages", [])
                    conversation_log_str_fallback = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages_fallback])
                    updated_results.append({
                        "conversation_id": conv_id_fallback,
                        "conversation_content": conversation_log_str_fallback,
                        REACTION_PATTERN_TEXT_FIELD: f"Analysis failed for {conv_id_fallback} due to parallel processing error: {e}",
                        CORE_DRIVE_TEXT_FIELD: f"Analysis failed for {conv_id_fallback} due to parallel processing error: {e}"
                    })
                analyzed_data_results = updated_results
    
    print(f"所有对话的并行分析任务已提交/完成。收集到 {len(analyzed_data_results)} 个结果。")
    analyzed_data_results = [res for res in analyzed_data_results if res is not None] # Should not happen with current logic
    return analyzed_data_results

# Example usage (can be removed or kept for direct testing of this module)
if __name__ == '__main__':
    # --- LLM Connectivity Check (Required for testing this module standalone) ---
    print("\nStandalone Analyzer Test: Checking LLM connectivity...")
    if not check_llm_connectivity(verbose=True): 
        print("LLM connectivity check failed. Analyzer module test might not produce meaningful results.")
        print("Please ensure your .env file is configured and llm_utils.py is accessible.")
    else:
        print("LLM connectivity OK for analyzer test.")
    # --------------------------------------------------------------------------

    print(f"加载测试数据从: {INPUT_DATA_PATH}")
    test_raw_data = load_json_data(INPUT_DATA_PATH)
    if test_raw_data:
        print("开始独立分析测试 (并行)...")
        analyzed_results = analyze_conversations(test_raw_data)
        print(f"\n分析完成，结果数量: {len(analyzed_results)}")

        print("\n--- Sample of Analyzed Results (Strictly 4 Fields) ---")
        for i, res in enumerate(analyzed_results[:min(3, len(analyzed_results))]): # Print first 3 or fewer
            print(f"\nResult {i+1}:")
            print(f"  Conversation ID: {res.get('conversation_id')}")
            print(f"  Conversation Content (Snippet): {res.get('conversation_content', 'N/A')[:100]}...")
            print(f"  Reaction Pattern Text: {res.get(REACTION_PATTERN_TEXT_FIELD, 'N/A')}")
            print(f"  Core Drive Text: {res.get(CORE_DRIVE_TEXT_FIELD, 'N/A')}")

        save_intermediate_analysis(OUTPUT_DATA_PATH, analyzed_results)
    else:
        print(f"无法加载测试数据从 {INPUT_DATA_PATH}") 