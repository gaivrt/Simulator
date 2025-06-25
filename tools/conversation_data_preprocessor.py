import json
import os

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
OUTPUT_FILE = "DifyLog_0613-中小学_plus_preprocessed.json"
MIN_MESSAGE_THRESHOLD = 3

if __name__ == '__main__':
    print(f"\n从 {INPUT_FILE} 加载对话数据进行预处理...")
    loaded_data = _load_json_for_test(INPUT_FILE)

    if loaded_data:
        print(f"\n原始对话数量: {len(loaded_data)}")
        print(f"开始过滤对话 (保留消息数量 >= {MIN_MESSAGE_THRESHOLD} 的对话)...")
        
        cleaned_data = filter_short_conversations(loaded_data, MIN_MESSAGE_THRESHOLD)
        
        print(f"\n过滤后的对话数量: {len(cleaned_data)}")
        print("保留的对话 (ID 和消息数) - 抽样前5条:")
        for conv_test in cleaned_data[:5]: # Print sample of first 5
            print(f"  - {conv_test.get('conversation_id')} (消息数: {len(conv_test.get('messages', []))})")
        
        _save_json_for_test(OUTPUT_FILE, cleaned_data)
        print(f"\n预处理完成。结果保存在 {OUTPUT_FILE}")
    else:
        print(f"未能加载数据从 {INPUT_FILE}，跳过过滤。") 