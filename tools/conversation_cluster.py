import json
import numpy as np
from sklearn.cluster import KMeans
import os
import sys # Added sys
import random # Added for selecting random cluster member
import yaml # Added for YAML output

# Add the project root directory (one level up from the 'tools' directory) to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- LLM Utils Import ---
from tools.llm_utils import get_embeddings_from_api, check_embedding_connectivity, call_llm_api, check_llm_connectivity
# ------------------------

# --- Configuration for standalone testing ---
N_CLUSTERS_DEFAULT = 10 # Default, can be overridden by orchestrator

# --- Constants for new text fields and output files ---
CORE_DRIVE_TEXT_FIELD = "core_drive_text_for_clustering"
REACTION_PATTERN_TEXT_FIELD = "reaction_pattern_text_for_clustering"

CORE_DRIVES_YAML_OUTPUT_FILE = "generated_core_drives.yaml"
REACTION_PATTERNS_YAML_OUTPUT_FILE = "generated_reaction_patterns.yaml"

# --- Configuration for standalone execution ---
ANALYZED_INPUT_FILE = "DifyLog_0613-中小学_plus_analyzed.json"
# N_CLUSTERS_TEST remains as a configurable parameter for testing.
# Output YAML filenames remain as defined constants.

# --- Data Handling (for standalone testing) ---
def load_json_data_for_clustering(filepath: str) -> list:
    """从 JSON 文件加载分析后的数据（用于独立测试聚类模块）。"""
    if not os.path.exists(filepath):
        print(f"错误: 输入文件 {filepath} 未找到。请提供一个包含分析后数据的JSON文件进行测试。")
        # Example structure for testing, now includes new fields
        return [
            {
                "conversation_id": "analyzed_sample_1",
                CORE_DRIVE_TEXT_FIELD: "This is a sample core drive text.",
                REACTION_PATTERN_TEXT_FIELD: "This is a sample reaction pattern text.",
                "messages": [] # Other fields as expected from analyzer
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

def save_to_yaml(filepath: str, data: list):
    """将数据列表保存到 YAML 文件。"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)
        print(f"YAML 数据已成功保存到 {filepath}")
    except Exception as e:
        print(f"保存 YAML 文件 {filepath} 时发生错误: {e}")

# --- Core Clustering Logic (Modified) ---
def perform_embedding_and_clustering(
    analyzed_data: list, 
    n_clusters_config: int,
    text_field_for_embedding: str,      # New: Which text field to use for embedding
    output_embedding_vector_field: str, # New: Field name to store embedding vector
    output_cluster_label_field: str     # New: Field name to store cluster label
) -> list:
    """
    对指定文本字段的数据进行 embedding 生成和 KMeans 聚类。
    结果（embedding 向量和聚类标签）将写入到原始数据项中指定的输出字段名下。
    """
    print(f"\n开始为字段 '{text_field_for_embedding}' 进行 Embedding 和聚类...")

    if not analyzed_data:
        print(f"没有分析数据可供基于字段 '{text_field_for_embedding}' 的 embedding 和聚类。")
        return []

    all_texts_for_embedding = []
    valid_data_indices = [] 

    for idx, conv_data in enumerate(analyzed_data):
        # Initialize output fields to ensure they exist
        conv_data[output_embedding_vector_field] = None
        conv_data[output_cluster_label_field] = -1 # Default error/skipped label

        if "analysis_error" in conv_data: # General error check, might be refined
            print(f"对话 {conv_data.get('conversation_id', 'N/A')} 存在分析错误，跳过字段 '{text_field_for_embedding}' 的 embedding。")
            continue
        
        description = conv_data.get(text_field_for_embedding)
        if not description:
            print(f"警告: 对话 {conv_data.get('conversation_id', 'N/A')} 的字段 '{text_field_for_embedding}' 为空，跳过 embedding。")
            continue
        all_texts_for_embedding.append(description)
        valid_data_indices.append(idx)

    if not all_texts_for_embedding:
        print(f"字段 '{text_field_for_embedding}' 没有有效的文本可用于 embedding。")
        return analyzed_data # analyzed_data already has fields initialized

    print(f"\n正在为 {len(all_texts_for_embedding)} 条来自字段 '{text_field_for_embedding}' 的有效文本生成 embeddings...")
    embeddings_list_of_lists = None
    embeddings_np = np.array([])
    try:
        embeddings_list_of_lists = get_embeddings_from_api(all_texts_for_embedding)
        if embeddings_list_of_lists is None:
            raise Exception("Failed to retrieve embeddings from API (returned None).")
        
        if not embeddings_list_of_lists and all_texts_for_embedding:
            print(f"警告: Embedding API 为字段 '{text_field_for_embedding}' 的非空输入返回了空列表。")
        
        if embeddings_list_of_lists:
            embeddings_np = np.array(embeddings_list_of_lists)
            print(f"Embeddings (for '{text_field_for_embedding}') 生成完成. Shape: {embeddings_np.shape}")
        else:
            print(f"没有为字段 '{text_field_for_embedding}' 生成 embeddings，或者API返回了空列表。")

    except Exception as e:
        print(f"错误：为字段 '{text_field_for_embedding}' 生成 embeddings 时出错: {e}")
        print(f"将不进行基于 '{text_field_for_embedding}' 的聚类。")
        # valid_data_indices items already have cluster label -1
        return analyzed_data

    embedding_source_idx = 0
    for target_idx in valid_data_indices:
        if embeddings_list_of_lists and embedding_source_idx < len(embeddings_list_of_lists):
            analyzed_data[target_idx][output_embedding_vector_field] = embeddings_list_of_lists[embedding_source_idx]
            embedding_source_idx += 1
        else:
            print(f"警告: Embedding 数量与有效文本数量不匹配 (字段 '{text_field_for_embedding}')。对话 ID: {analyzed_data[target_idx].get('conversation_id', 'N/A')} 将没有 embedding。")
            # Already set to None and -1
    
    if embeddings_np.shape[0] == 0:
        print(f"没有可用的 embeddings (来自字段 '{text_field_for_embedding}') 进行聚类。")
    elif embeddings_np.shape[0] < n_clusters_config:
        print(f"警告: 数据点数量 ({embeddings_np.shape[0]}) 少于聚类数量 ({n_clusters_config}) (字段 '{text_field_for_embedding}')。将不会进行聚类。")
    else:
        print(f"\n对 {embeddings_np.shape[0]} 个 embeddings (来自 '{text_field_for_embedding}') 进行 KMeans 聚类 (k={n_clusters_config})...")
        try:
            kmeans = KMeans(n_clusters=n_clusters_config, random_state=42, n_init='auto')
            cluster_labels_for_embedded_items = kmeans.fit_predict(embeddings_np)
            print(f"聚类 (为字段 '{text_field_for_embedding}') 完成。")

            label_idx = 0
            for original_data_idx in valid_data_indices:
                if analyzed_data[original_data_idx].get(output_embedding_vector_field) is not None:
                    if label_idx < len(cluster_labels_for_embedded_items):
                        analyzed_data[original_data_idx][output_cluster_label_field] = int(cluster_labels_for_embedded_items[label_idx])
                        label_idx += 1
                    else:
                        print(f"警告: 聚类标签索引越界 (字段 '{text_field_for_embedding}')。对话 ID: {analyzed_data[original_data_idx].get('conversation_id', 'N/A')}")
                        analyzed_data[original_data_idx][output_cluster_label_field] = -1 # Error label
        except Exception as e:
            print(f"错误：进行聚类时出错 (字段 '{text_field_for_embedding}'): {e}")
            for original_data_idx in valid_data_indices:
                if analyzed_data[original_data_idx].get(output_embedding_vector_field) is not None:
                    analyzed_data[original_data_idx][output_cluster_label_field] = -2 # Clustering error
    
    print(f"\nEmbedding 和聚类处理 (为字段 '{text_field_for_embedding}') 完成。")
    return analyzed_data

# --- YAML Output Generation ---
def generate_yaml_output_structure(
    all_conversations_data: list, 
    text_field_for_description: str, 
    cluster_label_field: str, 
    num_clusters: int, 
    cluster_type_name_prefix: str
) -> list:
    """
    为指定类型的聚类结果生成 YAML 输出结构。
    会尝试使用 LLM 对每个簇的描述进行总结和命名。
    """
    print(f"\n开始为 '{cluster_type_name_prefix}' (基于字段 '{cluster_label_field}') 生成 YAML 结构...")
    output_list = []

    # LLM Prompt Template for summarization and naming
    llm_summarization_prompt_template = """
    请仔细分析以下关于"{cluster_type}"的文本集合。这些文本描述了同一个聚类中的多个独立观察。
    你的任务是：
    1. 为这个聚类总结出一个高度概括、精准且能体现其核心特征的描述。请着重强调这个聚类与其他可能存在的聚类之间的潜在差异点。
    2. 基于此描述，为这个聚类起一个简洁、独特且具有高辨识度的中文名称。目标是让这个名称能清晰地与其他聚类的名称区分开。

    请以JSON格式返回结果，确保只包含两个键：
    - "name": (字符串) 你为这个聚类起的独特中文名称。
    - "description": (字符串) 你总结的体现核心特征与差异点的中文描述。

    待分析的文本集合如下（每个文本由 "---" 分隔，只选取了部分代表性文本）：
    {texts_to_summarize}

    确保返回的是一个有效的JSON对象，并且内容具有良好的区分度。
    """

    if not all_conversations_data:
        print(f"警告: 没有对话数据可供为 '{cluster_type_name_prefix}' 生成 YAML。")
        # Even with no data, create structure if num_clusters is positive
        for i in range(num_clusters):
             output_list.append({
                "name": f"{cluster_type_name_prefix} Cluster {i} (No Data)", 
                "description": "无可用数据或所有成员描述为空。"
            })
        if not output_list: # If num_clusters was 0 or negative
            return [{"name": f"{cluster_type_name_prefix} Cluster (No Data)", "description": "无可用数据。"}]
        return output_list


    for i in range(num_clusters):
        cluster_members = [
            conv for conv in all_conversations_data 
            if isinstance(conv.get(cluster_label_field), int) and conv.get(cluster_label_field) == i
        ]
        
        name = f"{cluster_type_name_prefix} Cluster {i}" # Default name
        description = f"聚类 {i} ({cluster_type_name_prefix}) 为空或无有效描述。" # Default description

        if cluster_members:
            member_descriptions = [
                member.get(text_field_for_description) 
                for member in cluster_members 
                if member.get(text_field_for_description) # Ensure description is not None or empty
            ]

            if member_descriptions:
                # Limit the number of descriptions to pass to LLM to avoid overly long prompts
                # and to manage API costs/rate limits.
                sample_size = min(len(member_descriptions), 10) # Take up to 10 random samples
                descriptions_for_llm = random.sample(member_descriptions, sample_size)
                texts_to_summarize_str = "\n\n---\n\n".join(descriptions_for_llm)
                
                llm_input_prompt = llm_summarization_prompt_template.format(
                    cluster_type=cluster_type_name_prefix,
                    texts_to_summarize=texts_to_summarize_str
                )
                
                print(f"聚类 {i} ({cluster_type_name_prefix}): 尝试使用 LLM 进行总结和命名 (基于 {len(descriptions_for_llm)} 条描述)...")
                llm_response = call_llm_api(llm_input_prompt, expect_json=True)

                if llm_response and isinstance(llm_response, dict) and "name" in llm_response and "description" in llm_response:
                    name = llm_response["name"]
                    description = llm_response["description"]
                    print(f"聚类 {i} ({cluster_type_name_prefix}): LLM 成功 - 名称='{name}', 描述='{description[:100].replace('\n', ' ')}...'")
                else:
                    print(f"警告: 聚类 {i} ({cluster_type_name_prefix}) 的 LLM 总结失败或返回格式不正确。将使用随机成员的描述。")
                    # Fallback: use a random member's description if LLM fails
                    random_member = random.choice(cluster_members) # cluster_members is guaranteed non-empty here
                    desc_text = random_member.get(text_field_for_description)
                    if desc_text: # This should be true if member_descriptions was non-empty
                        description = desc_text
                    else: # Should ideally not happen if member_descriptions was populated
                        description = f"聚类 {i} ({cluster_type_name_prefix}) 的随机选择成员描述为空。"
                    name = f"{cluster_type_name_prefix} Cluster {i} (Fallback)" 
            else: # No non-empty descriptions in cluster_members
                print(f"警告: 聚类 {i} ({cluster_type_name_prefix}) 的所有成员描述均为空。")
                description = f"聚类 {i} ({cluster_type_name_prefix}) 的所有成员描述均为空。"
                name = f"{cluster_type_name_prefix} Cluster {i} (No Descriptions)"
        else: # cluster_members is empty
             print(f"警告: 聚类 {i} ({cluster_type_name_prefix}) 为空。")
             # Name and description already set to default for empty/no-description clusters

        output_list.append({
            "name": name,
            "description": description
        })
    
    print(f"YAML 结构 (为 '{cluster_type_name_prefix}') 生成完毕。包含 {len(output_list)} 个条目。")
    return output_list

# --- Main Execution (Standalone Test) ---
if __name__ == '__main__':
    # This is for testing the clusterer module independently
    # DUMMY_ANALYZED_INPUT_FILE = "enriched_conversations_output.json" # Replaced by ANALYZED_INPUT_FILE
    # SUMMARIZED_CLUSTERS_OUTPUT_FILE = "generated_reaction_patterns_and_core_drives.json" # This was old, specific YAMLs are used.
    N_CLUSTERS_TEST = 10 # Reduced for quicker testing, can be 10

    print(f"独立测试 Clustering 模块 (双重聚类并输出为YAML)...")
    
    # --- Embedding Connectivity Check (Required for testing this module standalone) ---
    print("\nStandalone Clusterer Test: Checking Embedding (DeepInfra) connectivity...")
    # Ensure llm_utils.py is accessible
    embedding_conn_ok = check_embedding_connectivity(verbose=True)
    if not embedding_conn_ok:
        print("Embedding (DeepInfra) connectivity check failed. Clusterer module test might not produce meaningful embeddings.")
        print("Please ensure your .env file is configured and llm_utils.py is accessible.")
    else:
        print("Embedding (DeepInfra) connectivity OK for clusterer test.")
    
    # --- LLM Connectivity Check ---
    print("\nStandalone Clusterer Test: Checking LLM connectivity...")
    llm_conn_ok = check_llm_connectivity(verbose=True)
    if not llm_conn_ok:
        print("LLM connectivity check failed. Summarization and naming by LLM might not work.")
        print("Please ensure your .env file is configured for the LLM provider.")
    else:
        print("LLM connectivity OK for clusterer test.")
    # --------------------------------------------------------------------------

    print(f"加载分析后数据从: {ANALYZED_INPUT_FILE}") # Use ANALYZED_INPUT_FILE
    test_analyzed_data = load_json_data_for_clustering(ANALYZED_INPUT_FILE) # Use ANALYZED_INPUT_FILE

    if test_analyzed_data:
        # --- Perform Core Drives Clustering ---
        print(f"\n=== 开始核心驱动 (Core Drives) 的 Embedding 和 Clustering (k={N_CLUSTERS_TEST}) ===")
        # Note: perform_embedding_and_clustering modifies test_analyzed_data in place
        test_analyzed_data_after_cd = perform_embedding_and_clustering(
            test_analyzed_data, 
            N_CLUSTERS_TEST,
            text_field_for_embedding=CORE_DRIVE_TEXT_FIELD,
            output_embedding_vector_field="core_drive_embedding_vector", # New field
            output_cluster_label_field="core_drive_cluster_label"      # New field
        )
        
        # --- Perform Reaction Patterns Clustering ---
        print(f"\n=== 开始反应模式 (Reaction Patterns) 的 Embedding 和 Clustering (k={N_CLUSTERS_TEST}) ===")
        # Pass the (potentially modified by previous step) data again
        test_analyzed_data_after_rp = perform_embedding_and_clustering(
            test_analyzed_data_after_cd, # Use data from previous step
            N_CLUSTERS_TEST,
            text_field_for_embedding=REACTION_PATTERN_TEXT_FIELD,
            output_embedding_vector_field="reaction_pattern_embedding_vector", # New field
            output_cluster_label_field="reaction_pattern_cluster_label"      # New field
        )

        print(f"\nClustering 测试完成，对话结果数量: {len(test_analyzed_data_after_rp)}")

        # --- Generate and Save Core Drives YAML ---
        core_drives_yaml_structure = generate_yaml_output_structure(
            test_analyzed_data_after_rp,
            text_field_for_description=CORE_DRIVE_TEXT_FIELD,
            cluster_label_field="core_drive_cluster_label",
            num_clusters=N_CLUSTERS_TEST,
            cluster_type_name_prefix="Core Drive"
        )
        save_to_yaml(CORE_DRIVES_YAML_OUTPUT_FILE, core_drives_yaml_structure)

        # --- Generate and Save Reaction Patterns YAML ---
        reaction_patterns_yaml_structure = generate_yaml_output_structure(
            test_analyzed_data_after_rp,
            text_field_for_description=REACTION_PATTERN_TEXT_FIELD,
            cluster_label_field="reaction_pattern_cluster_label",
            num_clusters=N_CLUSTERS_TEST,
            cluster_type_name_prefix="Reaction Pattern"
        )
        save_to_yaml(REACTION_PATTERNS_YAML_OUTPUT_FILE, reaction_patterns_yaml_structure)
        
        print("\n独立测试运行完毕。请检查生成的 YAML 文件。")
        # print("\nFinal data state (sample):")
        # for i, item in enumerate(test_analyzed_data_after_rp[:2]): # Print first 2 items
        #     print(f"Item {i}: {json.dumps(item, indent=2, ensure_ascii=False)}")

    else:
        print(f"无法加载分析数据从 {ANALYZED_INPUT_FILE}") # Use ANALYZED_INPUT_FILE 