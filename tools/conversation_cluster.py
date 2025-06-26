import json
import numpy as np
# from sklearn.cluster import KMeans # Replaced by HDBSCAN
import os
import sys # Added sys
import random # Added for selecting random cluster member
import yaml # Added for YAML output
import umap # Added for UMAP
import hdbscan # Added for HDBSCAN

# Add the project root directory (one level up from the 'tools' directory) to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- LLM Utils Import ---
from tools.llm_utils import get_embeddings_from_api, check_embedding_connectivity, call_llm_api, check_llm_connectivity
# ------------------------

# --- Configuration for standalone testing ---
# N_CLUSTERS_DEFAULT = 10 # Default, can be overridden by orchestrator # No longer directly used for HDBSCAN k

# --- Constants for new text fields and output files ---
CORE_DRIVE_TEXT_FIELD = "core_drive_text_for_clustering"
REACTION_PATTERN_TEXT_FIELD = "reaction_pattern_text_for_clustering"

CORE_DRIVES_YAML_OUTPUT_FILE = "generated_core_drives.yaml"
REACTION_PATTERNS_YAML_OUTPUT_FILE = "generated_reaction_patterns.yaml"

# --- Clustering Algorithm Parameters ---
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.0
UMAP_N_COMPONENTS = 20 # Number of dimensions for UMAP reduction
UMAP_RANDOM_STATE = 42
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_CLUSTER_SELECTION_METHOD = 'leaf'
MIN_SAMPLES_FOR_CLUSTERING = HDBSCAN_MIN_CLUSTER_SIZE # Minimum samples to attempt HDBSCAN

# --- Configuration for standalone execution ---
ANALYZED_INPUT_FILE = "DifyLog_0613-中小学_plus_analyzed.json"
# N_CLUSTERS_TEST remains as a configurable parameter for testing. # No longer directly sets k for HDBSCAN
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
    # n_clusters_config: int, # No longer used as HDBSCAN determines k
    text_field_for_embedding: str,
    output_embedding_vector_field: str,
    output_cluster_label_field: str
) -> list:
    """
    对指定文本字段的数据进行 embedding 生成（如果需要）、UMAP降维和 HDBSCAN 聚类。
    如果已有 embedding 向量，则直接使用。
    结果（embedding 向量和聚类标签）将写入到原始数据项中指定的输出字段名下。
    Cluster labels:
    - Positive integers (0, 1, ...): Cluster assignment
    - -1: Noise point (from HDBSCAN)
    - -2: Clustering algorithm error (UMAP/HDBSCAN failed during clustering)
    - -10: Default initial state / Not yet processed by clustering specific to this run
    - -11: Skipped due to analysis_error from a previous step
    - -12: No text for embedding and no pre-existing embedding
    - -13: Embedding API error (when fetching new embedding)
    - -14: Embedding count mismatch from API (when fetching new embedding)
    """
    print(f"\n开始为字段 '{text_field_for_embedding}' 进行 Embedding (如果需要), UMAP 和 HDBSCAN 聚类...")

    if not analyzed_data:
        print(f"没有分析数据可供基于字段 '{text_field_for_embedding}' 的处理。")
        return []

    all_embeddings_for_clustering = []  # List to hold embeddings (as np.array) for actual clustering
    original_indices_for_clustering = [] # Original indices in analyzed_data for items in all_embeddings_for_clustering

    texts_to_fetch_embeddings_for = []
    original_indices_needing_new_embeddings = []

    for idx, conv_data in enumerate(analyzed_data):
        # Reset cluster label for this run; embedding vector is preserved if valid
        conv_data[output_cluster_label_field] = -10 # Default: "pending clustering this run"

        if "analysis_error" in conv_data:
            print(f"对话 {conv_data.get('conversation_id', 'N/A')} 存在分析错误，跳过字段 '{text_field_for_embedding}' 的处理。")
            conv_data[output_cluster_label_field] = -11 # Skipped due to prior analysis error
            continue

        existing_embedding = conv_data.get(output_embedding_vector_field)
        description_text = conv_data.get(text_field_for_embedding)

        is_existing_embedding_valid = False
        if isinstance(existing_embedding, (list, np.ndarray)) and len(existing_embedding) > 0:
            try:
                # Attempt to convert to NumPy array to ensure consistency and catch potential issues
                embedding_np_array = np.array(existing_embedding, dtype=float)
                if embedding_np_array.ndim == 1 and embedding_np_array.size > 0: # Assuming 1D embedding vector
                    is_existing_embedding_valid = True
                else:
                    print(f"警告: 对话 {conv_data.get('conversation_id', 'N/A')} 的预存 embedding 格式无效 (维度/大小: {embedding_np_array.ndim}/{embedding_np_array.size})。将尝试重新获取。")
                    conv_data[output_embedding_vector_field] = None # Invalidate it
            except ValueError:
                print(f"警告: 对话 {conv_data.get('conversation_id', 'N/A')} 的预存 embedding 无法转换为浮点数组。将尝试重新获取。")
                conv_data[output_embedding_vector_field] = None # Invalidate it


        if is_existing_embedding_valid:
            # print(f"对话 {conv_data.get('conversation_id', 'N/A')} 已有有效 embedding (字段 '{text_field_for_embedding}')，将直接使用。")
            all_embeddings_for_clustering.append(np.array(existing_embedding)) # Already validated as convertible
            original_indices_for_clustering.append(idx)
        else: # No valid existing embedding, try to fetch
            if not description_text:
                print(f"警告: 对话 {conv_data.get('conversation_id', 'N/A')} 的字段 '{text_field_for_embedding}' 为空，且无有效预存 embedding，跳过。")
                conv_data[output_embedding_vector_field] = None # Ensure it's None
                conv_data[output_cluster_label_field] = -12 # No text and no pre-existing embedding
                continue
            else:
                texts_to_fetch_embeddings_for.append(description_text)
                original_indices_needing_new_embeddings.append(idx)

    if texts_to_fetch_embeddings_for:
        print(f"\n正在为 {len(texts_to_fetch_embeddings_for)} 条来自字段 '{text_field_for_embedding}' 且无有效预存 embedding 的文本生成新的 embeddings...")
        newly_generated_embeddings_list = None
        try:
            newly_generated_embeddings_list = get_embeddings_from_api(texts_to_fetch_embeddings_for)
            if newly_generated_embeddings_list is None:
                raise Exception("Failed to retrieve embeddings from API (returned None).")
            if not newly_generated_embeddings_list and texts_to_fetch_embeddings_for:
                 print(f"警告: Embedding API 为字段 '{text_field_for_embedding}' 的非空输入返回了空列表。")


            new_embedding_source_idx = 0
            for target_idx in original_indices_needing_new_embeddings:
                if newly_generated_embeddings_list and new_embedding_source_idx < len(newly_generated_embeddings_list):
                    current_new_embedding = newly_generated_embeddings_list[new_embedding_source_idx]
                    analyzed_data[target_idx][output_embedding_vector_field] = current_new_embedding
                    all_embeddings_for_clustering.append(np.array(current_new_embedding))
                    original_indices_for_clustering.append(target_idx)
                    new_embedding_source_idx += 1
                else: # Mismatch or empty list from API for this item
                    print(f"警告: 为对话 ID {analyzed_data[target_idx].get('conversation_id', 'N/A')} (字段 '{text_field_for_embedding}') 生成 embedding 失败或返回空。")
                    analyzed_data[target_idx][output_embedding_vector_field] = None
                    analyzed_data[target_idx][output_cluster_label_field] = -14 # Embedding count mismatch / API return issue for item
            if new_embedding_source_idx > 0:
                 print(f"成功为 {new_embedding_source_idx} 个项目获取了新的 embeddings。")


        except Exception as e:
            print(f"错误：为字段 '{text_field_for_embedding}' 生成新 embeddings 时出错: {e}")
            for target_idx in original_indices_needing_new_embeddings: # Mark all that were attempted
                analyzed_data[target_idx][output_embedding_vector_field] = None
                analyzed_data[target_idx][output_cluster_label_field] = -13 # Embedding API error during batch fetch

    if not all_embeddings_for_clustering:
        print(f"字段 '{text_field_for_embedding}' 没有有效的文本或预存的 embeddings 可用于聚类。")
        return analyzed_data

    embeddings_np = np.array(all_embeddings_for_clustering)
    if embeddings_np.ndim == 1 and embeddings_np.shape[0] > 0 and isinstance(all_embeddings_for_clustering[0], (list, np.ndarray)):
        # This can happen if all_embeddings_for_clustering is a list of lists/arrays, and np.array tries to make it 2D
        # but if only one embedding, it might become 1D array of objects.
        # Or if it's a list of 1D arrays, np.array(list_of_1d_arrays) should produce a 2D array.
        # This check is a bit complex, primary goal is to ensure embeddings_np is 2D for UMAP.
        # A simpler approach is to check shape[0] and then MIN_SAMPLES_FOR_CLUSTERING.
        # If shape is (num_samples, num_features), it's good.
        # If shape is (num_samples,) it means it's likely a list of objects or something went wrong.
        # Let's assume get_embeddings_from_api and existing_embeddings are consistently lists of floats (vectors)
        pass # np.array should handle list of 1D lists/arrays correctly into a 2D array.

    print(f"总共收集到 {embeddings_np.shape[0]} 个 embeddings (预存或新生成) 用于字段 '{text_field_for_embedding}' 的聚类。Shape: {embeddings_np.shape}")

    if embeddings_np.shape[0] == 0: # Should be caught by 'if not all_embeddings_for_clustering'
        print(f"没有可用的 embeddings (来自字段 '{text_field_for_embedding}') 进行聚类。")
    elif embeddings_np.shape[0] < MIN_SAMPLES_FOR_CLUSTERING:
        print(f"警告: 有效 embedding 数量 ({embeddings_np.shape[0]}) 少于进行HDBSCAN聚类所需的最小数量 ({MIN_SAMPLES_FOR_CLUSTERING}) (字段 '{text_field_for_embedding}')。将不会进行聚类。")
        # For items that had embeddings, set their label to something like "not_clustered_due_to_low_sample_size"
        for original_data_idx in original_indices_for_clustering:
             analyzed_data[original_data_idx][output_cluster_label_field] = -16 # Not clustered (low sample size for overall batch)
    else:
        print(f"\n对 {embeddings_np.shape[0]} 个 embeddings (来自 '{text_field_for_embedding}') 进行 UMAP降维和HDBSCAN聚类...")
        try:
            print(f"正在应用 UMAP (n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, n_components={UMAP_N_COMPONENTS}) 到 {embeddings_np.shape} embeddings...")
            reducer = umap.UMAP(
                n_neighbors=min(UMAP_N_NEIGHBORS, embeddings_np.shape[0] -1), # n_neighbors must be less than n_samples
                min_dist=UMAP_MIN_DIST,
                n_components=min(UMAP_N_COMPONENTS, embeddings_np.shape[0] - 1) if embeddings_np.shape[0] > UMAP_N_COMPONENTS else UMAP_N_COMPONENTS, # n_components must be <= n_samples - 1 (for UMAP default metric)
                random_state=UMAP_RANDOM_STATE
            )
            # Ensure n_components is not greater than effective dimensionality after potential duplicates
            # UMAP handles n_components > n_features internally by reducing n_components.
            # However, n_neighbors > n_samples is an issue.
            
            # Adjust UMAP_N_COMPONENTS if it's too large for the number of samples
            actual_n_components = UMAP_N_COMPONENTS
            if embeddings_np.shape[0] <= UMAP_N_COMPONENTS :
                actual_n_components = max(1, embeddings_np.shape[0] - 1) # Ensure at least 1, and less than n_samples
                print(f"警告: UMAP n_components ({UMAP_N_COMPONENTS}) 大于等于样本数 ({embeddings_np.shape[0]})。调整 n_components 为 {actual_n_components}。")


            reducer = umap.UMAP(
                n_neighbors=min(UMAP_N_NEIGHBORS, embeddings_np.shape[0] -1 if embeddings_np.shape[0] > 1 else 1),
                min_dist=UMAP_MIN_DIST,
                n_components=actual_n_components,
                random_state=UMAP_RANDOM_STATE,
                # low_memory=True # Consider if dataset is very large
            )
            embedding_reduced = reducer.fit_transform(embeddings_np)
            print(f"UMAP 降维完成. 降维后 embedding 形状: {embedding_reduced.shape}")

            print(f"正在应用 HDBSCAN (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, method='{HDBSCAN_CLUSTER_SELECTION_METHOD}') 到 {embedding_reduced.shape} 降维后 embeddings...")
            
            actual_min_cluster_size = HDBSCAN_MIN_CLUSTER_SIZE
            if embedding_reduced.shape[0] < HDBSCAN_MIN_CLUSTER_SIZE:
                actual_min_cluster_size = max(2, embedding_reduced.shape[0]) # HDBSCAN min_cluster_size must be >= 2
                print(f"警告: HDBSCAN min_cluster_size ({HDBSCAN_MIN_CLUSTER_SIZE}) 大于等于样本数 ({embedding_reduced.shape[0]})。调整 min_cluster_size 为 {actual_min_cluster_size}。")
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=actual_min_cluster_size,
                min_samples=None, # If min_cluster_size is small, min_samples might also need adjustment or be left to default based on min_cluster_size
                cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
            )
            cluster_labels_for_embedded_items = clusterer.fit_predict(embedding_reduced)

            num_discovered_clusters = len(set(label for label in cluster_labels_for_embedded_items if label != -1))
            print(f"HDBSCAN 聚类 (为字段 '{text_field_for_embedding}') 完成。发现 {num_discovered_clusters} 个聚类 (噪音点用-1标记)。标签: {np.unique(cluster_labels_for_embedded_items)}")

            label_idx = 0
            for original_data_idx in original_indices_for_clustering:
                # This item was part of the clustering process
                if label_idx < len(cluster_labels_for_embedded_items):
                    analyzed_data[original_data_idx][output_cluster_label_field] = int(cluster_labels_for_embedded_items[label_idx])
                    label_idx += 1
                else:
                    print(f"警告: 聚类标签索引越界 (字段 '{text_field_for_embedding}')。对话 ID: {analyzed_data[original_data_idx].get('conversation_id', 'N/A')}")
                    analyzed_data[original_data_idx][output_cluster_label_field] = -2 # Clustering error (indexing issue)
        except Exception as e:
            print(f"错误：进行 UMAP/HDBSCAN 聚类时出错 (字段 '{text_field_for_embedding}'): {e}")
            for original_data_idx in original_indices_for_clustering: # Mark all that were intended for clustering
                analyzed_data[original_data_idx][output_cluster_label_field] = -2 # Clustering algorithm error

    print(f"\nEmbedding (如果需要), UMAP 和 HDBSCAN 聚类处理 (为字段 '{text_field_for_embedding}') 完成。")
    return analyzed_data

# --- YAML Output Generation ---
def generate_yaml_output_structure(
    all_conversations_data: list, 
    text_field_for_description: str, 
    cluster_label_field: str, 
    # num_clusters: int, # Replaced by unique_cluster_indices
    unique_cluster_indices: list[int], # List of actual cluster indices (e.g., [0, 1, 2, ...])
    cluster_type_name_prefix: str
) -> list:
    """
    为指定类型的聚类结果生成 YAML 输出结构。
    会尝试使用 LLM 对每个簇的描述进行总结和命名。
    Iterates over unique_cluster_indices (non-noise clusters).
    """
    print(f"\n开始为 '{cluster_type_name_prefix}' (基于字段 '{cluster_label_field}') 生成 YAML 结构 (使用 {len(unique_cluster_indices)} 个聚类)...")
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
        print(f"警告: 没有提供对话数据给 '{cluster_type_name_prefix}' 的 YAML 生成。")
        # If all_conversations_data is empty, unique_cluster_indices should also be empty.
        # The next check will handle this combined case.

    if not unique_cluster_indices: # Handles no data or no clusters found (e.g. all noise)
        print(f"警告: '{cluster_type_name_prefix}' (基于字段 '{cluster_label_field}') 未找到有效聚类或无数据可处理。")
        output_list.append({
            "name": f"{cluster_type_name_prefix} - No Clusters Found or No Data", 
            "description": "没有找到有效的聚类，或者没有数据进行聚类，或者所有数据点都被分类为噪音。"
        })
        return output_list


    for cluster_idx in unique_cluster_indices: # Iterate over actual discovered cluster indices
        cluster_members = [
            conv for conv in all_conversations_data 
            if isinstance(conv.get(cluster_label_field), int) and conv.get(cluster_label_field) == cluster_idx # Match specific cluster_idx
        ]
        
        name = f"{cluster_type_name_prefix} Cluster {cluster_idx}" # Default name using actual cluster_idx
        description = f"聚类 {cluster_idx} ({cluster_type_name_prefix}) 为空或无有效描述。" # Default description

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
                
                print(f"聚类 {cluster_idx} ({cluster_type_name_prefix}): 尝试使用 LLM 进行总结和命名 (基于 {len(descriptions_for_llm)} 条描述)...")
                llm_response = call_llm_api(llm_input_prompt, expect_json=True)

                if llm_response and isinstance(llm_response, dict) and "name" in llm_response and "description" in llm_response:
                    name = llm_response["name"]
                    description = llm_response["description"]
                    print(f"聚类 {cluster_idx} ({cluster_type_name_prefix}): LLM 成功 - 名称='{name}', 描述='{description[:100].replace('\n', ' ')}...'")
                else:
                    print(f"警告: 聚类 {cluster_idx} ({cluster_type_name_prefix}) 的 LLM 总结失败或返回格式不正确。将使用随机成员的描述。")
                    # Fallback: use a random member's description if LLM fails
                    random_member = random.choice(cluster_members) # cluster_members is guaranteed non-empty here
                    desc_text = random_member.get(text_field_for_description)
                    if desc_text: # This should be true if member_descriptions was non-empty
                        description = desc_text
                    else: # Should ideally not happen if member_descriptions was populated
                        description = f"聚类 {cluster_idx} ({cluster_type_name_prefix}) 的随机选择成员描述为空。"
                    name = f"{cluster_type_name_prefix} Cluster {cluster_idx} (Fallback)" 
            else: # No non-empty descriptions in cluster_members
                print(f"警告: 聚类 {cluster_idx} ({cluster_type_name_prefix}) 的所有成员描述均为空。")
                description = f"聚类 {cluster_idx} ({cluster_type_name_prefix}) 的所有成员描述均为空。"
                name = f"{cluster_type_name_prefix} Cluster {cluster_idx} (No Descriptions)"
        else: # cluster_members is empty (should not happen if cluster_idx is from unique_cluster_indices derived from data)
             print(f"警告: 聚类 {cluster_idx} ({cluster_type_name_prefix}) 为空。这不应该发生，如果 cluster_idx 来自有效的聚类。")
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
    # N_CLUSTERS_TEST = 10 # No longer directly used to set k for HDBSCAN

    print(f"独立测试 Clustering 模块 (双重聚类 - UMAP+HDBSCAN 并输出为YAML)...")
    
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
        print(f"\n=== 开始核心驱动 (Core Drives) 的 Embedding, UMAP 和 HDBSCAN Clustering ===")
        # Note: perform_embedding_and_clustering modifies test_analyzed_data in place
        test_analyzed_data_after_cd = perform_embedding_and_clustering(
            test_analyzed_data, 
            # N_CLUSTERS_TEST, # Removed, HDBSCAN determines k
            text_field_for_embedding=CORE_DRIVE_TEXT_FIELD,
            output_embedding_vector_field="core_drive_embedding_vector", 
            output_cluster_label_field="core_drive_cluster_label"      
        )
        
        # --- Perform Reaction Patterns Clustering ---
        print(f"\n=== 开始反应模式 (Reaction Patterns) 的 Embedding, UMAP 和 HDBSCAN Clustering ===")
        # Pass the (potentially modified by previous step) data again
        test_analyzed_data_after_rp = perform_embedding_and_clustering(
            test_analyzed_data_after_cd, # Use data from previous step
            # N_CLUSTERS_TEST, # Removed
            text_field_for_embedding=REACTION_PATTERN_TEXT_FIELD,
            output_embedding_vector_field="reaction_pattern_embedding_vector", 
            output_cluster_label_field="reaction_pattern_cluster_label"      
        )

        print(f"\nClustering 测试完成，对话结果数量: {len(test_analyzed_data_after_rp)}")

        # --- Generate and Save Core Drives YAML ---
        core_drive_labels = [
            d.get("core_drive_cluster_label", -1) 
            for d in test_analyzed_data_after_rp 
            if isinstance(d.get("core_drive_cluster_label"), int)
        ]
        unique_core_drive_clusters = sorted([label for label in set(core_drive_labels) if label != -1])
        print(f"为核心驱动发现 {len(unique_core_drive_clusters)} 个有效聚类: {unique_core_drive_clusters}")

        core_drives_yaml_structure = generate_yaml_output_structure(
            test_analyzed_data_after_rp,
            text_field_for_description=CORE_DRIVE_TEXT_FIELD,
            cluster_label_field="core_drive_cluster_label",
            unique_cluster_indices=unique_core_drive_clusters, # Pass actual unique cluster indices
            cluster_type_name_prefix="Core Drive"
        )
        save_to_yaml(CORE_DRIVES_YAML_OUTPUT_FILE, core_drives_yaml_structure)

        # --- Generate and Save Reaction Patterns YAML ---
        reaction_pattern_labels = [
            d.get("reaction_pattern_cluster_label", -1) 
            for d in test_analyzed_data_after_rp 
            if isinstance(d.get("reaction_pattern_cluster_label"), int)
        ]
        unique_reaction_pattern_clusters = sorted([label for label in set(reaction_pattern_labels) if label != -1])
        print(f"为反应模式发现 {len(unique_reaction_pattern_clusters)} 个有效聚类: {unique_reaction_pattern_clusters}")
        
        reaction_patterns_yaml_structure = generate_yaml_output_structure(
            test_analyzed_data_after_rp,
            text_field_for_description=REACTION_PATTERN_TEXT_FIELD,
            cluster_label_field="reaction_pattern_cluster_label",
            unique_cluster_indices=unique_reaction_pattern_clusters, # Pass actual unique cluster indices
            cluster_type_name_prefix="Reaction Pattern"
        )
        save_to_yaml(REACTION_PATTERNS_YAML_OUTPUT_FILE, reaction_patterns_yaml_structure)
        
        print("\n独立测试运行完毕。请检查生成的 YAML 文件。")
        # print("\nFinal data state (sample):")
        # for i, item in enumerate(test_analyzed_data_after_rp[:2]): # Print first 2 items
        #     print(f"Item {i}: {json.dumps(item, indent=2, ensure_ascii=False)}")

    else:
        print(f"无法加载分析数据从 {ANALYZED_INPUT_FILE}") # Use ANALYZED_INPUT_FILE 