# K12 心理咨询模拟用户角色生成器

本项目旨在自动化创建大量多样化、真实且具有内在逻辑一致性的青少年学生角色画像（Personas）。这些画像主要用于模拟 K12 阶段（幼儿园到12年级）学生在心理咨询中的场景，可为人工智能模型的训练或心理咨询专业人士提供丰富的案例研究素材。

## 项目概述

该生成器通过组合不同的核心元素来构建角色：

*   **核心驱动 (Core Drives)**: 定义角色行为的内在动机和根本原因。
*   **困扰情境 (Situation)**: 角色面临的核心挑战，由预设的"主题"和"子主题"构成。
*   **反应模式 (Reaction Patterns)**: 角色在特定压力情境下的典型行为和情绪表现。
*   **角色职业 (Occupation)**: 包括小学生、初中生、高中生、老师、家长等。

利用大型语言模型（LLM，支持 OpenAI GPT 和 Google Gemini），脚本根据这些元素的组合以及详细的系统指令，生成包含角色背景故事、具体困扰情境描述、情感体验、应对策略及期望目标等的完整角色卡。

## 主要特性

*   **三维正交组合**: 通过"核心驱动 x 困扰情境 x 反应模式"的组合方式，最大化生成角色的多样性。
*   **策略性增强配置**: 允许开发者配置生成特定数量的稀有但关键的"边缘案例"，以提高生成数据的覆盖面和鲁棒性。
*   **模块化设计**: 核心驱动、反应模式及系统指令均通过外部文件（YAML 和 TXT）进行配置，方便扩展和维护。
*   **LLM 驱动的深度角色构建**: 每个角色都拥有由 LLM 生成的独特背景故事，确保其"历史感"和"灵魂"。
*   **JSON 输出**: 生成的角色卡以结构化的 JSON 格式输出，易于后续处理和使用。

## 项目结构

```
Simulator/
├── archetypes.py                   # 核心角色生成脚本
├── core_drives.yaml                # 定义角色核心驱动
├── reaction_patterns.yaml          # 定义角色反应模式 
├── system_prompt.txt               # LLM 系统指令模板
├── tools/
│   └── calculate_valid_combinations.py # 用于计算有效的角色元素组合
├── .env.example                    # 环境变量配置示例
├── .gitignore                      # Git忽略配置
└── README.md                       # 本文件
```

## 如何使用
0. **分析核心驱动及反应模式**：
   * 在之前数据分析+数据富集的基础上，使用大模型整理归纳出 `core drives` 以及 `reaction patterns`。

1.  **环境配置**:
    *   确保已安装 Python 3.x。
    *   安装所需依赖包：
        ```bash
        pip install PyYAML python-dotenv tqdm openai google-generativeai
        ```
    *   复制 `.env.example` 为 `.env` 文件。
    *   在 `.env` 文件中配置您的 LLM API 密钥及其他相关参数 (如 `LLM_PROVIDER`, `OPENAI_API_KEY`, `GEMINI_API_KEY` 等)。

2.  **配置文件检查**:
    *   确保 `core_drives.yaml`, `reaction_patterns.yaml` (或 `.txt`), 和 `system_prompt.txt` 文件与 `archetypes.py` 脚本在同一目录下。
    *   根据需要修改这些配置文件以调整角色生成的元素。

3.  **运行脚本**:
    *   打开 `archetypes.py` 脚本。
    *   在脚本末尾的 `if __name__ == "__main__":` 部分，根据需求配置：
        *   `TOTAL_PROFILES_TO_GENERATE`: 要生成的总角色画像数量。
        *   `STRATEGIC_CONFIGS`: （可选）策略性增强配置，用于生成特定类型的角色。
    *   执行脚本：
        ```bash
        python archetypes.py
        ```

4.  **查看结果**:
    *   生成的角色画像将保存在项目根目录下的 `generated_character_profiles_v3.json` 文件中 (文件名可能根据脚本版本有所不同)。

## 依赖

*   Python 3.x
*   PyYAML
*   python-dotenv
*   tqdm
*   openai
*   google-generativeai

## 注意

*   `.gitignore` 文件已配置为忽略 `.env` 和所有 `*.json` 文件，以保护敏感信息和避免将大量生成数据提交到版本库。