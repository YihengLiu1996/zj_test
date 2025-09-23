# config/settings.py

import os

# ========================
# 全局硬编码配置
# ========================

# 用户数据根目录 (您后续会根据实际情况修改)
USER_DATA_ROOT_PATH = "/tmp/sft_data_distillation_users"  # 临时路径，您后续会修改

# 管理员账户信息
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "111111"

# 初始数据路径 (即admin用户的路径)
INITIAL_DATA_PATH = os.path.join(USER_DATA_ROOT_PATH, ADMIN_USERNAME)

# 初始文件结构常量
PROMPTS_DIR_NAME = "prompts"
MODEL_CONFIG_FILE_NAME = "model_config.yaml"
DATA_GENERATE_CONFIG_FILE_NAME = "data_generate_config.yaml"

# 提示词文件名映射 (用于确保复制和读取的文件名一致)
PROMPT_FILE_NAMES = {
    "question_prompt_cn": "question_prompt_cn.txt",
    "question_prompt_en": "question_prompt_en.txt",
    "followup_prompt_cn": "followup_prompt_cn.txt",
    "followup_prompt_en": "followup_prompt_en.txt",
    "answer_prompt_cn": "answer_prompt_cn.txt",  # 根据您的需求补充
    "answer_prompt_en": "answer_prompt_en.txt",
    "rewrite_prompt_cn": "rewrite_prompt_cn.txt",
    "rewrite_prompt_en": "rewrite_prompt_en.txt",
    "question_filter_prompt": "question_filter_prompt.txt",  # 中文
    "answer_filter_prompt": "answer_filter_prompt.txt",      # 中文
    "qa_score_prompt": "qa_score_prompt.txt",               # 中文
}

# 数据集分片大小 (10MB)
DATASET_SHARD_SIZE = 10 * 1024 * 1024  # 10 Megabytes

# 会话状态键名 (用于Streamlit Session State)
SESSION_KEY_LOGGED_IN = "logged_in"
SESSION_KEY_CURRENT_USER = "current_user"
SESSION_KEY_CURRENT_DATASET = "current_dataset"
SESSION_KEY_DISTILLATION_TASK_ID = "distillation_task_id"
