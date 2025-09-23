# services/prompt_service.py

import os
from typing import Dict, Optional
from services.file_service import FileService
from config import settings

class PromptService:
    """
    提示词服务类
    负责从用户目录加载提示词，并将编辑后的提示词保存回文件。
    """

    # 定义提示词的键名和对应的文件名映射
    # 这个映射必须与 settings.PROMPT_FILE_NAMES 保持一致
    PROMPT_KEYS = [
        "question_prompt_cn",   # 提问提示词（中）
        "question_prompt_en",   # 提问提示词（英）
        "followup_prompt_cn",   # 追问提示词（中）
        "followup_prompt_en",   # 追问提示词（英）
        "answer_prompt_cn",     # 回答提示词（中）
        "answer_prompt_en",     # 回答提示词（英）
        "rewrite_prompt_cn",    # 改写提示词（中）
        "rewrite_prompt_en",    # 改写提示词（英）
        "question_filter_prompt", # 问题过滤提示词（中）
        "answer_filter_prompt",  # 回答过滤提示词（中）
        "qa_score_prompt",      # 质量打分提示词（中）
    ]

    @classmethod
    def load_prompts(cls, user_prompts_path: str) -> Dict[str, str]:
        """
        从用户提示词目录加载所有提示词。
        Returns:
            Dict[str, str]: 键为提示词键名，值为提示词内容。
        """
        prompts = {}
        for prompt_key in cls.PROMPT_KEYS:
            # 从 settings 获取实际的文件名
            filename = settings.PROMPT_FILE_NAMES.get(prompt_key, f"{prompt_key}.txt")
            file_path = os.path.join(user_prompts_path, filename)
            content = FileService.read_file(file_path)
            if content is None:
                content = f"# {filename} - 文件不存在或读取失败，请检查。"
            prompts[prompt_key] = content
        return prompts

    @classmethod
    def save_prompts(cls, user_prompts_path: str, prompts: Dict[str, str]) -> bool:
        """
        将编辑后的提示词保存回用户目录。
        Args:
            user_prompts_path: 用户的提示词目录路径。
            prompts: 包含所有提示词键值对的字典。
        Returns:
            bool: 全部保存成功返回 True，任一失败返回 False。
        """
        all_success = True
        for prompt_key, content in prompts.items():
            filename = settings.PROMPT_FILE_NAMES.get(prompt_key, f"{prompt_key}.txt")
            file_path = os.path.join(user_prompts_path, filename)
            success = FileService.write_file(file_path, content)
            if not success:
                all_success = False
        return all_success

    @classmethod
    def get_prompt_key_display_name(cls, prompt_key: str) -> str:
        """
        获取提示词键的显示名称，用于UI展示。
        """
        display_names = {
            "question_prompt_cn": "提问提示词（中文）",
            "question_prompt_en": "提问提示词（英文）",
            "followup_prompt_cn": "追问提示词（中文）",
            "followup_prompt_en": "追问提示词（英文）",
            "answer_prompt_cn": "回答提示词（中文）",
            "answer_prompt_en": "回答提示词（英文）",
            "rewrite_prompt_cn": "改写提示词（中文）",
            "rewrite_prompt_en": "改写提示词（英文）",
            "question_filter_prompt": "问题过滤提示词（中文）",
            "answer_filter_prompt": "回答过滤提示词（中文）",
            "qa_score_prompt": "质量打分提示词（中文）",
        }
        return display_names.get(prompt_key, prompt_key)
