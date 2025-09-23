# models/user.py

import os
import shutil
from typing import Optional
from config import settings

class User:
    """
    用户模型类，封装用户信息及其所有相关数据路径。
    负责在用户注册时创建文件夹结构并复制初始文件。
    """

    def __init__(self, username: str):
        self.username = username
        # 核心路径
        self.data_path = os.path.join(settings.USER_DATA_ROOT_PATH, username)
        self.prompts_path = os.path.join(self.data_path, settings.PROMPTS_DIR_NAME)
        self.model_config_path = os.path.join(self.data_path, settings.MODEL_CONFIG_FILE_NAME)
        self.data_generate_config_path = os.path.join(self.data_path, settings.DATA_GENERATE_CONFIG_FILE_NAME)

        # 数据集路径 (用户保存的数据集会放在这里)
        self.datasets_path = os.path.join(self.data_path, "datasets")
        os.makedirs(self.datasets_path, exist_ok=True)

        # 蒸馏任务输出路径
        self.distillation_output_path = os.path.join(self.data_path, "distillation_output")
        os.makedirs(self.distillation_output_path, exist_ok=True)

    def is_admin(self) -> bool:
        """判断当前用户是否为管理员"""
        return self.username == settings.ADMIN_USERNAME

    def create_user_directory(self) -> bool:
        """
        为新用户创建目录结构，并从admin目录复制初始文件。
        如果用户是admin，则直接创建其目录（不复制）。
        Returns:
            bool: 创建成功返回True，失败返回False
        """
        try:
            # 创建用户主目录
            os.makedirs(self.data_path, exist_ok=True)
            # 创建prompts子目录
            os.makedirs(self.prompts_path, exist_ok=True)

            if not self.is_admin():
                # 如果不是admin，则从admin目录复制初始文件
                self._copy_initial_files()
            else:
                # 如果是admin，创建空的初始文件（实际部署时，您会手动放入）
                self._create_empty_initial_files_for_admin()

            return True
        except Exception as e:
            print(f"创建用户 {self.username} 的目录时出错: {e}")
            return False

    def _copy_initial_files(self):
        """从admin目录复制初始文件到新用户目录"""
        # 复制所有提示词文件
        for prompt_key, prompt_filename in settings.PROMPT_FILE_NAMES.items():
            src_file = os.path.join(settings.INITIAL_DATA_PATH, settings.PROMPTS_DIR_NAME, prompt_filename)
            dst_file = os.path.join(self.prompts_path, prompt_filename)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
            else:
                # 如果源文件不存在，创建一个空文件，避免程序报错
                with open(dst_file, 'w', encoding='utf-8') as f:
                    f.write(f"# {prompt_filename} - Initial template for user {self.username}")

        # 复制模型配置文件
        src_model_config = os.path.join(settings.INITIAL_DATA_PATH, settings.MODEL_CONFIG_FILE_NAME)
        if os.path.exists(src_model_config):
            shutil.copy2(src_model_config, self.model_config_path)
        else:
            with open(self.model_config_path, 'w', encoding='utf-8') as f:
                f.write("# model_config.yaml - Initial config for user\nmodels: []\n")

        # 复制数据生成配置文件
        src_data_gen_config = os.path.join(settings.INITIAL_DATA_PATH, settings.DATA_GENERATE_CONFIG_FILE_NAME)
        if os.path.exists(src_data_gen_config):
            shutil.copy2(src_data_gen_config, self.data_generate_config_path)
        else:
            with open(self.data_generate_config_path, 'w', encoding='utf-8') as f:
                f.write("# data_generate_config.yaml - Initial config for user\n")

    def _create_empty_initial_files_for_admin(self):
        """为admin用户创建空的初始文件结构"""
        # 创建空的提示词文件
        for prompt_filename in settings.PROMPT_FILE_NAMES.values():
            file_path = os.path.join(self.prompts_path, prompt_filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# {prompt_filename} - Admin's template file. Please fill in the content.\n")

        # 创建空的配置文件
        with open(self.model_config_path, 'w', encoding='utf-8') as f:
            f.write("# Admin's model_config.yaml\nmodels: []\n")

        with open(self.data_generate_config_path, 'w', encoding='utf-8') as f:
            f.write("# Admin's data_generate_config.yaml\n")

    def get_dataset_path(self, dataset_name: str) -> str:
        """获取指定数据集的完整路径"""
        return os.path.join(self.datasets_path, dataset_name)

    def list_datasets(self) -> list:
        """列出用户所有已保存的数据集名称（不包括子文件夹）"""
        datasets = []
        for item in os.listdir(self.datasets_path):
            item_path = os.path.join(self.datasets_path, item)
            if os.path.isdir(item_path):  # 假设每个数据集是一个文件夹
                datasets.append(item)
        return datasets

    def __repr__(self):
        return f"User(username='{self.username}', data_path='{self.data_path}')"
