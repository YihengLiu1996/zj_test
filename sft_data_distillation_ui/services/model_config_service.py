# services/model_config_service.py

import os
import yaml
from typing import List, Dict, Any, Optional
from services.file_service import FileService

class ModelConfigService:
    """
    大模型配置服务类
    负责加载、保存和管理大模型配置。
    """

    @classmethod
    def load_model_config(cls, config_file_path: str) -> List[Dict[str, Any]]:
        """
        从YAML文件加载大模型配置。
        如果文件不存在或格式错误，返回一个空列表。
        Returns:
            List[Dict]: 模型配置列表。
        """
        if not os.path.exists(config_file_path):
            return []

        content = FileService.read_file(config_file_path)
        if content is None:
            return []

        try:
            config = yaml.safe_load(content)
            if config is None:
                return []
            # 确保返回的是一个列表
            if isinstance(config, dict) and 'models' in config:
                return config.get('models', [])
            elif isinstance(config, list):
                return config
            else:
                return []
        except yaml.YAMLError as e:
            print(f"解析YAML配置文件 {config_file_path} 时出错: {e}")
            return []

    @classmethod
    def save_model_config(cls, config_file_path: str, models: List[Dict[str, Any]]) -> bool:
        """
        将模型配置保存到YAML文件。
        Args:
            config_file_path: 配置文件路径。
            models: 模型配置列表。
        Returns:
            bool: 保存成功返回 True，失败返回 False。
        """
        config_data = {
            'models': models
        }
        try:
            yaml_str = yaml.dump(config_data, allow_unicode=True, default_flow_style=False, sort_keys=False)
            return FileService.write_file(config_file_path, yaml_str)
        except Exception as e:
            print(f"序列化YAML配置到文件 {config_file_path} 时出错: {e}")
            return False

    @classmethod
    def validate_model_config(cls, model_config: Dict[str, Any]) -> List[str]:
        """
        验证单个模型配置的完整性。
        Returns:
            List[str]: 错误信息列表，如果为空则表示验证通过。
        """
        errors = []
        required_fields = ['api_path', 'model_name', 'api_key', 'concurrency', 'model_type']
        for field in required_fields:
            if field not in model_config or not model_config[field]:
                errors.append(f"缺少必需字段: {field}")

        valid_model_types = ['thinking', 'instruct', 'mix']
        if model_config.get('model_type') not in valid_model_types:
            errors.append(f"model_type 必须是 {valid_model_types} 之一")

        if not isinstance(model_config.get('concurrency', 0), int) or model_config.get('concurrency', 0) <= 0:
            errors.append("concurrency 必须是正整数")

        return errors

    @classmethod
    def get_default_model_config(cls) -> Dict[str, Any]:
        """
        返回一个默认的模型配置模板。
        """
        return {
            "api_path": "https://api.example.com/v1",
            "model_name": "your-model-name",
            "api_key": "your-api-key-here",
            "concurrency": 5,
            "model_type": "thinking" # 默认为thinking
        }
