# utils/tokenizer_utils.py

from transformers import AutoTokenizer
from config import settings

class TokenizerManager:
    """
    Tokenizer管理器，用于计算文本的token长度。
    """

    _instance = None
    _tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._tokenizer is None:
            # 从配置或默认路径加载tokenizer
            # 这里我们使用一个默认路径，您可以根据实际情况修改
            tokenizer_path = "./qwen_model_fold"  # 请根据您的环境修改此路径
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            except Exception as e:
                raise RuntimeError(f"无法加载Tokenizer: {e}. 请确保模型文件在 {tokenizer_path}")

    def get_tokenizer(self):
        """获取tokenizer实例"""
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        if not text:
            return 0
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

# 创建一个全局可用的实例
tokenizer_manager = TokenizerManager()

# 为了方便使用，提供一个快捷函数
def count_tokens(text: str) -> int:
    return tokenizer_manager.count_tokens(text)
