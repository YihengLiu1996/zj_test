# models/dataset.py

import os
import json
import uuid
from typing import List, Dict, Any, Optional, Union
from services import file_service
from config import settings
from utils import tokenizer_utils, chunk_utils # 我们稍后会创建这些工具模块

class Dataset:
    """
    数据集模型类
    封装数据集的加载、查看、编辑和保存逻辑。
    """

    def __init__(self, name: str, user_data_path: str):
        """
        初始化数据集对象。
        Args:
            name: 数据集名称。
            user_data_path: 用户的数据根路径，用于构建数据集的保存路径。
        """
        self.name = name
        self.dataset_path = os.path.join(user_data_path, "datasets", name)
        os.makedirs(self.dataset_path, exist_ok=True)

        # 核心数据
        self.data_items: List[Dict] = []  # 存储所有数据项
        self.current_index: int = 0       # 当前查看的数据项索引

        # 加载状态
        self.is_loaded: bool = False
        self.source_type: str = ""        # "raw_markdown" 或 "saved_jsonl"
        self.source_path: str = ""        # 原始数据源路径

    def load_from_raw_markdown(self, markdown_dir: str, max_tokens: int, min_tokens: int) -> bool:
        """
        从原始Markdown目录加载数据。
        会调用数据预处理功能进行chunk处理。
        Returns:
            bool: 加载成功返回 True，失败返回 False。
        """
        if not os.path.exists(markdown_dir) or not os.path.isdir(markdown_dir):
            return False

        # 构建临时输出文件路径
        temp_output_file = os.path.join(self.dataset_path, "temp_preprocessed.jsonl")

        try:
            # 调用数据预处理工具 (utils.chunk_utils)
            # 注意：这里假设 chunk_utils.process_markdown_folder 是一个同步函数
            # 在实际实现中，您可能需要将其改为异步或在后台线程中执行，以避免阻塞UI。
            chunk_utils.process_markdown_folder(
                input_dir=markdown_dir,
                output_file_path=temp_output_file,
                max_tokens=max_tokens,
                min_tokens=min_tokens
            )

            # 读取预处理后的数据
            self.data_items = file_service.FileService.read_jsonl_file(temp_output_file)
            # 删除临时文件
            os.remove(temp_output_file)

            self.is_loaded = True
            self.source_type = "raw_markdown"
            self.source_path = markdown_dir
            self.current_index = 0

            return True

        except Exception as e:
            print(f"从原始Markdown加载数据集时出错: {e}")
            return False

    def load_from_saved_jsonl(self, dataset_name: str, user: 'User') -> bool: # 'User' 是前向引用
        """
        从用户已保存的数据集加载数据。
        Args:
            dataset_name: 要加载的数据集名称。
            user: 当前用户对象，用于获取其数据路径。
        Returns:
            bool: 加载成功返回 True，失败返回 False。
        """
        saved_dataset_path = user.get_dataset_path(dataset_name)
        if not os.path.exists(saved_dataset_path) or not os.path.isdir(saved_dataset_path):
            return False

        # 读取该目录下所有的JSONL分片文件
        all_data_items = []
        jsonl_files = file_service.FileService.list_files_in_directory(saved_dataset_path, ".jsonl")
        for jsonl_file in jsonl_files:
            file_path = os.path.join(saved_dataset_path, jsonl_file)
            items = file_service.FileService.read_jsonl_file(file_path)
            all_data_items.extend(items)

        if not all_data_items:
            return False

        self.data_items = all_data_items
        self.name = dataset_name # 更新当前数据集名称
        self.dataset_path = saved_dataset_path # 更新当前数据集路径
        self.is_loaded = True
        self.source_type = "saved_jsonl"
        self.source_path = saved_dataset_path
        self.current_index = 0

        return True

    def get_current_item(self) -> Optional[Dict]:
        """获取当前索引指向的数据项"""
        if not self.is_loaded or not self.data_items or self.current_index >= len(self.data_items):
            return None
        return self.data_items[self.current_index]

    def has_next(self) -> bool:
        """检查是否有下一条数据"""
        return self.is_loaded and self.current_index < len(self.data_items) - 1

    def has_prev(self) -> bool:
        """检查是否有上一条数据"""
        return self.is_loaded and self.current_index > 0

    def next_item(self) -> bool:
        """跳到下一条数据"""
        if self.has_next():
            self.current_index += 1
            return True
        return False

    def prev_item(self) -> bool:
        """跳到上一条数据"""
        if self.has_prev():
            self.current_index -= 1
            return True
        return False

    def jump_to_index(self, index: int) -> bool:
        """跳转到指定索引"""
        if self.is_loaded and 0 <= index < len(self.data_items):
            self.current_index = index
            return True
        return False

    def add_question_to_current_item(self, question: str) -> bool:
        """
        为当前数据项添加一个问题。
        根据需求，messages 字段初始只包含 user 角色的消息。
        Returns:
            bool: 添加成功返回 True，失败返回 False。
        """
        current_item = self.get_current_item()
        if current_item is None:
            return False

        if "messages" not in current_item:
            current_item["messages"] = []

        # 添加用户问题
        current_item["messages"].append({
            "role": "user",
            "content": question.strip()
        })

        return True

    def save_dataset(self, save_name: str, user: 'User') -> bool:
        """
        将当前数据集保存到用户目录。
        会自动分片，每个分片不超过10M。
        Args:
            save_name: 保存的数据集名称。
            user: 当前用户对象。
        Returns:
            bool: 保存成功返回 True，失败返回 False。
        """
        if not self.is_loaded or not self.data_items:
            return False

        save_path = user.get_dataset_path(save_name)
        os.makedirs(save_path, exist_ok=True)

        # 临时文件路径
        temp_file_path = os.path.join(save_path, "dataset_temp.jsonl")

        # 先将所有数据写入一个临时文件
        if not file_service.FileService.write_jsonl_file(temp_file_path, self.data_items):
            return False

        # 对临时文件进行分片
        shard_files = file_service.FileService.split_jsonl_file(temp_file_path)

        # 删除临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        if not shard_files:
            return False

        # 更新当前数据集的元信息
        self.name = save_name
        self.dataset_path = save_path

        return True

    def get_total_count(self) -> int:
        """获取数据集总条数"""
        return len(self.data_items) if self.is_loaded else 0

    def get_current_index_info(self) -> str:
        """获取当前索引信息，例如 '1 / 100'"""
        if not self.is_loaded:
            return "0 / 0"
        return f"{self.current_index + 1} / {self.get_total_count()}"

    def __repr__(self):
        return f"Dataset(name='{self.name}', loaded={self.is_loaded}, items={self.get_total_count()})"
