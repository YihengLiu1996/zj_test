# services/file_service.py

import os
import json
import shutil
from typing import List, Dict, Any, Optional
from config import settings

class FileService:
    """
    文件系统服务类
    提供安全、原子化的文件和目录操作。
    """

    @staticmethod
    def read_file(file_path: str, encoding: str = 'utf-8') -> Optional[str]:
        """
        读取文件内容。
        Returns:
            str: 文件内容，如果文件不存在或读取失败，返回 None。
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return None

    @staticmethod
    def write_file(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
        """
        写入文件内容。
        如果目录不存在，会自动创建。
        Returns:
            bool: 写入成功返回 True，失败返回 False。
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"写入文件 {file_path} 时出错: {e}")
            return False

    @staticmethod
    def read_json_file(file_path: str) -> Optional[Dict]:
        """
        读取JSON文件。
        Returns:
            dict: 解析后的JSON对象，如果失败，返回 None。
        """
        content = FileService.read_file(file_path)
        if content is None:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"解析JSON文件 {file_path} 时出错: {e}")
            return None

    @staticmethod
    def write_json_file(file_path: str, data: Dict, indent: int = 2) -> bool:
        """
        写入JSON文件。
        Returns:
            bool: 写入成功返回 True，失败返回 False。
        """
        try:
            json_str = json.dumps(data, ensure_ascii=False, indent=indent)
            return FileService.write_file(file_path, json_str)
        except Exception as e:
            print(f"序列化JSON到文件 {file_path} 时出错: {e}")
            return False

    @staticmethod
    def read_jsonl_file(file_path: str) -> List[Dict]:
        """
        读取JSONL文件，返回一个字典列表。
        如果某行解析失败，会跳过该行并打印错误。
        Returns:
            List[Dict]: 解析成功的数据列表。
        """
        data_list = []
        if not os.path.exists(file_path):
            return data_list

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        data_list.append(data)
                    except json.JSONDecodeError as e:
                        print(f"解析JSONL文件 {file_path} 第{line_num}行时出错: {e}")
        except Exception as e:
            print(f"读取JSONL文件 {file_path} 时出错: {e}")

        return data_list

    @staticmethod
    def write_jsonl_file(file_path: str, data_list: List[Dict]) -> bool:
        """
        将字典列表写入JSONL文件。
        Returns:
            bool: 写入成功返回 True，失败返回 False。
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                for data in data_list:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
            return True
        except Exception as e:
            print(f"写入JSONL文件 {file_path} 时出错: {e}")
            return False

    @staticmethod
    def split_jsonl_file(file_path: str, max_size: int = settings.DATASET_SHARD_SIZE) -> List[str]:
        """
        将一个大的JSONL文件按大小分片。
        每个分片不超过 max_size 字节。
        Returns:
            List[str]: 分片文件的路径列表。
        """
        if not os.path.exists(file_path):
            return []

        shard_files = []
        current_shard_size = 0
        current_shard_lines = []
        shard_index = 1

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dir_name = os.path.dirname(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_size = len(line.encode('utf-8'))
                    if current_shard_size + line_size > max_size and current_shard_lines:
                        # 保存当前分片
                        shard_file_name = f"{base_name}_part_{shard_index}.jsonl"
                        shard_file_path = os.path.join(dir_name, shard_file_name)
                        with open(shard_file_path, 'w', encoding='utf-8') as shard_f:
                            shard_f.writelines(current_shard_lines)
                        shard_files.append(shard_file_path)
                        # 重置
                        current_shard_lines = []
                        current_shard_size = 0
                        shard_index += 1

                    current_shard_lines.append(line)
                    current_shard_size += line_size

                # 保存最后一个分片
                if current_shard_lines:
                    shard_file_name = f"{base_name}_part_{shard_index}.jsonl"
                    shard_file_path = os.path.join(dir_name, shard_file_name)
                    with open(shard_file_path, 'w', encoding='utf-8') as shard_f:
                        shard_f.writelines(current_shard_lines)
                    shard_files.append(shard_file_path)

            # 删除原始文件（可选，根据需求决定是否保留）
            # os.remove(file_path)

        except Exception as e:
            print(f"分片JSONL文件 {file_path} 时出错: {e}")
            return []

        return shard_files

    @staticmethod
    def list_files_in_directory(directory: str, extension: str = None) -> List[str]:
        """
        列出目录下的所有文件。
        Args:
            directory: 目录路径。
            extension: 文件扩展名过滤器（例如 '.jsonl'），如果为None，则返回所有文件。
        Returns:
            List[str]: 文件名列表。
        """
        if not os.path.exists(directory):
            return []

        files = []
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                if extension is None or item.endswith(extension):
                    files.append(item)
        return files

    @staticmethod
    def copy_file(src: str, dst: str) -> bool:
        """复制文件"""
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            print(f"复制文件 {src} 到 {dst} 时出错: {e}")
            return False

    @staticmethod
    def copy_directory(src: str, dst: str) -> bool:
        """复制整个目录"""
        try:
            shutil.copytree(src, dst)
            return True
        except Exception as e:
            print(f"复制目录 {src} 到 {dst} 时出错: {e}")
            return False

    @staticmethod
    def create_directory(path: str) -> bool:
        """创建目录"""
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            print(f"创建目录 {path} 时出错: {e}")
            return False
