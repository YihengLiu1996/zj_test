# services/auth_service.py

import os
import hashlib
from typing import Optional
from models.user import User
from config import settings

class AuthService:
    """
    用户认证服务类
    负责处理登录、注册、密码验证等核心认证逻辑。
    """

    @staticmethod
    def hash_password(password: str) -> str:
        """
        对密码进行简单的哈希处理（在实际生产环境中应使用更安全的库如 bcrypt）
        这里为了简化，使用SHA256。
        """
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    @classmethod
    def verify_password(cls, plain_password: str, hashed_password: str) -> bool:
        """验证明文密码与哈希密码是否匹配"""
        return cls.hash_password(plain_password) == hashed_password

    @classmethod
    def authenticate_user(cls, username: str, password: str) -> Optional[User]:
        """
        验证用户凭据。
        如果验证成功，返回 User 对象；否则返回 None。
        """
        # 首先检查是否是admin用户
        if username == settings.ADMIN_USERNAME:
            if password == settings.ADMIN_PASSWORD:  # admin使用明文密码
                return User(username)
            else:
                return None

        # 对于普通用户，检查其数据目录和密码文件是否存在
        user_data_path = os.path.join(settings.USER_DATA_ROOT_PATH, username)
        password_file = os.path.join(user_data_path, "password.hash")

        if not os.path.exists(password_file):
            return None

        try:
            with open(password_file, 'r', encoding='utf-8') as f:
                stored_hash = f.read().strip()
            if cls.verify_password(password, stored_hash):
                return User(username)
            else:
                return None
        except Exception:
            return None

    @classmethod
    def register_user(cls, username: str, password: str) -> bool:
        """
        注册新用户。
        1. 创建 User 对象。
        2. 调用 User.create_user_directory() 初始化目录。
        3. 将密码哈希后保存到用户目录。
        Returns:
            bool: 注册成功返回 True，失败返回 False。
        """
        if not username or not password:
            return False

        # 检查用户名是否已存在（admin 或 普通用户）
        if username == settings.ADMIN_USERNAME:
            return False

        user_data_path = os.path.join(settings.USER_DATA_ROOT_PATH, username)
        if os.path.exists(user_data_path):
            return False

        # 创建 User 对象
        user = User(username)

        # 创建用户目录并复制初始文件
        if not user.create_user_directory():
            return False

        # 保存密码哈希
        password_hash = cls.hash_password(password)
        password_file = os.path.join(user_data_path, "password.hash")
        try:
            with open(password_file, 'w', encoding='utf-8') as f:
                f.write(password_hash)
            return True
        except Exception:
            return False

    @classmethod
    def get_user(cls, username: str) -> Optional[User]:
        """
        根据用户名获取 User 对象。
        此方法不验证密码，仅用于在已登录状态下获取用户实例。
        """
        user_data_path = os.path.join(settings.USER_DATA_ROOT_PATH, username)
        if not os.path.exists(user_data_path):
            return None

        return User(username)
