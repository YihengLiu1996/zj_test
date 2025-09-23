# background_tasks/task_manager.py

import threading
import asyncio
import time
from typing import Dict, Any, Callable, Optional, List

class TaskManager:
    """
    后台任务管理器
    负责创建、管理和监控后台异步任务。
    使用线程来运行 asyncio 事件循环。
    """

    # 类级别的任务存储，所有实例共享
    _tasks: Dict[str, Dict[str, Any]] = {}
    _task_lock = threading.Lock()

    @classmethod
    def start_background_task(
        cls,
        task_id: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None
    ) -> None:
        """
        在后台线程中启动一个异步任务。
        Args:
            task_id: 任务的唯一标识符。
            func: 要执行的异步函数。
            args: 传递给函数的位置参数。
            kwargs: 传递给函数的关键字参数。
        """
        if kwargs is None:
            kwargs = {}

        # 初始化任务状态
        with cls._task_lock:
            cls._tasks[task_id] = {
                "status": "running",
                "progress": 0,
                "total": 0,
                "message": "任务已启动",
                "logs": [],
                "start_time": time.time(),
                "end_time": None
            }

        # 定义在新线程中运行的包装函数
        def run_async_task():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # 运行异步函数
                result = loop.run_until_complete(func(*args, **kwargs))
                # 更新任务状态为完成
                with cls._task_lock:
                    cls._tasks[task_id]["status"] = "completed"
                    cls._tasks[task_id]["message"] = "任务成功完成"
                    cls._tasks[task_id]["end_time"] = time.time()
            except Exception as e:
                # 更新任务状态为失败
                with cls._task_lock:
                    cls._tasks[task_id]["status"] = "failed"
                    cls._tasks[task_id]["message"] = f"任务执行失败: {str(e)}"
                    cls._tasks[task_id]["end_time"] = time.time()
                # 记录错误日志
                cls.append_task_log(task_id, f"任务失败: {str(e)}")
            finally:
                loop.close()

        # 创建并启动后台线程
        thread = threading.Thread(target=run_async_task, daemon=True)
        thread.start()

    @classmethod
    def update_task_progress(cls, task_id: str, current: int, total: int) -> None:
        """
        更新任务的进度。
        Args:
            task_id: 任务ID。
            current: 当前完成的数量。
            total: 总数量。
        """
        with cls._task_lock:
            if task_id in cls._tasks:
                cls._tasks[task_id]["progress"] = current
                cls._tasks[task_id]["total"] = total

    @classmethod
    def update_task_status(cls, task_id: str, status: str, message: str) -> None:
        """
        更新任务的状态和消息。
        Args:
            task_id: 任务ID。
            status: 任务状态 (e.g., 'running', 'completed', 'failed')。
            message: 状态消息。
        """
        with cls._task_lock:
            if task_id in cls._tasks:
                cls._tasks[task_id]["status"] = status
                cls._tasks[task_id]["message"] = message

    @classmethod
    def append_task_log(cls, task_id: str, log_message: str) -> None:
        """
        向任务日志中追加一条消息。
        Args:
            task_id: 任务ID。
            log_message: 日志消息。
        """
        with cls._task_lock:
            if task_id in cls._tasks:
                cls._tasks[task_id]["logs"].append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {log_message}")

    @classmethod
    def get_task_info(cls, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务的完整信息。
        Returns:
            dict: 任务信息字典，如果任务不存在则返回 None。
        """
        with cls._task_lock:
            return cls._tasks.get(task_id, None)

    @classmethod
    def get_task_logs(cls, task_id: str) -> List[str]:
        """
        获取任务的日志列表。
        Returns:
            List[str]: 日志消息列表。
        """
        with cls._task_lock:
            if task_id in cls._tasks:
                return cls._tasks[task_id]["logs"].copy()
            else:
                return []

    @classmethod
    def list_all_tasks(cls) -> Dict[str, Dict[str, Any]]:
        """
        获取所有任务的信息。
        Returns:
            dict: 所有任务的信息。
        """
        with cls._task_lock:
            return cls._tasks.copy()
