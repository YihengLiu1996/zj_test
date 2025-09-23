# pages/04_数据蒸馏配置页.py

import streamlit as st
import asyncio
import time
from services.model_config_service import ModelConfigService
from services.distillation_service import DistillationService
from background_tasks.task_manager import TaskManager
from config import settings

def main():
    """数据蒸馏配置页主函数"""
    # 从Session State获取当前用户
    if settings.SESSION_KEY_CURRENT_USER not in st.session_state:
        st.error("未检测到登录用户，请先登录。")
        return

    user = st.session_state[settings.SESSION_KEY_CURRENT_USER]

    st.title("数据蒸馏配置")

    # 检查是否加载了数据集
    if settings.SESSION_KEY_CURRENT_DATASET not in st.session_state:
        st.warning("请先在'数据页'加载一个数据集。")
        return

    dataset = st.session_state[settings.SESSION_KEY_CURRENT_DATASET]
    if not dataset.is_loaded:
        st.warning("当前数据集未加载，请先加载数据。")
        return

    # 加载大模型配置
    model_configs = ModelConfigService.load_model_config(user.model_config_path)
    if not model_configs:
        st.error("您还没有配置任何大模型。请先到'大模型配置页'进行配置。")
        return

    st.info(f"已加载 {len(model_configs)} 个大模型配置。")

    # ========== 蒸馏配置表单 ==========
    st.subheader("蒸馏参数配置")

    with st.form(key="distillation_config_form"):
        st.markdown("### 问题生成")
        question_types = st.number_input(
            "题型数量",
            min_value=1,
            max_value=5,
            value=1,
            help="为每个文本chunk生成几种不同类型的问题"
        )

        st.markdown("### 回答生成")
        answer_mode = st.radio(
            "对话模式",
            options=["single", "multi_turn"],
            format_func=lambda x: "单轮对话" if x == "single" else "多轮对话",
            index=0
        )

        max_rounds = 3
        round_probabilities = [0.8, 0.2/6, 0.2/6, 0.2/6, 0.2/6, 0.2/6, 0.2/6]
        if answer_mode == "multi_turn":
            max_rounds = st.number_input(
                "最大对话轮数",
                min_value=1,
                max_value=10,
                value=3
            )
            # 为了简化，我们不提供每个轮数的概率配置，使用默认值
            st.info("多轮对话概率使用默认配置: [0.8, 0.2/6, 0.2/6, 0.2/6, 0.2/6, 0.2/6, 0.2/6]")

        st.markdown("### 质量过滤")
        filter_options = st.multiselect(
            "选择启用的过滤器",
            options=["question_filter", "answer_filter", "qa_score"],
            format_func=lambda x: {
                "question_filter": "问题过滤",
                "answer_filter": "回答过滤",
                "qa_score": "质量打分"
            }[x],
            default=["question_filter", "answer_filter", "qa_score"]
        )

        submitted = st.form_submit_button("🚀 开始蒸馏", type="primary")

        if submitted:
            # 构造蒸馏配置
            distill_config = {
                "question_types": int(question_types),
                "answer_mode": answer_mode,
                "max_rounds": int(max_rounds),
                "round_probabilities": round_probabilities,
                "enabled_filters": filter_options
            }

            # 生成任务ID
            task_id = f"task_{int(time.time())}"
            st.session_state[settings.SESSION_KEY_DISTILLATION_TASK_ID] = task_id

            # 设置输出目录
            output_dir = os.path.join(user.distillation_output_path, task_id)

            # 在后台启动蒸馏任务
            # 注意：在Streamlit中，不能直接 await 一个耗时的异步函数，会阻塞UI
            # 我们需要在后台线程中运行
            distillation_service = DistillationService(user)
            
            # 启动后台任务
            TaskManager.start_background_task(
                task_id=task_id,
                func=distillation_service.start_distillation_task,
                kwargs={
                    "task_id": task_id,
                    "dataset_path": dataset.dataset_path,
                    "model_configs": model_configs,
                    "distill_config": distill_config,
                    "output_dir": output_dir
                }
            )

            st.success(f"蒸馏任务已启动！任务ID: {task_id}")
            st.info("您可以在下方查看任务进度，或随时切换到其他页面。")

    # ========== 任务进度显示 ==========
    st.markdown("---")
    st.subheader("任务进度")

    if settings.SESSION_KEY_DISTILLATION_TASK_ID in st.session_state:
        task_id = st.session_state[settings.SESSION_KEY_DISTILLATION_TASK_ID]
        task_info = TaskManager.get_task_info(task_id)

        if task_info:
            status = task_info.get("status", "unknown")
            progress = task_info.get("progress", 0)
            total = task_info.get("total", 1)
            message = task_info.get("message", "")

            st.write(f"**任务ID**: {task_id}")
            st.write(f"**状态**: {status}")
            if total > 0:
                progress_percent = int((progress / total) * 100)
                st.progress(progress_percent, text=f"进度: {progress}/{total} ({progress_percent}%)")
            st.write(f"**消息**: {message}")

            # 显示日志 (如果有)
            logs = TaskManager.get_task_logs(task_id)
            if logs:
                st.markdown("**日志**:")
                for log in logs[-10:]: # 只显示最近10条
                    st.text(log)

            # 提供刷新按钮
            if st.button("🔄 刷新进度"):
                st.rerun()

            # 如果任务已完成，提供下载链接或查看结果的按钮
            if status in ["completed", "failed"]:
                st.markdown("---")
                if status == "completed":
                    st.success("🎉 任务已完成！")
                    output_dir = os.path.join(user.distillation_output_path, task_id, "final_output")
                    if os.path.exists(output_dir):
                        st.write(f"结果保存在: `{output_dir}`")
                        # 可以在这里添加一个按钮，跳转到数据页查看结果
                        if st.button("前往数据页查看结果"):
                            # 这里可以设置一个标志，让数据页自动加载这个输出目录
                            st.session_state['auto_load_distilled_dataset'] = output_dir
                            st.switch_page("pages/01_数据页.py") # Streamlit 1.27+ 支持
                else:
                    st.error("任务失败！")
        else:
            st.warning("任务信息不存在。")
    else:
        st.info("尚未启动任何蒸馏任务。")

# 调用主函数
if __name__ == "__main__":
    main()
