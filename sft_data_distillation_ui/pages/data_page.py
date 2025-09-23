# pages/01_数据页.py

import os
import time
import streamlit as st
from models.dataset import Dataset
from services.file_service import FileService
from config import settings
from components.markdown_viewer import MarkdownViewerComponent # 我们稍后会创建这个组件

def main():
    """数据页主函数"""
    # 从Session State获取当前用户
    if settings.SESSION_KEY_CURRENT_USER not in st.session_state:
        st.error("未检测到登录用户，请先登录。")
        return

    user = st.session_state[settings.SESSION_KEY_CURRENT_USER]

    st.title("数据管理")

    # 创建两列布局：左列配置，右列数据展示
    col_config, col_display = st.columns([1, 2])

    # ========== 左侧：数据集配置 ==========
    with col_config:
        st.subheader("数据集配置")

        # 数据加载方式选择
        load_option = st.radio(
            "选择数据加载方式",
            ("加载原始数据集", "加载用户保存的数据集"),
            key="data_load_option"
        )

        if load_option == "加载原始数据集":
            # 输入原始数据集文件夹路径
            raw_data_dir = st.text_input(
                "原始数据集文件夹路径",
                key="raw_data_dir_input",
                help="请输入包含 .md 文件的文件夹路径"
            )
            # 配置文本长度
            max_tokens = st.number_input(
                "文本最大长度 (max_tokens)",
                min_value=100,
                max_value=50000,
                value=15000,
                step=100,
                key="max_tokens_input"
            )
            min_tokens = st.number_input(
                "文本最小长度 (min_tokens)",
                min_value=10,
                max_value=10000,
                value=2000,
                step=100,
                key="min_tokens_input"
            )

            # 加载按钮
            if st.button("加载数据集", key="load_raw_dataset_btn", type="primary"):
                if not raw_data_dir or not os.path.exists(raw_data_dir) or not os.path.isdir(raw_data_dir):
                    st.error("请输入有效的原始数据集文件夹路径。")
                else:
                    with st.spinner("正在加载并预处理数据..."):
                        # 创建一个临时数据集对象
                        temp_dataset_name = f"temp_dataset_{int(time.time())}"
                        dataset = Dataset(temp_dataset_name, user.data_path)
                        success = dataset.load_from_raw_markdown(
                            markdown_dir=raw_data_dir,
                            max_tokens=int(max_tokens),
                            min_tokens=int(min_tokens)
                        )
                        if success:
                            st.session_state[settings.SESSION_KEY_CURRENT_DATASET] = dataset
                            st.success(f"成功加载 {dataset.get_total_count()} 条数据！")
                            st.rerun() # 重新加载页面以显示数据
                        else:
                            st.error("数据加载失败，请检查路径或日志。")

        else: # 加载用户保存的数据集
            # 获取用户已保存的数据集列表
            saved_datasets = user.list_datasets()
            if not saved_datasets:
                st.info("您还没有保存任何数据集。")
            else:
                selected_dataset = st.selectbox(
                    "选择已保存的数据集",
                    saved_datasets,
                    key="saved_dataset_selector"
                )
                # 加载按钮
                if st.button("加载数据集", key="load_saved_dataset_btn", type="primary"):
                    with st.spinner("正在加载数据..."):
                        dataset = Dataset(selected_dataset, user.data_path)
                        success = dataset.load_from_saved_jsonl(selected_dataset, user)
                        if success:
                            st.session_state[settings.SESSION_KEY_CURRENT_DATASET] = dataset
                            st.success(f"成功加载 {dataset.get_total_count()} 条数据！")
                            st.rerun()
                        else:
                            st.error("数据加载失败。")

    # ========== 右侧：数据展示与编辑 ==========
    with col_display:
        # 检查是否有数据集被加载
        if settings.SESSION_KEY_CURRENT_DATASET not in st.session_state:
            st.info("请在左侧配置并加载一个数据集。")
            return

        dataset: Dataset = st.session_state[settings.SESSION_KEY_CURRENT_DATASET]

        if not dataset.is_loaded or not dataset.data_items:
            st.warning("数据集为空。")
            return

        # 显示当前数据索引信息
        st.markdown(f"**当前数据**: {dataset.get_current_index_info()}")

        # 获取当前数据项
        current_item = dataset.get_current_item()
        if current_item is None:
            st.error("无法获取当前数据项。")
            return

        # ========== 数据内容显示 ==========
        st.subheader("数据内容")
        text_content = current_item.get("text", "")
        # 使用自定义的Markdown查看器组件渲染
        MarkdownViewerComponent.render(text_content)

        # ========== 问题添加区域 ==========
        st.subheader("为当前数据添加问题")

        # 初始化当前数据项的messages列表（如果不存在）
        if "messages" not in current_item:
            current_item["messages"] = []

        # 获取当前已添加的问题数量
        existing_questions = [msg for msg in current_item["messages"] if msg.get("role") == "user"]
        question_count = len(existing_questions)

        st.write(f"已添加问题数: {question_count} / 10")

        # 使用表单来管理问题的添加
        with st.form(key=f"add_question_form_{dataset.current_index}"):
            new_question = st.text_area(
                "输入问题",
                height=100,
                key=f"new_question_input_{dataset.current_index}",
                help="针对上方数据内容提出的问题"
            )
            add_question_submitted = st.form_submit_button("添加问题")

            if add_question_submitted:
                if not new_question.strip():
                    st.error("问题内容不能为空。")
                elif question_count >= 10:
                    st.error("最多只能添加10个问题。")
                else:
                    success = dataset.add_question_to_current_item(new_question)
                    if success:
                        st.success("问题添加成功！")
                        # 由于dataset是对象，其内部状态已改变，直接重新渲染即可
                        st.rerun()
                    else:
                        st.error("添加问题失败。")

        # ========== 显示已添加的问题 ==========
        if existing_questions:
            st.write("**已添加的问题列表**:")
            for i, msg in enumerate(existing_questions):
                st.markdown(f"**问题 {i+1}**: {msg['content']}")

        # ========== 分页导航 ==========
        st.markdown("---")
        col_prev, col_next = st.columns(2)

        with col_prev:
            if st.button("⬅️ 上一条", key="prev_item_btn", disabled=not dataset.has_prev()):
                dataset.prev_item()
                st.rerun()

        with col_next:
            if st.button("下一条 ➡️", key="next_item_btn", disabled=not dataset.has_next()):
                dataset.next_item()
                st.rerun()

        # ========== 保存数据集 ==========
        st.markdown("---")
        st.subheader("保存数据集")
        save_dataset_name = st.text_input(
            "数据集名称",
            value=f"dataset_{int(time.time())}",
            key="save_dataset_name_input",
            help="请输入数据集名称，将保存到您的用户目录下"
        )
        if st.button("💾 保存数据集", key="save_dataset_btn", type="primary"):
            if not save_dataset_name.strip():
                st.error("数据集名称不能为空。")
            else:
                with st.spinner("正在保存数据集..."):
                    success = dataset.save_dataset(save_dataset_name.strip(), user)
                    if success:
                        st.success(f"数据集 '{save_dataset_name}' 保存成功！")
                        # 更新Session中的数据集对象，使其指向新保存的路径
                        st.session_state[settings.SESSION_KEY_CURRENT_DATASET] = dataset
                    else:
                        st.error("数据集保存失败。")

# 调用主函数
if __name__ == "__main__":
    main()
