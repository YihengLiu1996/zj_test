import os
import json
import asyncio
import random
import csv
import base64
import pandas as pd
import streamlit as st
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import collections
import copy
import markdown
from bs4 import BeautifulSoup
 
# 设置页面配置
st.set_page_config(
    page_title="对话数据查看器",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# 添加自定义CSS样式
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # 如果CSS文件不存在，创建默认样式
        default_css = """
        /* 默认样式 */
        .message-box {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s;
        }
        
        .user-message {
            background-color: #f0f7ff;
            border-left: 4px solid #1890ff;
        }
        
        .assistant-message {
            background-color: #f9f9f9;
            border-left: 4px solid #52c41a;
        }
        
        .message-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 10px;
            margin-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        
        .role-badge {
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        
        .user-badge {
            background-color: #1890ff;
        }
        
        .assistant-badge {
            background-color: #52c41a;
        }
        
        .message-content {
            line-height: 1.6;
            font-size: 15px;
        }
        
        .message-content h1, .message-content h2, .message-content h3 {
            color: #1890ff;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        
        .message-content pre {
            background-color: #f6f8fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        
        .message-content code {
            background-color: #f6f8fa;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: monospace;
        }
        
        .original-text-box {
            padding: 20px;
            background-color: #fff7e6;
            border-left: 4px solid #fa8c16;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        
        .stButton>button {
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        """
        st.markdown(f"<style>{default_css}</style>", unsafe_allow_html=True)
 
 
local_css("style.css")
 
 
def markdown_to_html(markdown_text):
    """将Markdown转换为HTML"""
    if markdown_text is None:
        return ""
    html = markdown.markdown(markdown_text)
    # 美化HTML输出
    soup = BeautifulSoup(html, 'html.parser')
    return str(soup)
 
 
def render_message(role, content):
    """渲染单条消息"""
    if role == "user":
        badge_class = "user-badge"
        message_class = "user-message"
        role_display = "用户"
    else:
        badge_class = "assistant-badge"
        message_class = "assistant-message"
        role_display = "助手"
    
    html = f"""
    <div class="message-box {message_class}">
        <div class="message-header">
            <span class="role-badge {badge_class}">{role_display}</span>
        </div>
        <div class="message-content">
            {markdown_to_html(content)}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
 
 
def main():
    # 初始化session state
    if "data" not in st.session_state:
        st.session_state.data = []
    
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    
    if "deleted_indices" not in st.session_state:
        st.session_state.deleted_indices = set()

    # 页面标题
    st.title("💬 对话数据查看器")
    st.markdown("""
    <style>
    .title {
        color: #1890ff;
        border-bottom: 2px solid #1890ff;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 文件上传
    uploaded_file = st.sidebar.file_uploader("上传JSONL文件", type=["jsonl"])

    if uploaded_file is not None:
        # 读取上传的文件
        lines = uploaded_file.readlines()
        data = []
        for i, line in enumerate(lines):
            try:
                item = json.loads(line.decode('utf-8'))
                data.append(item)
            except json.JSONDecodeError:
                st.sidebar.error(f"第 {i+1} 行不是有效的JSON格式")
        
        if data:
            st.session_state.data = data
            st.session_state.current_index = 0
            st.session_state.deleted_indices = set()
            st.sidebar.success(f"成功加载 {len(data)} 条数据")

    # 如果没有数据，显示提示
    if not st.session_state.data:
        st.info("请上传JSONL文件开始查看对话数据")
        return

    # 侧边栏导航
    with st.sidebar:
        st.header("数据导航")
        
        # 显示进度
        total_items = len(st.session_state.data)
        deleted_count = len(st.session_state.deleted_indices)
        valid_count = total_items - deleted_count
        st.caption(f"总数据: {total_items} 条")
        st.caption(f"有效数据: {valid_count} 条")
        st.caption(f"已删除: {deleted_count} 条")
        
        # 导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("上一条", use_container_width=True):
                # 找到上一条未删除的数据
                prev_index = st.session_state.current_index - 1
                while prev_index >= 0:
                    if prev_index not in st.session_state.deleted_indices:
                        st.session_state.current_index = prev_index
                        st.rerun()
                    prev_index -= 1
        
        with col2:
            if st.button("下一条", use_container_width=True):
                # 找到下一条未删除的数据
                next_index = st.session_state.current_index + 1
                while next_index < total_items:
                    if next_index not in st.session_state.deleted_indices:
                        st.session_state.current_index = next_index
                        st.rerun()
                    next_index += 1
        
        # 删除当前样本按钮
        if st.button("删除当前样本", type="secondary", use_container_width=True):
            st.session_state.deleted_indices.add(st.session_state.current_index)
            # 找到下一条未删除的数据
            next_index = st.session_state.current_index + 1
            while next_index < total_items:
                if next_index not in st.session_state.deleted_indices:
                    st.session_state.current_index = next_index
                    break
                next_index += 1
            else:
                # 如果没有下一条，找到上一条
                prev_index = st.session_state.current_index - 1
                while prev_index >= 0:
                    if prev_index not in st.session_state.deleted_indices:
                        st.session_state.current_index = prev_index
                        break
                    prev_index -= 1
                else:
                    # 如果没有上一条，设为0
                    st.session_state.current_index = 0
            st.rerun()
        
        # 查看原文按钮
        if st.button("查看原文", use_container_width=True):
            st.session_state.show_original = not st.session_state.get('show_original', False)
            st.rerun()
        
        # 下载按钮
        if st.button("导出数据集", type="primary", use_container_width=True):
            # 创建未删除的数据
            filtered_data = [
                item for i, item in enumerate(st.session_state.data) 
                if i not in st.session_state.deleted_indices
            ]
            
            # 转换为JSONL格式
            jsonl_content = "\n".join([json.dumps(item, ensure_ascii=False) for item in filtered_data])
            
            # 提供下载
            st.download_button(
                label="下载修改后的数据集",
                data=jsonl_content,
                file_name="filtered_dataset.jsonl",
                mime="application/json",
                use_container_width=True
            )

    # 获取当前数据项
    current_item = st.session_state.data[st.session_state.current_index]
    
    # 显示当前数据位置
    st.caption(f"当前数据: {st.session_state.current_index + 1} / {total_items}")
    
    # 显示对话
    if "messages" in current_item:
        messages = current_item["messages"]
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                render_message(msg["role"], msg["content"])
    else:
        st.warning("当前数据项中没有'messages'字段")
    
    # 显示原文（如果用户点击了查看原文按钮）
    if st.session_state.get('show_original', False):
        st.markdown("---")
        st.subheader("原文内容")
        if "text" in current_item:
            st.markdown(f"<div class='original-text-box'>{current_item['text']}</div>", unsafe_allow_html=True)
        else:
            st.info("当前样本中没有'text'字段")


if __name__ == "__main__":
    main()
