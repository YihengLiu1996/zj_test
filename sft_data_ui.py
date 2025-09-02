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
import re

# 设置页面配置
st.set_page_config(
    page_title="JSONL数据集查看器",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# JSONL文件路径
JSONL_PATH = "dataset.jsonl"  # 请修改为您的JSONL文件路径

# 添加自定义CSS样式
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # 如果CSS文件不存在，创建默认样式
        default_css = """
        /* 默认样式 */
        .message-container {
            display: flex;
            margin-bottom: 20px;
        }
        
        .user-container {
            justify-content: flex-end;
        }
        
        .assistant-container {
            justify-content: flex-start;
        }
        
        .message-box {
            max-width: 80%;
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            position: relative;
        }
        
        .user-message {
            background-color: #e6f7ff;
            border: 1px solid #91d5ff;
            margin-left: 10%;
        }
        
        .assistant-message {
            background-color: #f6ffed;
            border: 1px solid #b7eb8f;
            margin-right: 10%;
        }
        
        .message-header {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .message-role {
            font-weight: bold;
            margin-left: 10px;
            font-size: 14px;
            color: #666;
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
            background-color: #fffbe6;
            border: 1px solid #ffe58f;
            border-radius: 8px;
            margin-top: 20px;
        }
        
        .avatar {
            font-size: 20px;
        }
        
        .stats-box {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .deleted-item {
            opacity: 0.6;
            background-color: #fff2f0;
            border: 1px solid #ffccc7;
        }
        """
        st.markdown(f"<style>{default_css}</style>", unsafe_allow_html=True)

local_css("style.css")

def clean_html_tags(text):
    """清除HTML标签"""
    if text is None:
        return ""
    # 使用正则表达式移除HTML标签
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def markdown_to_html(markdown_text):
    """将Markdown转换为HTML"""
    if markdown_text is None:
        return ""
    
    # 先清理HTML标签
    clean_text = clean_html_tags(markdown_text)
    
    # 然后转换为Markdown
    html = markdown.markdown(clean_text)
    # 美化HTML输出
    soup = BeautifulSoup(html, 'html.parser')
    return str(soup)

def load_jsonl_data(file_path):
    """加载JSONL文件数据"""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        st.error(f"解析JSONL行时出错: {e}")
    return data

def save_jsonl_data(file_path, data):
    """保存数据到JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def render_message(role, msg):
    """渲染单条消息"""
    if role == "user":
        container_class = "user-container"
        message_class = "user-message"
        avatar_emoji = "👤"
        role_display = "用户"
        
        html = f"""
        <div class="message-container {container_class}">
            <div class="message-box {message_class}">
                <div class="message-header">
                    <div class="avatar">{avatar_emoji}</div>
                    <div class="message-role">{role_display}</div>
                </div>
                <div class="message-content">
                    {markdown_to_html(msg.get('content', ''))}
                </div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:  # assistant
        container_class = "assistant-container"
        message_class = "assistant-message"
        avatar_emoji = "🤖"
        role_display = "助手"
        
        # 检查是否有content字段
        if 'content' in msg and msg['content']:
            # 有content字段，直接显示
            content = msg['content']
        else:
            # 没有content字段，拼接think和answer
            think_part = f"# 思考过程\n{msg.get('think', '')}\n\n" if 'think' in msg and msg['think'] else ""
            answer_part = f"# 最终答案\n{msg.get('answer', '')}" if 'answer' in msg and msg['answer'] else ""
            content = think_part + answer_part
        
        html = f"""
        <div class="message-container {container_class}">
            <div class="message-box {message_class}">
                <div class="message-header">
                    <div class="avatar">{avatar_emoji}</div>
                    <div class="message-role">{role_display}</div>
                </div>
                <div class="message-content">
                    {markdown_to_html(content)}
                </div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

def get_filtered_data():
    """获取过滤后的数据（排除已删除的项）"""
    return [
        item for i, item in enumerate(st.session_state.jsonl_data) 
        if i not in st.session_state.deleted_indices
    ]

def get_next_index(current_index, direction):
    """获取下一个有效的索引（跳过已删除的项）"""
    filtered_data = get_filtered_data()
    if not filtered_data:
        return current_index
    
    # 获取当前索引在过滤后数据中的位置
    try:
        current_pos = [i for i, item in enumerate(st.session_state.jsonl_data) 
                      if i not in st.session_state.deleted_indices].index(current_index)
    except ValueError:
        # 如果当前索引已被删除，从第一个有效项开始
        valid_indices = [i for i in range(len(st.session_state.jsonl_data)) 
                        if i not in st.session_state.deleted_indices]
        return valid_indices[0] if valid_indices else 0
    
    # 计算下一个位置
    if direction == "next":
        next_pos = (current_pos + 1) % len(filtered_data)
    else:  # previous
        next_pos = (current_pos - 1) % len(filtered_data)
    
    # 返回原始数据中的索引
    valid_indices = [i for i in range(len(st.session_state.jsonl_data)) 
                    if i not in st.session_state.deleted_indices]
    return valid_indices[next_pos]

def main():
    # 初始化session state
    if "jsonl_data" not in st.session_state:
        st.session_state.jsonl_data = load_jsonl_data(JSONL_PATH)
    
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    
    if "show_original" not in st.session_state:
        st.session_state.show_original = False
    
    if "deleted_indices" not in st.session_state:
        st.session_state.deleted_indices = set()

    # 页面标题
    st.title("📊 JSONL数据集查看器")
    
    # 检查数据是否为空
    if not st.session_state.jsonl_data:
        st.warning("没有找到数据或JSONL文件为空")
        return
    
    # 获取过滤后的数据
    filtered_data = get_filtered_data()
    
    # 如果没有有效数据，显示提示
    if not filtered_data:
        st.warning("所有样本已被删除，请重置或导入新数据")
        if st.button("重置数据"):
            st.session_state.deleted_indices = set()
            st.session_state.current_index = 0
            st.rerun()
        return
    
    # 确保当前索引有效（未被删除）
    if st.session_state.current_index in st.session_state.deleted_indices:
        st.session_state.current_index = get_next_index(st.session_state.current_index, "next")
    
    # 获取当前数据项
    current_item = st.session_state.jsonl_data[st.session_state.current_index]
    
    # 侧边栏
    with st.sidebar:
        st.header("导航控制")
        
        # 显示统计信息
        total_items = len(st.session_state.jsonl_data)
        remaining_items = len(filtered_data)
        deleted_items = total_items - remaining_items
        
        # 获取当前在剩余项中的位置
        valid_indices = [i for i in range(total_items) if i not in st.session_state.deleted_indices]
        current_pos = valid_indices.index(st.session_state.current_index) + 1 if st.session_state.current_index in valid_indices else 1
        
        st.markdown(f"""
        <div class="stats-box">
            <p><strong>总样本数:</strong> {total_items}</p>
            <p><strong>剩余样本数:</strong> {remaining_items}</p>
            <p><strong>已删除样本:</strong> {deleted_items}</p>
            <p><strong>当前样本:</strong> {current_pos}/{remaining_items}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏪ 上一条", use_container_width=True):
                st.session_state.current_index = get_next_index(st.session_state.current_index, "previous")
                st.session_state.show_original = False
                st.rerun()
        
        with col2:
            if st.button("⏩ 下一条", use_container_width=True):
                st.session_state.current_index = get_next_index(st.session_state.current_index, "next")
                st.session_state.show_original = False
                st.rerun()
        
        # 删除当前样本按钮
        if st.button("🗑️ 删除当前样本", type="primary", use_container_width=True):
            st.session_state.deleted_indices.add(st.session_state.current_index)
            # 自动跳到下一条
            st.session_state.current_index = get_next_index(st.session_state.current_index, "next")
            st.rerun()
        
        # 查看原文按钮
        if st.button("📄 查看原文", use_container_width=True):
            st.session_state.show_original = not st.session_state.show_original
            st.rerun()
        
        # 重置删除按钮
        if st.button("🔄 重置删除标记", use_container_width=True):
            st.session_state.deleted_indices = set()
            st.rerun()
        
        # 导出数据按钮
        if st.button("💾 导出数据集", use_container_width=True):
            # 保存到临时文件
            temp_file = "filtered_dataset.jsonl"
            save_jsonl_data(temp_file, filtered_data)
            
            # 提供下载
            with open(temp_file, "rb") as f:
                st.download_button(
                    label="📥 下载修改后的数据集",
                    data=f,
                    file_name="filtered_dataset.jsonl",
                    mime="application/json",
                    use_container_width=True
                )
    
    # 显示当前样本的对话
    st.subheader(f"📝 对话样本 {current_pos}/{remaining_items}")
    
    if st.session_state.current_index in st.session_state.deleted_indices:
        st.warning("此样本已被标记为删除")
    
    if "messages" in current_item:
        messages = current_item["messages"]
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                render_message(msg["role"], msg)
    else:
        st.warning("当前样本中没有找到'messages'字段")
    
    # 显示原文（如果用户点击了查看原文按钮）
    if st.session_state.show_original:
        st.subheader("📄 原文内容")
        if "text" in current_item:
            # 增大文本框高度到400
            st.text_area("原文", current_item["text"], height=400, key="original_text")
        else:
            st.info("当前样本中没有'text'字段")

if __name__ == "__main__":
    main()
