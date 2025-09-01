import streamlit as st
import json
import pandas as pd
from io import StringIO
import re

# 页面设置
st.set_page_config(page_title="JSONL对话查看器", layout="wide")

# 初始化session state
if 'data' not in st.session_state:
    st.session_state.data = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'deleted_indices' not in st.session_state:
    st.session_state.deleted_indices = set()

# 自定义CSS样式
st.markdown("""
<style>
    .user-message {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4a90e2;
    }
    .assistant-message {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #6c757d;
    }
    .message-container {
        margin-bottom: 20px;
    }
    .message-role {
        font-weight: bold;
        margin-bottom: 5px;
        color: #555;
    }
    .original-text {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #ffc107;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# 文件上传和处理
def process_uploaded_file(uploaded_file):
    data = []
    for line in uploaded_file:
        try:
            data.append(json.loads(line.decode('utf-8')))
        except json.JSONDecodeError:
            st.error("文件包含无效的JSON行，请上传有效的JSONL文件")
            return []
    return data

# 渲染消息内容
def render_message(content):
    # 处理LaTeX公式：将$...$转换为LaTeX格式
    content = re.sub(r'\$\$(.*?)\$\$', r'$$\1$$', content)
    content = re.sub(r'\$(.*?)\$', r'$\1$', content)
    st.markdown(content)

# 显示对话
def display_conversation(messages):
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            st.markdown(f'<div class="user-message"><div class="message-role">👤 用户:</div>', unsafe_allow_html=True)
            render_message(content)
            st.markdown('</div>', unsafe_allow_html=True)
        elif role == "assistant":
            st.markdown(f'<div class="assistant-message"><div class="message-role">🤖 助手:</div>', unsafe_allow_html=True)
            render_message(content)
            st.markdown('</div>', unsafe_allow_html=True)

# 下载处理后的数据
def download_data(data):
    output = StringIO()
    for item in data:
        output.write(json.dumps(item, ensure_ascii=False) + '\n')
    return output.getvalue().encode('utf-8')

# 主应用
st.title("JSONL对话查看器")

# 文件上传区域
uploaded_file = st.file_uploader("上传JSONL文件", type=["jsonl"])

if uploaded_file is not None:
    if st.session_state.get('uploaded_file_name') != uploaded_file.name:
        st.session_state.data = process_uploaded_file(uploaded_file)
        st.session_state.current_index = 0
        st.session_state.deleted_indices = set()
        st.session_state.uploaded_file_name = uploaded_file.name
    
    if not st.session_state.data:
        st.error("无法处理上传的文件，请确保它是有效的JSONL格式")
    else:
        # 显示当前样本信息
        total_samples = len(st.session_state.data)
        current_index = st.session_state.current_index
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        with col1:
            st.write(f"总样本数: {total_samples - len(st.session_state.deleted_indices)}")
        with col2:
            st.write(f"当前样本: {current_index + 1} / {total_samples}")
        
        # 导航按钮
        col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 1, 1])
        with col_nav1:
            if st.button("⏮️ 第一条", use_container_width=True):
                st.session_state.current_index = 0
                st.rerun()
        with col_nav2:
            if st.button("◀️ 上一条", use_container_width=True) and current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()
        with col_nav3:
            if st.button("下一条 ▶️", use_container_width=True) and current_index < total_samples - 1:
                st.session_state.current_index += 1
                st.rerun()
        with col_nav4:
            if st.button("⏭️ 最后一条", use_container_width=True):
                st.session_state.current_index = total_samples - 1
                st.rerun()
        
        # 操作按钮
        col_act1, col_act2, col_act3 = st.columns([1, 1, 1])
        with col_act1:
            delete_clicked = st.button("🗑️ 删除当前样本", use_container_width=True)
        with col_act2:
            show_original = st.button("📄 查看原文", use_container_width=True)
        with col_act3:
            # 下载按钮
            filtered_data = [item for i, item in enumerate(st.session_state.data) 
                            if i not in st.session_state.deleted_indices]
            download_bytes = download_data(filtered_data)
            st.download_button(
                label="💾 下载数据集",
                data=download_bytes,
                file_name="processed_dataset.jsonl",
                mime="application/json",
                use_container_width=True
            )
        
        # 处理删除操作
        if delete_clicked:
            st.session_state.deleted_indices.add(current_index)
            # 如果删除的是最后一个样本，调整当前索引
            if current_index >= total_samples - len(st.session_state.deleted_indices):
                st.session_state.current_index = max(0, current_index - 1)
            st.success(f"已删除样本 {current_index + 1}")
            st.rerun()
        
        # 获取当前样本
        if current_index < len(st.session_state.data):
            current_sample = st.session_state.data[current_index]
            
            # 显示对话
            st.subheader("对话内容")
            messages = current_sample.get("messages", [])
            # 过滤掉system消息
            filtered_messages = [msg for msg in messages if msg.get("role") in ["user", "assistant"]]
            display_conversation(filtered_messages)
            
            # 显示原文（如果有）
            if show_original:
                original_text = current_sample.get("text", None)
                st.subheader("原文内容")
                if original_text:
                    st.markdown('<div class="original-text">', unsafe_allow_html=True)
                    st.text(original_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("暂无原文")
else:
    st.info("请上传JSONL文件开始使用")
    st.markdown("""
    ### 使用说明
    1. 上传一个JSONL文件，每行应包含一个JSON对象
    2. 每个JSON对象应有一个"messages"字段，包含对话列表
    3. 对话列表中的每个消息应有"role"和"content"字段
    4. 可选：JSON对象可以包含"text"字段存储原文
    5. 使用导航按钮浏览不同样本
    6. 可以删除不需要的样本，然后下载处理后的数据集
    """)

# 页脚
st.markdown("---")
st.markdown("JSONL对话查看器 | 支持Markdown和LaTeX渲染")
