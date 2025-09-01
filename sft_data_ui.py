import streamlit as st
import json
import tempfile
import os
import re
from io import StringIO
import time

# 页面设置
st.set_page_config(page_title="JSONL对话查看器", layout="wide")

# 初始化session state
if 'data' not in st.session_state:
    st.session_state.data = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'deleted_indices' not in st.session_state:
    st.session_state.deleted_indices = set()
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'show_original' not in st.session_state:
    st.session_state.show_original = False

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
    .progress-text {
        font-size: 14px;
        color: #666;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 处理JSONL文件
def process_jsonl_file(file_obj):
    data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 获取文件大小
        file_obj.seek(0, 2)  # 移动到文件末尾
        file_size = file_obj.tell()
        file_obj.seek(0)  # 回到文件开头
        
        # 逐行读取文件
        bytes_read = 0
        line_count = 0
        
        while True:
            line = file_obj.readline()
            if not line:
                break
                
            bytes_read += len(line)
            line_count += 1
            
            # 更新进度
            progress = bytes_read / file_size
            progress_bar.progress(progress)
            status_text.text(f"已处理 {line_count} 行，已读取 {bytes_read/(1024*1024):.2f} MB / {file_size/(1024*1024):.2f} MB")
            
            try:
                # 解码并解析JSON
                line_str = line.decode('utf-8').strip()
                if line_str:  # 确保不是空行
                    data.append(json.loads(line_str))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                st.error(f"第 {line_count} 行解析错误: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        return data
    except Exception as e:
        st.error(f"处理文件时发生错误: {e}")
        return []

# 渲染消息内容
def render_message(content):
    if not content:
        return
        
    # 处理LaTeX公式：将$...$转换为LaTeX格式
    content = re.sub(r'\$\$(.*?)\$\$', r'$$\1$$', content)
    content = re.sub(r'\$(.*?)\$', r'$\1$', content)
    st.markdown(content)

# 显示对话
def display_conversation(messages):
    if not messages:
        st.info("此样本没有对话内容")
        return
        
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
    # 检查是否是新上传的文件
    if st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.show_original = False
        
        with st.spinner("正在处理文件，请稍候..."):
            data = process_jsonl_file(uploaded_file)
            if data:
                st.session_state.data = data
                st.session_state.current_index = 0
                st.session_state.deleted_indices = set()
                st.success(f"成功处理 {len(data)} 条记录")
            else:
                st.error("未能处理文件或文件为空")
    
    # 显示当前样本信息
    if st.session_state.data:
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
                st.session_state.show_original = False
                st.rerun()
        with col_nav2:
            if st.button("◀️ 上一条", use_container_width=True) and current_index > 0:
                st.session_state.current_index -= 1
                st.session_state.show_original = False
                st.rerun()
        with col_nav3:
            if st.button("下一条 ▶️", use_container_width=True) and current_index < total_samples - 1:
                st.session_state.current_index += 1
                st.session_state.show_original = False
                st.rerun()
        with col_nav4:
            if st.button("⏭️ 最后一条", use_container_width=True):
                st.session_state.current_index = total_samples - 1
                st.session_state.show_original = False
                st.rerun()
        
        # 操作按钮
        col_act1, col_act2, col_act3 = st.columns([1, 1, 1])
        with col_act1:
            delete_clicked = st.button("🗑️ 删除当前样本", use_container_width=True)
        with col_act2:
            if st.button("📄 查看原文", use_container_width=True):
                st.session_state.show_original = not st.session_state.show_original
                st.rerun()
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
            time.sleep(0.5)  # 短暂延迟让用户看到成功消息
            st.rerun()
        
        # 获取当前样本
        if 0 <= current_index < len(st.session_state.data):
            current_sample = st.session_state.data[current_index]
            
            # 显示对话
            st.subheader("对话内容")
            messages = current_sample.get("messages", [])
            # 过滤掉system消息
            filtered_messages = [msg for msg in messages if msg.get("role") in ["user", "assistant"]]
            display_conversation(filtered_messages)
            
            # 显示原文（如果有）
            if st.session_state.show_original:
                original_text = current_sample.get("text", None)
                st.subheader("原文内容")
                if original_text:
                    st.markdown('<div class="original-text">', unsafe_allow_html=True)
                    st.text(original_text)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("暂无原文")
    else:
        st.warning("没有有效数据可显示")
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
    
    ### 示例JSONL格式：
    ```
    {"messages": [{"role": "system", "content": "你是个有用无害的助手"}, {"role": "user", "content": "告诉我明天的天气"}, {"role": "assistant", "content": "明天天气晴朗"}]}
    {"messages": [{"role": "user", "content": "什么是勾股定理？"}, {"role": "assistant", "content": "勾股定理是$a^2 + b^2 = c^2$"}]}
    ```
    """)

# 页脚
st.markdown("---")
st.markdown("JSONL对话查看器 | 支持Markdown和LaTeX渲染")
