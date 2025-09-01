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
    page_title="数据集查看与编辑器",
    page_icon="📄",
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
        .chat-container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px;
            border-radius: 10px;
            max-width: 80%;
            line-height: 1.5;
        }
        
        .user-message {
            background-color: #e6f7ff;
            border-left: 4px solid #1890ff;
            margin-left: auto;
        }
        
        .assistant-message {
            background-color: #f0f0f0;
            border-left: 4px solid #52c41a;
            margin-right: auto;
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
        
        .stButton>button {
            transition: all 0.3s;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        
        .navigation-container {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 8px;
        }
        
        .original-text {
            background-color: #fff9e6;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            border-left: 4px solid #faad14;
        }
        
        .progress-container {
            margin-top: 10px;
        }
        
        .sample-counter {
            font-size: 16px;
            font-weight: bold;
            color: #1890ff;
            text-align: center;
            margin: 10px 0;
        }
        
        .action-buttons {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 20px;
        }
        
        .delete-warning {
            color: #ff4d4f;
            font-weight: bold;
        }
        """
        st.markdown(f"<style>{default_css}</style>", unsafe_allow_html=True)

local_css("style.css")

def parse_jsonl(file):
    """解析JSONL文件"""
    dataset = []
    for line in file:
        try:
            # 处理可能的空行
            if line.strip() == b"":
                continue
            data = json.loads(line)
            # 只保留user和assistant消息，忽略system
            messages = [
                msg for msg in data.get("messages", [])
                if msg.get("role") in ["user", "assistant"]
            ]
            # 保留text字段（如果存在）
            text = data.get("text", None)
            dataset.append({
                "messages": messages,
                "text": text,
                "original": data  # 保留原始数据用于导出
            })
        except Exception as e:
            st.error(f"解析行时出错: {str(e)}")
    return dataset

def render_markdown(content):
    """渲染Markdown内容，支持LaTeX"""
    # 处理LaTeX公式
    content = re.sub(r'\$(.*?)\$', r'\\(\1\\)', content)
    content = re.sub(r'\$\$(.*?)\$\$', r'\\[\1\\]', content)
    
    # 转换Markdown为HTML
    html = markdown.markdown(content, extensions=['extra', 'nl2br', 'sane_lists'])
    
    # 添加MathJax支持
    mathjax_script = """
    <script type="text/javascript" async
      src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
    </script>
    """
    
    return f"{html}{mathjax_script}"

def display_message(role, content):
    """显示单条消息"""
    if role == "user":
        st.markdown(f"""
        <div class="message user-message">
            <div class="message-content">
                {render_markdown(content)}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f"""
        <div class="message assistant-message">
            <div class="message-content">
                {render_markdown(content)}
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    # 初始化session state
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
        st.session_state.current_index = 0
        st.session_state.show_original = False
    
    # 页面标题
    st.title("📄 数据集查看与编辑器")
    st.markdown("上传JSONL文件查看对话数据集，支持Markdown和LaTeX渲染，可删除样本并导出修改后的数据集")
    
    # 侧边栏
    with st.sidebar:
        st.header("数据集管理")
        
        # 文件上传
        uploaded_file = st.file_uploader("上传JSONL文件", type=["jsonl"])
        if uploaded_file is not None:
            # 读取文件
            file_contents = uploaded_file.getvalue().splitlines()
            dataset = parse_jsonl(file_contents)
            
            if dataset:
                st.session_state.dataset = dataset
                st.session_state.current_index = 0
                st.session_state.show_original = False
                st.success(f"成功加载 {len(dataset)} 条样本")
            else:
                st.warning("文件解析后没有有效数据")
        
        # 数据集信息
        if st.session_state.dataset is not None:
            st.subheader("当前数据集")
            st.write(f"总样本数: **{len(st.session_state.dataset)}**")
            
            # 进度条
            if len(st.session_state.dataset) > 0:
                progress = (st.session_state.current_index + 1) / len(st.session_state.dataset)
                st.progress(progress)
                st.caption(f"当前样本: {st.session_state.current_index + 1}/{len(st.session_state.dataset)}")
            
            # 下载按钮
            if st.button("下载修改后的数据集", type="primary"):
                # 创建JSONL内容
                jsonl_content = ""
                for item in st.session_state.dataset:
                    # 使用原始数据导出（保留所有字段）
                    jsonl_content += json.dumps(item["original"], ensure_ascii=False) + "\n"
                
                # 提供下载
                st.download_button(
                    label="点击下载",
                    data=jsonl_content,
                    file_name="modified_dataset.jsonl",
                    mime="application/json"
                )
    
    # 主内容区域
    if st.session_state.dataset is None:
        st.info("请在侧边栏上传JSONL文件")
        
        # 显示示例格式
        st.markdown("""
        **JSONL文件格式示例**:
        ```json
        {"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}
        {"messages": [{"role": "user", "content": "明天天气？"}, {"role": "assistant", "content": "晴天"}]}
        ```
        """)
        
        st.markdown("""
        **支持的功能**:
        - 渲染Markdown格式和LaTeX公式
        - 查看原始文本内容
        - 删除无效样本
        - 导出修改后的数据集
        """)
        
    else:
        # 当前样本
        current_sample = st.session_state.dataset[st.session_state.current_index]
        
        # 样本计数器
        st.markdown(f"<div class='sample-counter'>样本 {st.session_state.current_index + 1} / {len(st.session_state.dataset)}</div>", 
                    unsafe_allow_html=True)
        
        # 显示对话
        st.subheader("对话内容")
        with st.container():
            for message in current_sample["messages"]:
                display_message(message["role"], message["content"])
        
        # 查看原文按钮
        if current_sample["text"] is not None:
            if st.button("查看原文", key="toggle_original"):
                st.session_state.show_original = not st.session_state.show_original
            
            if st.session_state.show_original:
                with st.expander("原文内容", expanded=True):
                    st.markdown(f"""
                    <div class="original-text">
                        {render_markdown(current_sample['text'])}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.caption("暂无原文内容")
        
        # 操作区域
        st.markdown("<div class='action-buttons'>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("⇦ 上一条", disabled=(st.session_state.current_index == 0)):
                st.session_state.current_index -= 1
                st.session_state.show_original = False
                st.rerun()
        
        with col2:
            if st.button("下一条 ⇨", disabled=(st.session_state.current_index == len(st.session_state.dataset) - 1)):
                st.session_state.current_index += 1
                st.session_state.show_original = False
                st.rerun()
        
        with col3:
            if st.button("🗑️ 删除当前样本"):
                # 删除当前样本
                del st.session_state.dataset[st.session_state.current_index]
                
                # 调整索引
                if st.session_state.current_index >= len(st.session_state.dataset):
                    st.session_state.current_index = max(0, len(st.session_state.dataset) - 1)
                
                st.session_state.show_original = False
                st.rerun()
        
        with col4:
            # 显示删除警告（如果数据集即将为空）
            if len(st.session_state.dataset) == 1:
                st.markdown("<p class='delete-warning'>警告：删除后数据集将为空</p>", 
                            unsafe_allow_html=True)
            else:
                st.write("")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 删除确认提示
        if len(st.session_state.dataset) == 1:
            st.warning("删除当前样本将清空整个数据集")

if __name__ == "__main__":
    main()
