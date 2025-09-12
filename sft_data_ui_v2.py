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

# 默认JSONL文件路径
DEFAULT_JSONL_PATH = "dataset.jsonl"

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
        
        .path-input {
            margin-bottom: 15px;
        }
        
        .format-example {
            background-color: #f6f8fa;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            overflow-x: auto;
            margin-bottom: 10px;
        }
        """
        st.markdown(f"<style>{default_css}</style>", unsafe_allow_html=True)

local_css("style.css")

def clean_html_tags(text):
    """清除HTML标签"""
    if text is None:
        return ""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def markdown_to_html(markdown_text):
    """将Markdown转换为HTML"""
    if markdown_text is None:
        return ""
    clean_text = clean_html_tags(markdown_text)
    html = markdown.markdown(clean_text)
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
    else:
        container_class = "assistant-container"
        message_class = "assistant-message"
        avatar_emoji = "🤖"
        role_display = "助手"
        if 'content' in msg and msg['content']:
            content = msg['content']
        else:
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

def sanitize_filename(name):
    """清理文件名中的非法字符"""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name[:50].strip()
    return name or "对话"

def get_filtered_indices_and_data():
    """返回 (过滤后原始索引列表, 过滤后数据列表)"""
    valid_indices = [
        i for i, item in enumerate(st.session_state.jsonl_data)
        if i not in st.session_state.deleted_indices
    ]
    filtered_data = [st.session_state.jsonl_data[i] for i in valid_indices]
    
    query = st.session_state.get("search_query", "").strip()
    if query:
        new_indices = []
        new_data = []
        for i, item in zip(valid_indices, filtered_data):
            if "text" in item and query.lower() in str(item["text"]).lower():
                new_indices.append(i)
                new_data.append(item)
        valid_indices = new_indices
        filtered_data = new_data
    
    return valid_indices, filtered_data

def get_next_index(current_index, direction):
    """获取下一个有效的原始索引（跳过已删除+满足搜索）"""
    valid_indices, _ = get_filtered_indices_and_data()
    if not valid_indices:
        return current_index

    try:
        current_pos = valid_indices.index(current_index)
    except ValueError:
        return valid_indices[0] if valid_indices else 0

    if direction == "next":
        next_pos = (current_pos + 1) % len(valid_indices)
    else:
        next_pos = (current_pos - 1) % len(valid_indices)

    return valid_indices[next_pos]

def main():
    # 初始化session state
    if "jsonl_data" not in st.session_state:
        st.session_state.jsonl_data = []
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "show_original" not in st.session_state:
        st.session_state.show_original = False
    if "deleted_indices" not in st.session_state:
        st.session_state.deleted_indices = set()
    if "current_file_path" not in st.session_state:
        st.session_state.current_file_path = DEFAULT_JSONL_PATH
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    st.title("📊 JSONL数据集查看器")
    
    # 侧边栏
    with st.sidebar:
        st.header("数据集配置")
        
        new_file_path = st.text_input(
            "数据集路径", 
            value=st.session_state.current_file_path,
            placeholder="请输入JSONL文件的完整路径",
            help="输入JSONL文件的完整路径，然后点击'加载数据集'按钮"
        )
        
        if st.button("📂 加载数据集", use_container_width=True):
            if os.path.exists(new_file_path):
                st.session_state.current_file_path = new_file_path
                st.session_state.jsonl_data = load_jsonl_data(new_file_path)
                st.session_state.current_index = 0
                st.session_state.deleted_indices = set()
                st.session_state.show_original = False
                st.session_state.search_query = ""
                st.success(f"已加载数据集: {new_file_path}")
                st.rerun()
            else:
                st.error(f"文件不存在: {new_file_path}")
        
        # 🔍 搜索框（修复核心：搜索后自动跳转）
        st.markdown("### 🔍 按原文内容筛选")
        search_query = st.text_input(
            "搜索 text 字段",
            value=st.session_state.get("search_query", ""),
            placeholder="输入关键词，按回车生效",
            help="输入关键词，仅显示 text 字段中包含该关键词的样本"
        )

        # 搜索词变化 → 自动跳转第一条
        if search_query != st.session_state.get("search_query", ""):
            st.session_state.search_query = search_query
            valid_indices, _ = get_filtered_indices_and_data()
            if valid_indices:
                st.session_state.current_index = valid_indices[0]
            st.rerun()
        else:
            st.session_state.search_query = search_query

        # 数据格式说明
        with st.expander("📋 数据格式说明", expanded=False):
            st.write("""
            支持两种格式的JSONL文件：
            
            1. **标准格式** - 包含content字段：
            """)
            st.markdown("""
            <div class="format-example">
{"messages": [{"role": "user", "content": "你好，请介绍下你自己"}, {"role": "assistant", "content": "我是AI助手，很高兴为您服务。"}], "text": "用户问候并请求介绍"}
            </div>
            """, unsafe_allow_html=True)
            st.write("""
            2. **思考过程格式** - 包含think和answer字段：
            """)
            st.markdown("""
            <div class="format-example">
{"messages": [{"role": "user", "content": "解释一下量子计算"}, {"role": "assistant", "think": "用户询问量子计算，我需要先解释基本概念，然后说明原理和应用", "answer": "量子计算是一种利用量子力学原理进行计算的技术..."}], "text": "用户询问量子计算解释"}
            </div>
            """, unsafe_allow_html=True)
            st.write("""
            **字段说明**：
            - `messages`: 对话消息列表（必需）
            - `role`: 角色，支持"user"和"assistant"
            - `content`: 消息内容（标准格式）
            - `think`: 思考过程（思考过程格式）
            - `answer`: 最终答案（思考过程格式）
            - `text`: 原文内容（可选）
            """)
        
        st.header("导航控制")
        
        if not st.session_state.jsonl_data:
            st.info("请先加载数据集")
            return

        valid_indices, filtered_data = get_filtered_indices_and_data()
        total_items = len(st.session_state.jsonl_data)
        remaining_items = len(filtered_data)
        deleted_items = total_items - remaining_items

        # 显示当前样本位置
        try:
            current_pos = valid_indices.index(st.session_state.current_index) + 1
        except ValueError:
            current_pos = 1 if valid_indices else 0

        # 统计信息框
        stats_html = f"""
        <div class="stats-box">
            <p><strong>当前文件:</strong> {os.path.basename(st.session_state.current_file_path)}</p>
            <p><strong>总样本数:</strong> {total_items}</p>
            <p><strong>剩余样本数:</strong> {remaining_items}</p>
            <p><strong>已删除样本:</strong> {deleted_items}</p>
            <p><strong>当前样本:</strong> {current_pos}/{remaining_items if remaining_items > 0 else 0}</p>
        """
        if st.session_state.search_query.strip():
            stats_html += f"<p><strong>🔍 搜索中:</strong> “{st.session_state.search_query}”</p>"
        stats_html += "</div>"
        st.markdown(stats_html, unsafe_allow_html=True)
        
        # 导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏪ 上一条", use_container_width=True, disabled=remaining_items == 0):
                st.session_state.current_index = get_next_index(st.session_state.current_index, "previous")
                st.session_state.show_original = False
                st.rerun()
        with col2:
            if st.button("⏩ 下一条", use_container_width=True, disabled=remaining_items == 0):
                st.session_state.current_index = get_next_index(st.session_state.current_index, "next")
                st.session_state.show_original = False
                st.rerun()
        
        # 删除当前样本
        if st.button("🗑️ 删除当前样本", type="primary", use_container_width=True, disabled=remaining_items == 0):
            st.session_state.deleted_indices.add(st.session_state.current_index)
            st.session_state.current_index = get_next_index(st.session_state.current_index, "next")
            st.rerun()
        
        # 查看原文
        if st.button("📄 查看原文", use_container_width=True, disabled=remaining_items == 0):
            st.session_state.show_original = not st.session_state.show_original
            st.rerun()
        
        # 重置删除
        if st.button("🔄 重置删除标记", use_container_width=True, disabled=len(st.session_state.deleted_indices) == 0):
            st.session_state.deleted_indices = set()
            st.rerun()
        
        # 导出数据集
        if st.button("💾 导出数据集", use_container_width=True, disabled=remaining_items == 0):
            temp_file = "filtered_dataset.jsonl"
            save_jsonl_data(temp_file, filtered_data)
            with open(temp_file, "rb") as f:
                st.download_button(
                    label="📥 下载修改后的数据集",
                    data=f,
                    file_name="filtered_dataset.jsonl",
                    mime="application/json",
                    use_container_width=True
                )
        
        # 导出当前对话为MD
        if st.button("📄 导出当前对话为MD", use_container_width=True, disabled=remaining_items == 0):
            current_item = st.session_state.jsonl_data[st.session_state.current_index]
            if "messages" not in current_item:
                st.warning("当前样本无 messages 字段，无法导出")
            else:
                messages = current_item["messages"]
                first_user_msg = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        first_user_msg = msg.get("content", "").strip()
                        break
                filename = f"multi-{sanitize_filename(first_user_msg)}.md"
                md_lines = ["# 蒸馏工具", "自建多轮对话管线", ""]
                round_num = 1
                for msg in messages:
                    if msg.get("role") == "user":
                        content = msg.get("content", "").strip()
                        md_lines.append(f"## 问题{round_num}")
                        md_lines.append(content)
                        md_lines.append("")
                    elif msg.get("role") == "assistant":
                        think = msg.get("think", "").strip()
                        answer = msg.get("answer", "").strip()
                        if think:
                            md_lines.append(f"## 思维链{round_num}")
                            md_lines.append(think)
                            md_lines.append("")
                        if answer:
                            md_lines.append(f"## 回答{round_num}")
                            md_lines.append(answer)
                            md_lines.append("")
                        round_num += 1
                md_content = "\n".join(md_lines)
                st.download_button(
                    label="📥 下载当前对话（MD格式）",
                    data=md_content,
                    file_name=filename,
                    mime="text/markdown",
                    use_container_width=True
                )
    
    # 主内容区域
    if not st.session_state.jsonl_data:
        st.info("请先在侧边栏加载数据集")
        return

    valid_indices, filtered_data = get_filtered_indices_and_data()

    if not filtered_data:
        st.warning("所有样本已被删除或未匹配搜索条件，请重置或调整搜索")
        return

    # 确保当前索引有效
    if st.session_state.current_index not in valid_indices:
        if valid_indices:
            st.session_state.current_index = valid_indices[0]
        else:
            st.warning("无匹配数据")
            return

    current_item = st.session_state.jsonl_data[st.session_state.current_index]

    # 显示当前样本编号
    try:
        current_pos = valid_indices.index(st.session_state.current_index) + 1
    except ValueError:
        current_pos = 1

    st.subheader(f"📝 对话样本 {current_pos}/{len(filtered_data)}")

    if st.session_state.current_index in st.session_state.deleted_indices:
        st.warning("此样本已被标记为删除")

    if "messages" in current_item:
        messages = current_item["messages"]
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                render_message(msg["role"], msg)
    else:
        st.warning("当前样本中没有找到'messages'字段")

    # 显示原文
    if st.session_state.show_original:
        st.subheader("📄 原文内容")
        if "text" in current_item:
            st.text_area("原文", current_item["text"], height=400, key="original_text")
        else:
            st.info("当前样本中没有'text'字段")

if __name__ == "__main__":
    main()
