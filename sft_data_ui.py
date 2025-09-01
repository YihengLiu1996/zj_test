import streamlit as st
import json
import re
from io import BytesIO

# 页面配置
st.set_page_config(
    page_title="JSONL对话数据查看器",
    page_icon="💬",
    layout="wide"
)

# 初始化session状态
if 'data' not in st.session_state:
    st.session_state.data = []
    st.session_state.current_index = 0
    st.session_state.modified = False

def parse_jsonl(file):
    """解析JSONL文件"""
    data = []
    for line in file:
        try:
            item = json.loads(line)
            # 确保messages字段存在且是列表
            if "messages" in item and isinstance(item["messages"], list):
                data.append(item)
        except Exception as e:
            st.error(f"解析错误: {str(e)} - 跳过该行")
            continue
    return data

def render_message(role, content):
    """渲染单条消息，支持Markdown和LaTeX"""
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    elif role == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)
            # 特别处理LaTeX公式
            # 先尝试渲染原生LaTeX，如果失败则尝试转换格式
            try:
                st.latex(content)
            except:
                # 转换 \(...\) 和 \[...\] 为 $...$ 和 $$...$$
                latex_content = re.sub(r'\\\(', '$', re.sub(r'\\\)', '$', content))
                latex_content = re.sub(r'\\\[\s*', '$$', re.sub(r'\s*\\\]', '$$', latex_content))
                st.markdown(latex_content)

def display_conversation():
    """显示当前对话"""
    if not st.session_state.data:
        st.warning("请先上传JSONL文件")
        return
    
    current_item = st.session_state.data[st.session_state.current_index]
    messages = current_item.get("messages", [])
    
    # 过滤掉system消息，只显示user和assistant
    filtered_messages = [m for m in messages if m["role"] in ["user", "assistant"]]
    
    if not filtered_messages:
        st.info("当前样本中没有用户或助手消息（可能只有system消息）")
        with st.expander("查看原始messages内容"):
            st.json(messages)
        return
    
    # 显示对话
    for msg in filtered_messages:
        render_message(msg["role"], msg["content"])

def display_text():
    """显示原文内容"""
    if not st.session_state.data:
        return
    
    current_item = st.session_state.data[st.session_state.current_index]
    text = current_item.get("text", "")
    
    if text:
        st.subheader("原文内容")
        st.text_area("text字段内容", value=text, height=200, disabled=True)
    else:
        st.info("暂无原文内容")

def download_data():
    """生成可下载的JSONL数据"""
    output = BytesIO()
    for item in st.session_state.data:
        output.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))
    return output.getvalue()

# 主界面
st.title("💬 JSONL对话数据查看与编辑器")

# 文件上传区域
uploaded_file = st.file_uploader("上传JSONL文件", type="jsonl")

if uploaded_file:
    # 解析文件
    if not st.session_state.modified:
        st.session_state.data = parse_jsonl(uploaded_file)
        st.session_state.current_index = 0
        st.session_state.modified = False
    
    total_items = len(st.session_state.data)
    
    if total_items == 0:
        st.error("文件中没有有效的对话数据，请检查文件格式")
    else:
        st.success(f"成功加载 {total_items} 条对话数据")
        
        # 显示导航控件
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        
        with col1:
            if st.button("⬅️ 上一条", disabled=st.session_state.current_index <= 0):
                st.session_state.current_index -= 1
                st.experimental_rerun()
        
        with col2:
            st.markdown(
                f"<div style='text-align: center; padding: 8px; background-color: #f0f2f6; border-radius: 4px; margin: 0 10px;'>"
                f"样本 {st.session_state.current_index + 1} / {total_items}"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with col3:
            if st.button("下一条 ➡️", disabled=st.session_state.current_index >= total_items - 1):
                st.session_state.current_index += 1
                st.experimental_rerun()
        
        with col4:
            if st.button("🗑️ 删除当前样本", type="primary"):
                del st.session_state.data[st.session_state.current_index]
                st.session_state.modified = True
                if st.session_state.current_index >= len(st.session_state.data):
                    st.session_state.current_index = max(0, len(st.session_state.data) - 1)
                st.experimental_rerun()
        
        # 显示对话
        st.subheader("对话内容")
        display_conversation()
        
        # 原文查看区域
        st.divider()
        with st.expander("🔍 查看原文"):
            display_text()
        
        # 下载区域
        st.divider()
        st.subheader("导出数据")
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.download_button(
                label="📥 下载修改后的JSONL",
                data=download_data(),
                file_name="modified_data.jsonl",
                mime="application/json",
                disabled=not st.session_state.modified
            )
        
        with col2:
            st.caption("提示：只有修改后才能下载新文件")
        
        # 显示状态
        if st.session_state.modified:
            st.success("数据已修改，可下载更新后的文件")
        else:
            st.info("当前数据未修改，原始文件保持不变")

else:
    st.info("请上传JSONL文件开始使用")
    st.markdown("""
    **文件格式要求：**
    - 每行一个JSON对象
    - 包含messages字段（数组）
    - messages中包含role和content字段
    - 示例：
      ```json
      {"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}
      ```
    """)

# 添加CSS美化
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    .stChatMessage[data-testid="user"] {
        background-color: #e6f7ff;
        border: 1px solid #bae7ff;
    }
    .stChatMessage[data-testid="assistant"] {
        background-color: #f6ffed;
        border: 1px solid #d9f7be;
    }
    .stButton>button {
        width: 100%;
        margin: 5px 0;
    }
    .stDownloadButton>button {
        width: 100%;
        background-color: #28a745;
        color: white;
    }
    .stDownloadButton>button:hover {
        background-color: #218838;
    }
    .stTextInput>div>div>input {
        font-family: monospace;
    }
    .st-expander {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)
