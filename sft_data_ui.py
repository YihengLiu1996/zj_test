import streamlit as st
import json
import re
from io import BytesIO
import traceback  # 用于获取完整的错误堆栈

# 页面配置
st.set_page_config(
    page_title="JSONL对话数据查看器 (调试模式)",
    page_icon="🐞",
    layout="wide"
)

# 初始化session状态
if 'data' not in st.session_state:
    st.session_state.data = []
    st.session_state.current_index = 0
    st.session_state.modified = False
    st.session_state.debug_info = []  # 用于存储调试信息

def log_debug(message, level="info"):
    """记录调试信息到session状态"""
    st.session_state.debug_info.append({
        "level": level,
        "message": message,
        "index": len(st.session_state.debug_info)
    })
    # 实时显示最新调试信息（可选）
    if level == "error":
        st.error(f"❌ 调试错误: {message}")
    elif level == "warning":
        st.warning(f"⚠️ 调试警告: {message}")

def show_debug_panel():
    """显示调试面板"""
    with st.expander("🐞 调试信息", expanded=False):
        if not st.session_state.debug_info:
            st.info("没有调试信息")
            return
            
        # 按级别过滤
        level_filter = st.selectbox("过滤级别", ["全部", "error", "warning", "info"], index=0)
        
        # 显示调试信息
        for entry in st.session_state.debug_info:
            if level_filter != "全部" and entry["level"] != level_filter:
                continue
                
            if entry["level"] == "error":
                st.error(f"❌ [{entry['index']}] {entry['message']}")
            elif entry["level"] == "warning":
                st.warning(f"⚠️ [{entry['index']}] {entry['message']}")
            else:
                st.info(f"ℹ️ [{entry['index']}] {entry['message']}")

def parse_jsonl(file):
    """解析JSONL文件 - 增强错误处理"""
    data = []
    log_debug(f"开始解析文件，文件类型: {type(file)}", "info")
    
    try:
        # 尝试读取文件内容用于调试
        file.seek(0)
        sample = file.read(200).decode('utf-8', errors='replace')
        log_debug(f"文件开头示例: {sample}", "info")
        file.seek(0)
    except Exception as e:
        log_debug(f"读取文件示例失败: {str(e)}", "warning")
    
    line_count = 0
    valid_count = 0
    
    try:
        file.seek(0)  # 确保从文件开头开始
        for line in file:
            line_count += 1
            try:
                line_str = line.decode('utf-8').strip()
                if not line_str:  # 跳过空行
                    continue
                    
                item = json.loads(line_str)
                log_debug(f"解析行 {line_count}: {line_str[:50]}...", "info")
                
                # 检查messages字段
                if "messages" not in item:
                    log_debug(f"行 {line_count} 缺少messages字段", "warning")
                    continue
                    
                if not isinstance(item["messages"], list):
                    log_debug(f"行 {line_count} 的messages不是列表类型", "warning")
                    continue
                    
                # 检查messages内容
                for i, msg in enumerate(item["messages"]):
                    if "role" not in msg or "content" not in msg:
                        log_debug(f"行 {line_count} 消息 {i} 缺少必要字段", "warning")
                
                data.append(item)
                valid_count += 1
                
            except Exception as e:
                log_debug(f"解析行 {line_count} 失败: {str(e)}\n内容: {line.decode('utf-8', errors='replace')[:100]}...", "error")
                continue
    except Exception as e:
        log_debug(f"读取文件时发生严重错误: {str(e)}\n{traceback.format_exc()}", "error")
    
    log_debug(f"解析完成: {valid_count}/{line_count} 条有效数据", "info")
    return data

def render_message(role, content):
    """渲染单条消息，支持Markdown和LaTeX - 增强错误处理"""
    try:
        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
                # 尝试渲染LaTeX
                try:
                    # 先尝试直接渲染
                    st.latex(content)
                except:
                    # 转换格式后尝试
                    latex_content = re.sub(r'\\\(', '$', re.sub(r'\\\)', '$', content))
                    latex_content = re.sub(r'\\\[\s*', '$$', re.sub(r'\s*\\\]', '$$', latex_content))
                    st.markdown(latex_content)
        return True
    except Exception as e:
        log_debug(f"渲染消息失败: {str(e)}\n内容: {content[:100]}...", "error")
        st.error(f"消息渲染错误: {str(e)}")
        return False

def display_conversation():
    """显示当前对话 - 增强错误处理"""
    # 检查session状态
    if 'data' not in st.session_state or not st.session_state.data:
        log_debug("session_state中没有数据", "error")
        st.warning("请先上传JSONL文件")
        return False
    
    if st.session_state.current_index >= len(st.session_state.data):
        log_debug(f"当前索引 {st.session_state.current_index} 超出数据范围 {len(st.session_state.data)}", "error")
        st.error("当前索引超出数据范围，请重新上传文件")
        return False
    
    try:
        current_item = st.session_state.data[st.session_state.current_index]
        log_debug(f"显示样本 {st.session_state.current_index}，包含 {len(current_item.get('messages', []))} 条消息", "info")
        
        messages = current_item.get("messages", [])
        if not messages:
            log_debug("当前样本中messages为空", "warning")
            st.info("当前样本中messages字段为空")
            return False
        
        # 检查messages结构
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                log_debug(f"消息 {i} 不是字典类型: {type(msg)}", "error")
            elif "role" not in msg or "content" not in msg:
                log_debug(f"消息 {i} 缺少必要字段: {msg.keys()}", "error")
        
        # 过滤掉system消息，只显示user和assistant
        filtered_messages = [m for m in messages if m.get("role") in ["user", "assistant"]]
        log_debug(f"过滤后保留 {len(filtered_messages)}/{len(messages)} 条消息", "info")
        
        if not filtered_messages:
            log_debug("过滤后没有保留任何消息", "warning")
            st.info("当前样本中没有用户或助手消息（可能只有system消息）")
            with st.expander("查看原始messages内容"):
                st.json(messages)
            return False
        
        # 显示对话
        for msg in filtered_messages:
            if not render_message(msg["role"], msg["content"]):
                log_debug(f"渲染消息失败: {msg}", "error")
                
        return True
    except Exception as e:
        log_debug(f"显示对话时发生错误: {str(e)}\n{traceback.format_exc()}", "error")
        st.error(f"显示对话时发生错误: {str(e)}")
        return False

def display_text():
    """显示原文内容 - 增强错误处理"""
    if 'data' not in st.session_state or not st.session_state.data:
        return
    
    try:
        current_item = st.session_state.data[st.session_state.current_index]
        text = current_item.get("text", "")
        
        if text:
            st.subheader("原文内容")
            st.text_area("text字段内容", value=text, height=200, disabled=True)
        else:
            st.info("暂无原文内容")
    except Exception as e:
        log_debug(f"显示原文时发生错误: {str(e)}", "error")
        st.error(f"显示原文时发生错误: {str(e)}")

def download_data():
    """生成可下载的JSONL数据 - 增强错误处理"""
    try:
        output = BytesIO()
        for i, item in enumerate(st.session_state.data):
            try:
                line = json.dumps(item, ensure_ascii=False) + "\n"
                output.write(line.encode("utf-8"))
            except Exception as e:
                log_debug(f"序列化样本 {i} 失败: {str(e)}", "error")
        return output.getvalue()
    except Exception as e:
        log_debug(f"生成下载数据时发生错误: {str(e)}", "error")
        st.error(f"生成下载数据失败: {str(e)}")
        return b""

# 主界面
st.title("💬 JSONL对话数据查看与编辑器 (调试模式)")

# 显示调试面板
show_debug_panel()

# 文件上传区域
uploaded_file = st.file_uploader("上传JSONL文件", type="jsonl")

if uploaded_file:
    log_debug(f"收到上传文件: {uploaded_file.name}，大小: {uploaded_file.size} 字节", "info")
    
    # 解析文件
    if not st.session_state.modified:
        log_debug("开始解析新上传的文件", "info")
        st.session_state.data = parse_jsonl(uploaded_file)
        st.session_state.current_index = 0
        st.session_state.modified = False
    
    total_items = len(st.session_state.data)
    
    if total_items == 0:
        log_debug("解析后没有有效数据", "error")
        st.error("文件中没有有效的对话数据，请检查文件格式")
        st.info("提示：请确保文件是标准JSONL格式，每行一个包含messages字段的JSON对象")
        
        # 显示文件内容预览
        with st.expander("查看文件内容预览"):
            uploaded_file.seek(0)
            sample = uploaded_file.read(1000).decode('utf-8', errors='replace')
            st.text(sample)
            uploaded_file.seek(0)
    else:
        log_debug(f"成功加载 {total_items} 条对话数据", "info")
        st.success(f"✅ 成功加载 {total_items} 条对话数据")
        
        # 显示导航控件
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        
        with col1:
            if st.button("⬅️ 上一条", disabled=st.session_state.current_index <= 0):
                log_debug(f"导航到上一条: {st.session_state.current_index-1}", "info")
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
                log_debug(f"导航到下一条: {st.session_state.current_index+1}", "info")
                st.session_state.current_index += 1
                st.experimental_rerun()
        
        with col4:
            if st.button("🗑️ 删除当前样本", type="primary"):
                log_debug(f"删除样本 {st.session_state.current_index}", "info")
                del st.session_state.data[st.session_state.current_index]
                st.session_state.modified = True
                if st.session_state.current_index >= len(st.session_state.data):
                    st.session_state.current_index = max(0, len(st.session_state.data) - 1)
                st.experimental_rerun()
        
        # 显示对话
        st.subheader("对话内容")
        if not display_conversation():
            log_debug("对话显示失败", "error")
        
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
    .debug-error { color: #ff4444; }
    .debug-warning { color: #ffaa33; }
    .debug-info { color: #44aaff; }
</style>
""", unsafe_allow_html=True)
