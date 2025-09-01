import streamlit as st
import json

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'show_text' not in st.session_state:
    st.session_state.show_text = False

# Page layout - middle for chat, right for controls
col1, col2 = st.columns([2, 1])

# Right column: Dataset upload and controls
with col2:
    st.header("数据集控制")
    
    # File uploader for JSONL files
    uploaded_file = st.file_uploader("上传 JSONL 文件", type="jsonl", accept_multiple_files=False)
    if uploaded_file is not None:
        # Parse JSONL file
        st.session_state.data = []
        for line in uploaded_file:
            line_str = line.decode('utf-8').strip()
            if line_str:
                try:
                    st.session_state.data.append(json.loads(line_str))
                except json.JSONDecodeError:
                    st.error(f"无法解析JSON行: {line_str}")
        st.session_state.current_index = 0
        st.success(f"成功加载 {len(st.session_state.data)} 条对话数据")
    
    # Display dataset navigation info
    if st.session_state.data:
        total = len(st.session_state.data)
        current = st.session_state.current_index + 1
        st.write(f"总样本数: {total}, 当前: {current}")
        
        # Navigation buttons
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("上一条", use_container_width=True) and st.session_state.current_index > 0:
                st.session_state.current_index -= 1
        with col_next:
            if st.button("下一条", use_container_width=True) and st.session_state.current_index < total - 1:
                st.session_state.current_index += 1
        
        # Delete current sample
        if st.button("删除当前样本", type="primary", use_container_width=True):
            if st.session_state.data:
                del st.session_state.data[st.session_state.current_index]
                if st.session_state.current_index >= len(st.session_state.data):
                    st.session_state.current_index = max(0, len(st.session_state.data) - 1)
                st.rerun()
        
        # View original text button
        if st.button("查看原文", use_container_width=True):
            st.session_state.show_text = not st.session_state.show_text
        
        # Display text if requested
        if st.session_state.show_text and st.session_state.data:
            current_entry = st.session_state.data[st.session_state.current_index]
            if 'text' in current_entry:
                st.text_area("原文内容", current_entry['text'], height=200)
            else:
                st.info("暂无原文")
        
        # Download modified dataset
        jsonl_data = "\n".join([json.dumps(entry, ensure_ascii=False) for entry in st.session_state.data])
        st.download_button(
            label="下载修改后的数据集",
            data=jsonl_data,
            file_name="modified_dataset.jsonl",
            mime="application/json",
            use_container_width=True
        )

# Left column: Chat display
with col1:
    st.header("对话查看器")
    
    if not st.session_state.data:
        st.info("请上传 JSONL 文件以查看对话数据")
        st.markdown("""
        **JSONL 格式示例:**
        ```json
        {"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮助你的？"}]}
        {"messages": [{"role": "user", "content": "如何做蛋糕？"}, {"role": "assistant", "content": "首先准备材料..."}]}
        ```
        """)
    elif 0 <= st.session_state.current_index < len(st.session_state.data):
        current_entry = st.session_state.data[st.session_state.current_index]
        messages = current_entry.get('messages', [])
        
        # Display conversation history
        for msg in messages:
            if msg['role'] in ['user', 'assistant']:
                with st.chat_message(msg['role']):
                    # Render markdown and LaTeX content properly [[5]]
                    st.markdown(msg['content'])
    else:
        st.warning("当前索引超出范围，请使用导航按钮")
