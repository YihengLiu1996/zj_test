import streamlit as st
import json
from io import BytesIO, StringIO

def main():
    # 设置页面配置
    st.set_page_config(page_title="JSONL对话查看器", layout="wide")
    st.title("JSONL对话数据查看器")
    
    # 初始化会话状态
    if 'data' not in st.session_state:
        st.session_state.data = []
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'original_data' not in st.session_state:
        st.session_state.original_data = []
    
    # 文件上传
    uploaded_file = st.file_uploader("上传JSONL文件", type=["jsonl"])
    
    if uploaded_file is not None:
        # 读取并解析JSONL文件
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        data = []
        for line in stringio:
            try:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError:
                st.warning(f"解析JSONL时出错，跳过无效行: {line}")
        
        st.session_state.data = data
        st.session_state.original_data = data.copy()
        st.session_state.current_index = 0
        st.success(f"成功加载 {len(data)} 条数据")
    
    # 只有当有数据时才显示后续内容
    if st.session_state.data:
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            # 上一条按钮
            if st.button("上一条"):
                if st.session_state.current_index > 0:
                    st.session_state.current_index -= 1
        
        with col2:
            # 显示当前位置
            st.write(f"当前: {st.session_state.current_index + 1}/{len(st.session_state.data)}")
        
        with col3:
            # 下一条按钮
            if st.button("下一条"):
                if st.session_state.current_index < len(st.session_state.data) - 1:
                    st.session_state.current_index += 1
        
        # 显示当前对话
        st.subheader("对话内容")
        current_item = st.session_state.data[st.session_state.current_index]
        
        if "messages" in current_item:
            for msg in current_item["messages"]:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(msg["content"])
                elif msg["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(msg["content"])
                # 忽略system角色的消息
        else:
            st.warning("当前数据中没有messages字段")
        
        # 查看原文按钮
        with st.expander("查看原文", expanded=False):
            if "text" in current_item:
                st.text_area("原文内容", current_item["text"], height=200)
            else:
                st.info("暂无原文")
        
        # 操作按钮
        col4, col5 = st.columns(2)
        
        with col4:
            # 删除当前样本
            if st.button("删除当前样本", type="primary", use_container_width=True):
                if len(st.session_state.data) > 0:
                    st.session_state.data.pop(st.session_state.current_index)
                    # 调整索引，防止越界
                    if st.session_state.current_index >= len(st.session_state.data) and len(st.session_state.data) > 0:
                        st.session_state.current_index = len(st.session_state.data) - 1
                    st.success("已删除当前样本")
                    st.experimental_rerun()
        
        with col5:
            # 下载修改后的数据集
            if st.button("下载修改后的数据集", use_container_width=True):
                # 将数据转换为JSONL格式
                jsonl_content = "\n".join([json.dumps(item, ensure_ascii=False) for item in st.session_state.data])
                # 创建字节流
                buffer = BytesIO(jsonl_content.encode("utf-8"))
                # 提供下载
                st.download_button(
                    label="点击下载",
                    data=buffer,
                    file_name="modified_data.jsonl",
                    mime="application/jsonl",
                    use_container_width=True
                )
    else:
        st.info("请上传JSONL文件以开始")

if __name__ == "__main__":
    main()
