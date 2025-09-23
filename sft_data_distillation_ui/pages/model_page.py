# pages/03_大模型配置页.py

import os
import streamlit as st
from openai import AsyncOpenAI
import asyncio
from services.model_config_service import ModelConfigService
from config import settings
from typing import List, Dict, Any, Optional, Union

def main():
    """大模型配置页主函数"""
    # 从Session State获取当前用户
    if settings.SESSION_KEY_CURRENT_USER not in st.session_state:
        st.error("未检测到登录用户，请先登录。")
        return

    user = st.session_state[settings.SESSION_KEY_CURRENT_USER]

    st.title("大模型配置")

    # 初始化Session State
    if 'model_configs' not in st.session_state:
        st.session_state['model_configs'] = ModelConfigService.load_model_config(user.model_config_path)
    if 'editing_index' not in st.session_state:
        st.session_state['editing_index'] = -1 # -1 表示不在编辑状态

    model_configs = st.session_state['model_configs']

    # 显示当前配置
    st.subheader("当前配置的模型")
    if not model_configs:
        st.info("您还没有配置任何模型。请在下方添加。")
    else:
        for idx, config in enumerate(model_configs):
            with st.expander(f"模型 {idx + 1}: {config.get('model_name', '未命名')} ({config.get('model_type', 'unknown')})"):
                st.json(config)
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("✏️ 编辑", key=f"edit_btn_{idx}"):
                        st.session_state['editing_index'] = idx
                        st.rerun()
                with col2:
                    if st.button("🗑️ 删除", key=f"delete_btn_{idx}"):
                        model_configs.pop(idx)
                        st.session_state['model_configs'] = model_configs
                        # 保存到文件
                        success = ModelConfigService.save_model_config(user.model_config_path, model_configs)
                        if success:
                            st.success("模型配置已删除并保存！")
                        else:
                            st.error("保存失败！")
                        st.rerun()

    st.markdown("---")

    # ========== 添加/编辑模型表单 ==========
    st.subheader("添加或编辑模型")

    editing_index = st.session_state['editing_index']
    if editing_index >= 0 and editing_index < len(model_configs):
        current_config = model_configs[editing_index]
        st.info(f"正在编辑模型 {editing_index + 1}")
    else:
        current_config = ModelConfigService.get_default_model_config()
        st.info("正在添加新模型")

    # 使用表单收集用户输入
    with st.form(key="model_config_form"):
        api_path = st.text_input("API Path", value=current_config.get("api_path", ""))
        model_name = st.text_input("Model Name", value=current_config.get("model_name", ""))
        api_key = st.text_input("API Key", value=current_config.get("api_key", ""), type="password")
        concurrency = st.number_input("Concurrency", min_value=1, max_value=100, value=current_config.get("concurrency", 5))
        
        model_type = st.selectbox(
            "Model Type",
            options=["thinking", "instruct", "mix"],
            index=["thinking", "instruct", "mix"].index(current_config.get("model_type", "thinking"))
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            submit_button = st.form_submit_button("💾 保存配置", type="primary")
        with col2:
            test_button = st.form_submit_button("🧪 测试连通性")
        with col3:
            if editing_index >= 0:
                cancel_button = st.form_submit_button("取消编辑")

        if submit_button:
            # 构造新配置
            new_config = {
                "api_path": api_path.strip(),
                "model_name": model_name.strip(),
                "api_key": api_key.strip(),
                "concurrency": int(concurrency),
                "model_type": model_type
            }
            # 验证配置
            errors = ModelConfigService.validate_model_config(new_config)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                if editing_index >= 0:
                    model_configs[editing_index] = new_config
                    action_msg = "更新"
                else:
                    model_configs.append(new_config)
                    action_msg = "添加"

                st.session_state['model_configs'] = model_configs
                # 保存到文件
                success = ModelConfigService.save_model_config(user.model_config_path, model_configs)
                if success:
                    st.success(f"✅ 模型配置已{action_msg}并保存！")
                    # 重置编辑状态
                    st.session_state['editing_index'] = -1
                    st.rerun()
                else:
                    st.error("❌ 保存失败！")

        if test_button:
            # 测试API连通性
            test_config = {
                "api_path": api_path.strip(),
                "model_name": model_name.strip(),
                "api_key": api_key.strip(),
                "concurrency": int(concurrency),
                "model_type": model_type
            }
            errors = ModelConfigService.validate_model_config(test_config)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                with st.spinner("正在测试API连通性..."):
                    is_reachable = asyncio.run(test_model_connectivity(test_config))
                    if is_reachable:
                        st.success("✅ API连通性测试成功！")
                    else:
                        st.error("❌ API连通性测试失败，请检查配置。")

        if 'cancel_button' in locals() and cancel_button:
            st.session_state['editing_index'] = -1
            st.rerun()

    # ========== 添加新模型按钮 ==========
    if editing_index == -1: # 只有在不在编辑状态时才显示
        if st.button("➕ 添加新模型"):
            st.session_state['editing_index'] = -1 # 确保是添加模式
            st.rerun()

# ========================
# 异步连通性测试函数
# ========================
async def test_model_connectivity(config: Dict[str, Any]) -> bool:
    """
    测试模型API的连通性。
    发送一个简单的请求，看是否能成功返回。
    """
    try:
        client = AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config["api_path"]
        )
        # 发送一个非常简单的请求
        completion = await client.chat.completions.create(
            model=config["model_name"],
            messages=[{"role": "user", "content": "Hello, are you there?"}],
            max_tokens=5
        )
        # 只要没有抛出异常，就认为连通性OK
        return True
    except Exception as e:
        print(f"测试模型 {config['model_name']} 连通性时出错: {e}")
        return False

# 调用主函数
if __name__ == "__main__":
    main()
