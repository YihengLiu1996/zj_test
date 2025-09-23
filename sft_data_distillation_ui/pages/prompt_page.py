# pages/02_提示词配置页.py

import streamlit as st
from services.prompt_service import PromptService
from config import settings

def main():
    """提示词配置页主函数"""
    # 从Session State获取当前用户
    if settings.SESSION_KEY_CURRENT_USER not in st.session_state:
        st.error("未检测到登录用户，请先登录。")
        return

    user = st.session_state[settings.SESSION_KEY_CURRENT_USER]

    st.title("提示词配置")

    # 加载提示词
    if 'current_prompts' not in st.session_state:
        st.session_state['current_prompts'] = PromptService.load_prompts(user.prompts_path)

    prompts = st.session_state['current_prompts']

    st.info("请在下方编辑您的提示词模板，编辑完成后点击 **保存** 按钮。")

    # 创建一个表单来包含所有提示词编辑器
    with st.form(key="prompt_edit_form"):
        # 为每个提示词创建一个文本区域
        edited_prompts = {}
        for prompt_key in PromptService.PROMPT_KEYS:
            display_name = PromptService.get_prompt_key_display_name(prompt_key)
            current_content = prompts.get(prompt_key, "")
            # 使用 st.text_area 创建可编辑的文本框
            edited_content = st.text_area(
                label=display_name,
                value=current_content,
                height=200,
                key=f"prompt_editor_{prompt_key}",
                help=f"编辑 {display_name}"
            )
            edited_prompts[prompt_key] = edited_content
            # 在每个文本框后添加一个分隔线，提高可读性
            st.markdown("---")

        # 提交按钮
        submitted = st.form_submit_button("💾 保存所有提示词", type="primary")

        if submitted:
            with st.spinner("正在保存提示词..."):
                success = PromptService.save_prompts(user.prompts_path, edited_prompts)
                if success:
                    # 更新Session State
                    st.session_state['current_prompts'] = edited_prompts
                    st.success("✅ 所有提示词已成功保存！")
                else:
                    st.error("❌ 保存失败，部分文件可能无法写入。")

    # 添加一个“重新加载”按钮，允许用户放弃编辑并从文件重新加载
    if st.button("🔄 重新加载提示词（放弃当前编辑）"):
        st.session_state['current_prompts'] = PromptService.load_prompts(user.prompts_path)
        st.rerun()

# 调用主函数
if __name__ == "__main__":
    main()
