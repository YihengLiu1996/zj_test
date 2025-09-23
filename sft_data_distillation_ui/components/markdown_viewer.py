# components/markdown_viewer.py

import streamlit as st

class MarkdownViewerComponent:
    """
    一个用于渲染复杂Markdown内容的Streamlit组件。
    支持标题、表格、公式、代码块等。
    """

    @staticmethod
    def render(content: str):
        """
        渲染Markdown内容。
        Args:
            content (str): 要渲染的Markdown字符串。
        """
        if not content:
            st.info("暂无数据")
            return

        # 使用 st.markdown 并开启 unsafe_allow_html 以支持更复杂的HTML渲染
        # 这对于渲染数学公式（如果公式是LaTeX格式）和复杂表格非常有用
        st.markdown(content, unsafe_allow_html=True)

        # 可选：添加一个分隔线
        st.markdown("---")
