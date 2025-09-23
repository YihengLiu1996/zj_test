# app.py

import streamlit as st
from services.auth_service import AuthService
from models.user import User
from config import settings

# ----------------------------
# 初始化 Session State
# ----------------------------
if settings.SESSION_KEY_LOGGED_IN not in st.session_state:
    st.session_state[settings.SESSION_KEY_LOGGED_IN] = False

if settings.SESSION_KEY_CURRENT_USER not in st.session_state:
    st.session_state[settings.SESSION_KEY_CURRENT_USER] = None

# ----------------------------
# 页面配置
# ----------------------------
st.set_page_config(
    page_title="SFT数据蒸馏平台",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# 登录/注册页面函数
# ----------------------------
def show_login_page():
    """渲染登录和注册页面"""
    st.title("🧊 SFT 数据蒸馏平台")

    tab1, tab2 = st.tabs(["登录", "注册"])

    # ========== 登录 Tab ==========
    with tab1:
        with st.form("login_form"):
            st.subheader("用户登录")
            username = st.text_input("用户名", key="login_username")
            password = st.text_input("密码", type="password", key="login_password")
            submit_button = st.form_submit_button("登录")

            if submit_button:
                if not username or not password:
                    st.error("用户名和密码不能为空")
                else:
                    user = AuthService.authenticate_user(username, password)
                    if user:
                        st.session_state[settings.SESSION_KEY_LOGGED_IN] = True
                        st.session_state[settings.SESSION_KEY_CURRENT_USER] = user
                        st.success(f"欢迎回来，{username}！")
                        st.rerun() # 重新运行应用以进入主界面
                    else:
                        st.error("用户名或密码错误")

    # ========== 注册 Tab ==========
    with tab2:
        with st.form("register_form"):
            st.subheader("新用户注册")
            new_username = st.text_input("设置用户名", key="register_username")
            new_password = st.text_input("设置密码", type="password", key="register_password")
            confirm_password = st.text_input("确认密码", type="password", key="register_confirm_password")
            register_button = st.form_submit_button("注册")

            if register_button:
                if not new_username or not new_password or not confirm_password:
                    st.error("所有字段均为必填")
                elif new_password != confirm_password:
                    st.error("两次输入的密码不一致")
                elif len(new_password) < 6:
                    st.error("密码长度至少为6位")
                else:
                    success = AuthService.register_user(new_username, new_password)
                    if success:
                        st.success("注册成功！请前往登录页进行登录。")
                        # 自动跳转到登录Tab（通过设置query param，但Streamlit原生不支持，这里用提示代替）
                        st.info("已自动切换到登录标签页，请登录。")
                    else:
                        st.error("注册失败。用户名可能已存在或系统错误。")

# ----------------------------
# 主应用页面函数
# ----------------------------
def show_main_app():
    """渲染主应用的四个Tab页"""
    user: User = st.session_state[settings.SESSION_KEY_CURRENT_USER]
    st.title(f"🧊 SFT 数据蒸馏平台 - 欢迎 {user.username}")

    # 创建四个Tab页
    tab_data, tab_prompt, tab_model, tab_distill = st.tabs([
        "数据页", "提示词配置页", "大模型配置页", "数据蒸馏配置页"
    ])

    # ========== 数据页 Tab ==========
    with tab_data:
        from pages import data_page  # 导入数据页模块
        data_page.main()  # 调用其main函数

    # ========== 提示词配置页 Tab ==========
    with tab_prompt:
        from pages import prompt_page  # 导入提示词配置页模块
        prompt_page.main()  # 调用其main函数

    # ========== 大模型配置页 Tab ==========
    with tab_model:
        from pages import model_page  # 导入大模型配置页模块
        model_page.main()  # 调用其main函数

    # ========== 数据蒸馏配置页 Tab ==========
    with tab_distill:
        from pages import distill_page  # 导入数据蒸馏配置页模块
        distill_page.main()  # 调用其main函数

    # ========== 侧边栏：用户信息和登出 ==========
    with st.sidebar:
        st.write(f"**当前用户**: {user.username}")
        if st.button("退出登录"):
            st.session_state[settings.SESSION_KEY_LOGGED_IN] = False
            st.session_state[settings.SESSION_KEY_CURRENT_USER] = None
            st.rerun()

# ----------------------------
# 主逻辑：根据登录状态渲染不同页面
# ----------------------------
def main():
    if st.session_state[settings.SESSION_KEY_LOGGED_IN]:
        show_main_app()
    else:
        show_login_page()

if __name__ == "__main__":
    main()
