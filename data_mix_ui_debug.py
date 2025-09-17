# 简化版版本2 - 用于定位无响应问题
import streamlit as st

# 配置页面
st.set_page_config(layout="wide", page_title="数据配比工具 - 简化版")
st.title("📊 数据配比分析与调整工具 - 简化版")

# ========== 左侧配置栏 ==========
st.sidebar.header("🔧 配置面板")

# 数据处理模式选择 (确保 Session State 初始化)
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = "内存模式（小数据）" # 默认值

selected_mode = st.sidebar.radio(
    "处理模式",
    ["内存模式（小数据）", "流式模式（大数据）"],
    index=0 if st.session_state.processing_mode == "内存模式（小数据）" else 1,
    help="内存模式适用于<100GB数据，流式模式适用于>100GB数据"
)
# 更新 session_state
st.session_state.processing_mode = selected_mode

# --- 极简测试按钮 ---
if st.sidebar.button("📁 测试加载按钮", type="primary"):
    st.write("✅ '测试加载按钮' 被点击了!")
    st.write(f"当前处理模式 (来自 st.session_state): {st.session_state.processing_mode}")
    st.write(f"当前处理模式 (来自 radio): {selected_mode}")
    
    # 模拟加载成功
    st.session_state.data_loaded = True
    st.sidebar.success("✅ 模拟加载成功!")

if st.sidebar.button("🎯 测试应用配比按钮", type="primary"):
    st.write("✅ '测试应用配比按钮' 被点击了!")
    st.write(f"当前处理模式 (来自 st.session_state): {st.session_state.processing_mode}")

# ========== 右侧状态显示 ==========
st.header("🔄 状态显示")

if st.session_state.processing_mode == "内存模式（小数据）":
    st.info("🟢 当前处于 内存模式（小数据）")
elif st.session_state.processing_mode == "流式模式（大数据）":
    st.info("🔵 当前处于 流式模式（大数据）")
else:
    st.warning(f"⚠️ 未知的处理模式: {st.session_state.processing_mode}")

if 'data_loaded' in st.session_state and st.session_state.data_loaded:
    st.success("🎉 数据已加载 (模拟)")
else:
    st.info("📭 数据未加载")

st.divider()
st.subheader("操作步骤:")
st.write("1. 观察上方的 '当前处理模式' 是否正确。")
st.write("2. 点击侧边栏的 '📁 测试加载按钮'，看右边是否立即出现响应信息。")
st.write("3. 切换 '处理模式'，再点击按钮，看响应信息是否更新。")
st.write("4. 点击 '🎯 测试应用配比按钮'，进行同样测试。")
