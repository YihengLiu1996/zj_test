# 极简调试版本 - 仅测试按钮响应和基本状态
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import tempfile
import shutil

# 配置页面
st.set_page_config(layout="wide", page_title="数据配比工具")
st.title("📊 数据配比分析与调整工具 - 极简调试版")

# ========== 左侧配置栏 ==========
st.sidebar.header("🔧 配置面板")
data_path = st.sidebar.text_input("数据集文件夹路径", value="./test_data")

# 初始化处理模式状态
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = "内存模式（小数据）" # 默认值

# 数据处理模式选择 (与 session_state 同步)
selected_mode = st.sidebar.radio(
    "处理模式",
    ["内存模式（小数据）", "流式模式（大数据）"],
    index=0 if st.session_state.processing_mode == "内存模式（小数据）" else 1,
    help="内存模式适用于<100GB数据，流式模式适用于>100GB数据"
)
# 更新 session_state (如果用户改变了选择)
st.session_state.processing_mode = selected_mode

# --- 简化到极致的加载按钮 ---
if st.sidebar.button("📁 加载数据集", type="primary"):
    st.write("DEBUG: Load button clicked!") # <-- 关键调试信息 1
    st.write(f"DEBUG: Current processing mode is '{st.session_state.processing_mode}'") # <-- 关键调试信息 2
    
    # 1. 检查路径
    if not data_path:
        st.sidebar.error("❌ 请先输入路径")
        st.stop() # <-- 确保脚本停止
    
    # 2. 规范化路径
    data_path_normalized = os.path.normpath(data_path)
    st.sidebar.info(f"正在处理路径: {data_path_normalized}")
    st.write(f"DEBUG: Path normalized to '{data_path_normalized}'") # <-- 关键调试信息 3

    # 3. 根据模式执行不同逻辑 (简化)
    try:
        if st.session_state.processing_mode == "内存模式（小数据）":
            st.write("DEBUG: Inside Memory Mode Logic") # <-- 关键调试信息 4
            # --- 极简内存模式逻辑 ---
            # a. 扫描文件 (简化)
            jsonl_files = []
            for root, _, files in os.walk(data_path_normalized):
                for file in files:
                    if file.lower().endswith('.jsonl'):
                        jsonl_files.append(os.path.join(root, file))
            st.sidebar.info(f"📁 找到 {len(jsonl_files)} 个JSONL文件")
            st.write(f"DEBUG: Found {len(jsonl_files)} JSONL files") # <-- 关键调试信息 5
            
            if not jsonl_files:
                 st.sidebar.warning("⚠️ 未找到JSONL文件")
                 st.stop()
            
            # b. 读取一个文件的前几行作为示例 (不并行，不处理全部)
            sample_file = jsonl_files[0]
            sample_data = []
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 5: # 只读5行
                            break
                        try:
                            sample_data.append(json.loads(line))
                        except json.JSONDecodeError:
                            st.sidebar.warning(f"文件 {sample_file} 第 {i+1} 行 JSON 解析失败")
            except Exception as e:
                 st.sidebar.error(f"读取文件 {sample_file} 失败: {e}")
                 st.stop()
            
            st.write("DEBUG: Sample data read successfully") # <-- 关键调试信息 6
            # c. 创建一个非常小的 DataFrame (仅用示例数据)
            if sample_data:
                df_sample = pd.DataFrame(sample_data)
                if 'token_count' in df_sample.columns:
                    df_sample['token_count'] = pd.to_numeric(df_sample['token_count'], errors='coerce').fillna(0).astype(int)
                
                # d. 存储到 session state
                st.session_state.df = df_sample
                st.session_state.total_tokens = df_sample['token_count'].sum() if 'token_count' in df_sample.columns else 0
                st.session_state.processing_mode = "内存模式（小数据）" # 确保状态一致
                
                st.sidebar.success(f"🎉 极简加载成功！示例数据 {len(df_sample)} 行")
                st.write("DEBUG: Data stored in session_state") # <-- 关键调试信息 7
            else:
                st.sidebar.warning("⚠️ 未读取到有效示例数据")
                st.stop()

        else: # 流式模式
            st.write("DEBUG: Inside Streaming Mode Logic") # <-- 关键调试信息 4
            # --- 极简流式模式逻辑 ---
            # a. 初始化 Sampler
            sampler = LargeDataSampler(data_path_normalized)
            # b. 扫描文件
            file_count = sampler.scan_files()
            st.sidebar.info(f"📁 找到 {file_count} 个JSONL文件")
            st.write(f"DEBUG: Streaming mode found {file_count} files") # <-- 关键调试信息 5
            
            if file_count == 0:
                st.sidebar.warning("⚠️ 未找到JSONL文件")
                st.stop()
            
            # c. 计算统计信息 (简化版，只处理一个文件)
            if sampler.jsonl_files:
                stats = sampler._calculate_statistics_single_file(sampler.jsonl_files[0]) # 使用简化方法
                # d. 存储到 session state
                st.session_state.sampler = sampler
                st.session_state.stats = stats
                st.session_state.processing_mode = "流式模式（大数据）" # 确保状态一致
                
                st.sidebar.success(f"🎉 极简流式统计完成！")
                st.write("DEBUG: Streaming stats stored in session_state") # <-- 关键调试信息 7
            else:
                st.sidebar.warning("⚠️ 无文件可统计")
                st.stop()

    except Exception as e:
        st.sidebar.error(f"加载过程中发生错误: {e}")
        st.write(f"DEBUG: Exception during loading: {e}") # <-- 关键调试信息 (如果出错)
        import traceback
        st.code(traceback.format_exc()) # 显示详细错误堆栈
        st.stop()

# --- 简化到极致的 UI 显示 ---
st.header("🔄 简化状态显示")
if 'df' in st.session_state and st.session_state.processing_mode == "内存模式（小数据）":
    st.success("✅ 内存模式数据已加载")
    st.write("**示例数据:**")
    st.dataframe(st.session_state.df.head())
    st.write(f"**总Token数 (示例):** {st.session_state.total_tokens}")

elif 'sampler' in st.session_state and st.session_state.processing_mode == "流式模式（大数据）":
    st.success("✅ 流式模式数据已加载")
    st.write("**示例统计信息:**")
    stats_to_show = st.session_state.stats
    if isinstance(stats_to_show, dict) and 'dimensions' in stats_to_show:
        for dim, counts in list(stats_to_show['dimensions'].items())[:2]: # 只显示前两个维度
            st.write(f"- **{dim}:** {dict(list(counts.items())[:3])}...") # 只显示前3个类别
    else:
        st.write(stats_to_show)

else:
    st.info("👈 请点击左侧 '加载数据集' 按钮")
    st.write("当前处理模式:", st.session_state.processing_mode)
    st.write("Session State Keys:", list(st.session_state.keys()))


# --- 简化版 LargeDataSampler (仅用于调试) ---
class LargeDataSampler:
    """处理大容量数据的采样器 - 简化调试版"""
    def __init__(self, data_path, chunk_size=1000): # 减小 chunk_size 用于调试
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.jsonl_files = []
        self.stats = {}

    def scan_files(self):
        """扫描所有JSONL文件"""
        self.jsonl_files = []
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.lower().endswith('.jsonl'):
                    self.jsonl_files.append(os.path.join(root, file))
        return len(self.jsonl_files)

    def _calculate_statistics_single_file(self, file_path):
        """简化版：只计算单个文件的统计信息"""
        stats = {
            'total_samples': 0,
            'total_tokens': 0,
            'dimensions': defaultdict(lambda: defaultdict(int))
        }
        try:
            chunk_iter = pd.read_json(file_path, lines=True, chunksize=self.chunk_size)
            if not hasattr(chunk_iter, '__iter__'):
                chunk_iter = [chunk_iter]
            
            # 只处理第一个 chunk
            for chunk in chunk_iter:
                 required_fields = ['source', 'category', 'domain', 'language', 'token_count', 'text']
                 if all(col in chunk.columns for col in required_fields):
                     chunk['token_count'] = pd.to_numeric(chunk['token_count'], errors='coerce')
                     chunk.dropna(subset=['token_count'], inplace=True)
                     chunk['token_count'] = chunk['token_count'].astype(int)
                     
                     stats['total_samples'] += len(chunk)
                     stats['total_tokens'] += chunk['token_count'].sum()
                     
                     for dim in ['source', 'category']:
                         dim_counts = chunk[dim].value_counts()
                         for val, count in dim_counts.items():
                             stats['dimensions'][dim][str(val)] += count
                 break # <-- 只处理一个 chunk 就停止
                 
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        # 转换为比例 (简化)
        for dim in stats['dimensions']:
            total = sum(stats['dimensions'][dim].values())
            if total > 0:
                for val in stats['dimensions'][dim]:
                    stats['dimensions'][dim][val] = stats['dimensions'][dim][val] / total
        return stats
