import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from io import StringIO
import time
from scipy.optimize import nnls
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import dask.dataframe as dd
from dask import delayed
import gc
from dask.distributed import Client
import psutil

# 配置Dask调度器
try:
    client = Client(processes=False, threads_per_worker=4, n_workers=2, memory_limit='4GB')
    st.sidebar.success("✅ Dask调度器已启动")
except Exception as e:
    st.sidebar.warning(f"⚠️ Dask调度器启动失败: {str(e)}")
    client = None

# 配置页面
st.set_page_config(layout="wide", page_title="数据配比工具")
st.title("📊 数据配比分析与调整工具（支持TB级数据）")

# 全局常量
TOKEN_BINS = [
    (0, 4096, "0-4k"),
    (4096, 8192, "4k-8k"),
    (8192, 16384, "8k-16k"),
    (16384, 32768, "16k-32k"),
    (32768, float('inf'), ">32k")
]
GB = 1024 * 1024 * 1024  # 1GB in bytes

# 工具函数
def get_token_bin(token_count):
    """确定token_count所属区间"""
    for low, high, label in TOKEN_BINS:
        if low <= token_count < high:
            return label
    return ">32k"

def calculate_distribution_chunked(df_chunk, column, weights_chunk=None):
    """计算分块加权分布"""
    if weights_chunk is None:
        weights_chunk = df_chunk['token_count']
    chunk_total = weights_chunk.sum()
    if chunk_total == 0:
        return pd.Series(dtype=float)
    chunk_dist = df_chunk.groupby(column).apply(lambda x: np.sum(weights_chunk[x.index]) / chunk_total)
    return chunk_dist.sort_values(ascending=False)

@st.cache_data
def calculate_distribution_cached(df_path, column):
    """缓存版本的分布计算（支持大文件）"""
    # 使用Dask读取并计算分布
    df = dd.read_json(df_path, lines=True)
    df['token_bin'] = df['token_count'].apply(get_token_bin, meta=('token_bin', 'object'))
    total_tokens = df['token_count'].sum().compute()
    dist = df.groupby(column)['token_count'].sum().compute() / total_tokens
    return dist.sort_values(ascending=False)

# ========== 全局分布计算函数 ==========
@st.cache_data(ttl=3600)
def compute_global_distribution(df_paths, dimensions):
    """使用 Dask 计算所有维度的确切分布"""
    st.info("🔄 正在计算全局分布...")
    
    # 初始化各维度的统计结果
    results = {}
    
    # 合并所有文件为一个 Dask DataFrame
    ddf_list = [dd.read_json(path, lines=True) for path in df_paths]
    combined_ddf = dd.concat(ddf_list, interleave_partitions=True)
    
    # 添加 token_bin 列
    combined_ddf['token_bin'] = combined_ddf['token_count'].apply(get_token_bin, meta=('token_bin', 'object'))
    
    # 总 token 数
    total_tokens = delayed(sum)([ddf['token_count'].sum() for ddf in ddf_list])
    total_tokens = total_tokens.compute()
    
    # 对每个维度进行分组统计
    for dim in dimensions:
        dist = combined_ddf.groupby(dim)['token_count'].sum().compute() / total_tokens
        results[dim] = dist.sort_values(ascending=False)
    
    st.success("✅ 全局分布计算完成！")
    return results, total_tokens

def advanced_ipf_solver_chunked(df_paths, target_ratios, target_total, priority_order, max_iter=100, tol=0.005):
    """
    改进的IPF求解器 - 支持大文件分块处理
    :param df_paths: 数据文件路径列表
    :param target_ratios: 目标比例字典 {维度: {类别: 比例}}
    :param target_total: 目标总token数
    :param priority_order: 优先级顺序列表
    :param max_iter: 最大迭代次数
    :param tol: 误差容忍度(0.5%)
    :return: 采样权重字典, 实际分布, 是否收敛
    """
    
    # 初始化：计算全局统计信息
    st.info("正在计算全局统计信息...")
    global_stats = {}
    total_tokens = 0
    
    # 分块计算总token数和各维度原始比例
    for dim in target_ratios.keys():
        global_stats[dim] = {}
    
    for file_path in df_paths:
        df_chunk = dd.read_json(file_path, lines=True)
        chunk_total = df_chunk['token_count'].sum().compute()
        total_tokens += chunk_total
        
        for dim in target_ratios.keys():
            dim_stats = df_chunk.groupby(dim)['token_count'].sum().compute()
            for cat, count in dim_stats.items():
                if cat not in global_stats[dim]:
                    global_stats[dim][cat] = 0
                global_stats[dim][cat] += count
    
    # 计算原始比例
    orig_ratios = {}
    for dim, cats in global_stats.items():
        orig_ratios[dim] = {cat: count/total_tokens for cat, count in cats.items()}
    
    # 检查目标比例可行性
    for dim, targets in target_ratios.items():
        for cat, ratio in targets.items():
            if cat not in orig_ratios.get(dim, {}):
                st.error(f"错误：维度 '{dim}' 中不存在类别 '{cat}'")
                return None, None, False
            orig_ratio = orig_ratios[dim][cat]
            if ratio > orig_ratio * 1.05:  # 允许5%缓冲
                st.warning(f"警告：'{dim}'中'{cat}'目标比例({ratio:.2%})可能超过原始比例({orig_ratio:.2%})")
        dim_sum = sum(targets.values())
        if not (0.99 <= dim_sum <= 1.01):
            st.error(f"错误：维度 '{dim}' 的目标比例和({dim_sum:.2%})不在[99%, 101%]范围内")
            return None, None, False

    if set(priority_order) != set(target_ratios.keys()):
        st.error(f"错误：优先级顺序必须包含所有维度且不能重复")
        return None, None, False

    # 初始化权重字典（按文件和索引存储）
    st.info("正在初始化权重...")
    weights_dict = {}  # {file_path: {index: weight}}
    
    # 第一次遍历：初始化权重
    initial_scale_guess = target_total / total_tokens if total_tokens > 0 else 1.0
    for file_path in df_paths:
        df_chunk = dd.read_json(file_path, lines=True)
        # 获取所有分区的索引
        partitions = df_chunk.to_delayed()
        weights_dict[file_path] = {}
        for i, partition in enumerate(partitions):
            partition_df = partition.compute()
            weights_dict[file_path][i] = np.full(len(partition_df), initial_scale_guess)

    # 开始IPF迭代
    all_dims = list(target_ratios.keys())
    
    for iter in range(max_iter):
        st.info(f"正在进行第 {iter+1} 轮迭代...")
        max_errors = {}
        
        # 计算当前总token数
        current_total = 0
        for file_path in df_paths:
            for partition_idx, partition_weights in weights_dict[file_path].items():
                df_partition = dd.read_json(file_path, lines=True).get_partition(partition_idx).compute()
                current_total += np.sum(partition_weights * df_partition['token_count'])
        
        # 计算总量误差因子
        total_factor = (target_total / current_total) if current_total > 1e-5 else 1.0
        total_factor = max(0.8, min(1.2, total_factor))

        # 按优先级顺序迭代调整维度
        for dim in priority_order:
            targets = target_ratios[dim]
            dim_max_error = 0
            
            for cat, target_ratio in targets.items():
                # 计算当前该类别的实际比例
                current_cat_tokens = 0
                for file_path in df_paths:
                    for partition_idx, partition_weights in weights_dict[file_path].items():
                        df_partition = dd.read_json(file_path, lines=True).get_partition(partition_idx).compute()
                        mask = (df_partition[dim] == cat)
                        current_cat_tokens += np.sum(partition_weights[mask] * df_partition.loc[mask, 'token_count'])
                
                current_ratio = current_cat_tokens / current_total if current_total > 1e-5 else 0.0
                
                # 计算比例调整因子
                if current_ratio > 1e-5 and target_ratio > 0:
                    ratio_factor = target_ratio / current_ratio
                    ratio_factor = max(0.7, min(1.4, ratio_factor))
                    combined_factor = ratio_factor * total_factor
                    
                    # 更新权重
                    for file_path in df_paths:
                        for partition_idx, partition_weights in weights_dict[file_path].items():
                            df_partition = dd.read_json(file_path, lines=True).get_partition(partition_idx).compute()
                            mask = (df_partition[dim] == cat)
                            partition_weights[mask] *= combined_factor
                
                # 记录最大误差
                error = abs(current_ratio - target_ratio)
                dim_max_error = max(dim_max_error, error)
            
            max_errors[dim] = dim_max_error

        # 检查所有维度是否都收敛
        if all(error < tol for error in max_errors.values()):
            st.info(f"✅ 所有维度在第 {iter+1} 轮迭代后收敛")
            break

    # 计算最终的实际分布
    st.info("正在计算最终分布...")
    actual_dist = {}
    final_errors = {}
    
    # 最终总量校准
    current_total = 0
    for file_path in df_paths:
        for partition_idx, partition_weights in weights_dict[file_path].items():
            df_partition = dd.read_json(file_path, lines=True).get_partition(partition_idx).compute()
            current_total += np.sum(partition_weights * df_partition['token_count'])
    
    if current_total > 0:
        final_scale_factor = target_total / current_total
        for file_path in df_paths:
            for partition_idx in weights_dict[file_path].keys():
                weights_dict[file_path][partition_idx] *= final_scale_factor
        current_total = target_total

    # 计算各维度实际分布
    for dim in target_ratios.keys():
        actual_dist[dim] = {}
        dim_max_error = 0
        
        for cat in target_ratios[dim].keys():
            actual_cat_tokens = 0
            for file_path in df_paths:
                for partition_idx, partition_weights in weights_dict[file_path].items():
                    df_partition = dd.read_json(file_path, lines=True).get_partition(partition_idx).compute()
                    mask = (df_partition[dim] == cat)
                    actual_cat_tokens += np.sum(partition_weights[mask] * df_partition.loc[mask, 'token_count'])
            
            actual_ratio = actual_cat_tokens / current_total
            actual_dist[dim][cat] = actual_ratio
            target_ratio = target_ratios[dim][cat]
            error = abs(actual_ratio - target_ratio)
            dim_max_error = max(dim_max_error, error)
        
        final_errors[dim] = dim_max_error

    # 显示各维度误差
    st.subheader("📊 各维度配比误差")
    for dim in priority_order:
        error = final_errors[dim]
        if error <= tol:
            st.success(f"✅ {dim}: 最大误差 {error:.3f} ({error*100:.1f}%)")
        else:
            st.warning(f"⚠️ {dim}: 最大误差 {error:.3f} ({error*100:.1f}%)")
            
    is_converged = all(error <= tol for error in final_errors.values())
    return weights_dict, actual_dist, is_converged

def sample_dataset_streaming(df_paths, weights_dict, target_total, output_path, shard_size_gb=1):
    """流式采样并导出"""
    os.makedirs(output_path, exist_ok=True)
    shard_size_bytes = shard_size_gb * GB
    
    current_shard = []
    current_size = 0
    shard_idx = 1
    total_sampled_tokens = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(df_paths)
    
    for file_idx, file_path in enumerate(df_paths):
        df_chunk = dd.read_json(file_path, lines=True)
        partitions = df_chunk.to_delayed()
        
        for partition_idx, partition in enumerate(partitions):
            partition_df = partition.compute()
            partition_weights = weights_dict[file_path][partition_idx]
            
            # 生成保留概率
            probs = np.minimum(partition_weights, 1.0)
            retained = np.random.random(len(partition_df)) < probs
            
            # 处理保留的行
            for idx, row in partition_df[retained].iterrows():
                record = {
                    'source': row['source'],
                    'category': row['category'],
                    'domain': row['domain'],
                    'language': row['language'],
                    'token_count': row['token_count'],
                    'text': row['text']
                }
                
                line = json.dumps(record, ensure_ascii=False) + '\n'
                line_bytes = len(line.encode('utf-8'))
                
                # 检查是否需要创建新分片
                if current_size + line_bytes > shard_size_bytes and current_shard:
                    # 写入当前分片
                    shard_path = os.path.join(output_path, f"shard_{shard_idx:04d}.jsonl")
                    with open(shard_path, 'w', encoding='utf-8') as f:
                        f.writelines(current_shard)
                    
                    current_shard = []
                    current_size = 0
                    shard_idx += 1
                
                current_shard.append(line)
                current_size += line_bytes
                total_sampled_tokens += row['token_count']
            
            # 更新进度
            progress = min((file_idx + (partition_idx + 1) / len(partitions)) / total_files, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"采样进度: {file_idx+1}/{total_files} 文件 | 分片: {shard_idx} | 已采样: {total_sampled_tokens/1e9:.2f}B tokens")
    
    # 写入最后一个分片
    if current_shard:
        shard_path = os.path.join(output_path, f"shard_{shard_idx:04d}.jsonl")
        with open(shard_path, 'w', encoding='utf-8') as f:
            f.writelines(current_shard)
    
    progress_bar.empty()
    status_text.empty()
    
    return total_sampled_tokens

def write_shard_batch(rows, shard_path):
    """批量写入分片文件"""
    try:
        with open(shard_path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps({
                    'source': row['source'],
                    'category': row['category'],
                    'domain': row['domain'],
                    'language': row['language'],
                    'token_count': row['token_count'],
                    'text': row['text']
                }, ensure_ascii=False) + '\n')
        return True, shard_path
    except Exception as e:
        return False, f"Error writing {shard_path}: {str(e)}"

def parse_jsonl_file_pandas(file_path, chunksize=50000):
    """使用pandas高效解析JSONL文件（支持分块读取）"""
    records = []
    try:
        chunk_iter = pd.read_json(file_path, lines=True, chunksize=chunksize)
        if not hasattr(chunk_iter, '__iter__'):
            chunk_iter = [chunk_iter]
        for chunk in chunk_iter:
            required_fields = ['source', 'category', 'domain', 'language', 'token_count', 'text']
            if all(col in chunk.columns for col in required_fields):
                chunk = chunk[required_fields]
                chunk['token_count'] = pd.to_numeric(chunk['token_count'], errors='coerce')
                chunk.dropna(subset=['token_count'], inplace=True)
                chunk['token_count'] = chunk['token_count'].astype(int)
                string_fields = ['source', 'category', 'domain', 'language', 'text']
                for field in string_fields:
                    chunk[field] = chunk[field].astype(str)
                records.extend(chunk.to_dict(orient='records'))
            else:
                print(f"Missing required fields in {file_path}")
                continue
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return records

# ========== 左侧配置栏 ==========
st.sidebar.header("🔧 配置面板")
data_path = st.sidebar.text_input("数据集文件夹路径", value="/path/to/datasets")

# 添加路径诊断工具
if st.sidebar.checkbox("🔍 启用路径诊断", value=False):
    st.sidebar.subheader("路径诊断")
    abs_path = os.path.abspath(data_path) if data_path else ""
    st.sidebar.code(f"绝对路径: {abs_path}")
    if data_path and os.path.exists(data_path):
        st.sidebar.success("✅ 路径存在")
        try:
            files = os.listdir(data_path)
            jsonl_files = [f for f in files if f.lower().endswith('.jsonl')]
            st.sidebar.info(f"包含 {len(files)} 个项目，其中 {len(jsonl_files)} 个JSONL文件")
        except Exception as e:
            st.sidebar.warning(f"无法列出目录内容: {str(e)}")
    else:
        st.sidebar.error("❌ 路径不存在或无效")

# 加载数据按钮
if st.sidebar.button("📁 加载数据集", type="primary"):
    if not data_path:
        st.sidebar.error("❌ 请先输入路径")
    else:
        data_path = os.path.normpath(data_path)
        with st.spinner("🔍 正在扫描数据集文件..."):
            try:
                jsonl_files = []
                for root, _, files in os.walk(data_path):
                    for file in files:
                        if file.lower().endswith('.jsonl'):
                            jsonl_files.append(os.path.join(root, file))
                st.sidebar.info(f"📁 找到 {len(jsonl_files)} 个JSONL文件")
                if not jsonl_files:
                    st.sidebar.warning("⚠️ 未找到JSONL文件，请检查：")
                    st.sidebar.caption("- 路径是否正确")
                    st.sidebar.caption("- 文件后缀是否为.jsonl（非.JSONL）")
                    st.sidebar.caption("- 是否有文件访问权限")
                    st.stop()
                
                # 存储文件路径到session state
                st.session_state.df_paths = jsonl_files
                st.session_state.data_path = data_path
                
                # 计算总样本数和token数（采样估算）
                sample_file = jsonl_files[0]
                try:
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        sample_lines = [next(f).strip() for _ in range(3)]
                    st.sidebar.caption(f"📄 预览 {os.path.basename(sample_file)}:")
                    for line in sample_lines:
                        st.sidebar.caption(f"`{line[:100]}...`")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ 无法读取示例文件: {str(e)}")
                
                # 估算总数据量
                st.sidebar.info("正在估算数据总量...")
                total_samples = 0
                total_tokens = 0
                sample_count = min(10, len(jsonl_files))  # 采样前10个文件
                
                for i, file_path in enumerate(jsonl_files[:sample_count]):
                    try:
                        df_sample = dd.read_json(file_path, lines=True).head(1000)  # 读取前1000行
                        total_samples += len(df_sample)
                        total_tokens += df_sample['token_count'].sum()
                    except Exception as e:
                        st.sidebar.warning(f"无法读取文件 {file_path}: {str(e)}")
                
                if total_samples > 0:
                    avg_tokens_per_sample = total_tokens / total_samples
                    estimated_total_samples = int(total_samples / sample_count * len(jsonl_files))
                    estimated_total_tokens = int(estimated_total_samples * avg_tokens_per_sample)
                    st.session_state.estimated_total_tokens = estimated_total_tokens
                    st.session_state.estimated_total_samples = estimated_total_samples
                    st.sidebar.success(f"🎉 数据扫描完成！")
                    st.sidebar.info(f"估算样本数: {estimated_total_samples:,}")
                    st.sidebar.info(f"估算Token数: {estimated_total_tokens/1e9:.2f}B tokens")
                else:
                    st.sidebar.error("❌ 无法估算数据量")
                    st.stop()
                    
            except Exception as e:
                st.sidebar.exception(f"_fatal error_: {str(e)}")
                st.stop()

# 检查数据是否已加载
if 'df_paths' in st.session_state:
    df_paths = st.session_state.df_paths
    estimated_total_tokens = st.session_state.get('estimated_total_tokens', 0)
    
    # ========== 配比调整配置 ==========
    st.sidebar.header("⚖️ 配比调整")
    
    # 目标总量输入
    target_total_b = st.sidebar.number_input(
        "目标总量 (B tokens)", 
        min_value=0.01, 
        value=1.0, 
        step=0.1,
        help="1B = 10亿tokens"
    )
    target_total = int(target_total_b * 1e9)
    
    # 定义维度
    dimensions = ['source', 'category', 'domain', 'language', 'token_bin']
    
    # 优先级排序输入
    st.sidebar.subheader("📌 优先级排序")
    st.sidebar.info("请按重要性从高到低选择维度")
    
    if 'priority_order' not in st.session_state:
        st.session_state.priority_order = dimensions.copy()
        
    priority_placeholders = {}
    temp_priority_order = st.session_state.priority_order.copy()
    
    for i in range(len(dimensions)):
        with st.sidebar.container():
            cols = st.sidebar.columns([1, 4])
            cols[0].markdown(f"**{i+1}.**")
            available_dims = [dim for dim in dimensions if dim not in temp_priority_order[:i]]
            selected_dim = cols[1].selectbox(
                f"选择第 {i+1} 优先级维度",
                options=available_dims,
                index=available_dims.index(temp_priority_order[i]) if temp_priority_order[i] in available_dims else 0,
                key=f"priority_{i}"
            )
            temp_priority_order[i] = selected_dim
            
    if temp_priority_order != st.session_state.priority_order:
        st.session_state.priority_order = temp_priority_order
        st.rerun()
    
    st.sidebar.caption(f"当前优先级顺序: {' > '.join(st.session_state.priority_order)}")

    # 动态生成各维度配比输入
    target_ratios = {}
    if 'target_ratios' not in st.session_state:
        st.session_state.target_ratios = {}
    
    token_bin_order = [label for _, _, label in TOKEN_BINS]
    
    # 获取维度的唯一值（基于采样数据）
    st.sidebar.info("正在分析维度分布...")
    dimension_values = {}
    for dim in dimensions:
        if dim == 'token_bin':
            dimension_values[dim] = token_bin_order
        else:
            all_values = set()
            sample_files = df_paths[:min(5, len(df_paths))]  # 采样前5个文件
            for file_path in sample_files:
                try:
                    df_sample = dd.read_json(file_path, lines=True)
                    if dim in df_sample.columns:
                        unique_vals = df_sample[dim].drop_duplicates().compute()
                        all_values.update(unique_vals.tolist())
                except Exception as e:
                    st.sidebar.warning(f"读取文件 {file_path} 时出错: {str(e)}")
            dimension_values[dim] = sorted(list(all_values))
    
    for dim in dimensions:
        st.sidebar.subheader(f"{dim.capitalize()} 配比")
        values = dimension_values[dim]
        
        if dim not in st.session_state.target_ratios:
            st.session_state.target_ratios[dim] = {}
        target_ratios[dim] = {}
        total_ratio = 0.0
        
        items_per_row = 3
        for i_start in range(0, len(values), items_per_row):
            cols = st.sidebar.columns(items_per_row)
            for i_offset, val in enumerate(values[i_start:i_start + items_per_row]):
                with cols[i_offset]:
                    ratio = st.number_input(
                        label=f"{val}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.01,
                        format="%.3f",
                        key=f"{dim}_{val}"
                    )
                    st.session_state.target_ratios[dim][val] = ratio
                    target_ratios[dim][val] = ratio
                    total_ratio += ratio
        
        st.sidebar.caption(f"当前和: {total_ratio:.2%}")
        if not (0.99 <= total_ratio <= 1.01):
            st.sidebar.warning("比例和应接近100%")
            
    # 应用配比按钮
    if st.sidebar.button("🎯 应用配比", type="primary"):
        with st.spinner("正在计算配比方案..."):
            target_ratios = st.session_state.target_ratios
            priority_order = st.session_state.priority_order
            
            # 运行改进的IPF求解器
            weights_dict, actual_dist, converged = advanced_ipf_solver_chunked(
                df_paths, 
                target_ratios, 
                target_total,
                priority_order,
                max_iter=50,  # 减少迭代次数以提高速度
                tol=0.01      # 放宽误差容忍度
            )
            
            if weights_dict is not None:
                st.session_state.weights_dict = weights_dict
                st.session_state.actual_dist = actual_dist
                st.session_state.converged = converged
                st.sidebar.success("配比方案已生成！")
                if converged:
                    st.sidebar.success("✅ 所有维度配比均已满足！")
                else:
                    st.sidebar.warning("⚠️ 部分维度配比未完全满足，请检查误差报告")
                    
    # ========== 导出配置 ==========
    st.sidebar.header("📤 导出设置")
    output_path = st.sidebar.text_input("导出路径", value="./balanced_datasets")
    shard_size = st.sidebar.number_input("分片大小 (GB)", min_value=0.1, value=1.0, step=0.1)
    if st.sidebar.button("💾 导出配比数据集", type="primary"):
        if 'weights_dict' not in st.session_state:
            st.sidebar.error("请先应用配比方案")
        else:
            with st.spinner("正在导出分片..."):
                sampled_tokens = sample_dataset_streaming(
                    df_paths, 
                    st.session_state.weights_dict, 
                    target_total, 
                    output_path, 
                    shard_size
                )
                st.sidebar.success(f"导出完成！实际采样: {sampled_tokens/1e9:.2f}B tokens")

    # ========== 右侧图表展示 ==========
    st.header("📊 数据分布分析")
    
    # 调用全局分布计算
    with st.spinner("🔍 正在分析全局分布..."):
        global_dist, actual_total_tokens = compute_global_distribution(df_paths, dimensions)

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    
    # 1. Source 配比图
    with col1:
        st.subheader("数据来源 (Source) 分布")
        try:
            source_dist = global_dist['source']
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(source_dist, labels=source_dist.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"绘图出错: {str(e)}")
    
    # 2. Category 配比图
    with col2:
        st.subheader("数据类别 (Category) 分布")
        try:
            category_dist = global_dist['category']
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(category_dist, labels=category_dist.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"绘图出错: {str(e)}")
    
    # 3. Domain 配比图
    with col3:
        st.subheader("数据领域 (Domain) 分布")
        try:
            domain_dist = global_dist['domain']
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(domain_dist, labels=domain_dist.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"绘图出错: {str(e)}")
    
    # 4. Language 配比图
    with col4:
        st.subheader("语言 (Language) 分布")
        try:
            lang_dist = global_dist['language']
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(lang_dist, labels=lang_dist.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"绘图出错: {str(e)}")
    
    # 5. Token Count 配比图
    with col5:
        st.subheader("Token长度分布")
        try:
            token_dist = global_dist['token_bin']
            ordered_labels = [label for _, _, label in TOKEN_BINS]
            dist_values = [token_dist.get(label, 0) for label in ordered_labels]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(ordered_labels, dist_values)
            ax.set_ylabel('Ratio')
            ax.set_title('Token length distribution')
            for i, v in enumerate(dist_values):
                ax.text(i, v + 0.01, f'{v:.1%}', ha='center')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"绘图出错: {str(e)}")
    
    # 6. 子类组合分布图
    with col6:
        st.subheader("子类组合分布 (Top 10)")
        try:
            subclass_data = []
            for file_path in df_paths[:min(3, len(df_paths))]:  # 控制采样数量
                df_sample = dd.read_json(file_path, lines=True)
                df_sample['subclass'] = df_sample['source'] + "+" + df_sample['category'] + "+" + df_sample['domain'] + "+" + df_sample['language']
                subclass_data.append(df_sample)
            if subclass_data:
                combined_df = dd.concat(subclass_data)
                total_tokens = combined_df['token_count'].sum().compute()
                subclass_dist = combined_df.groupby('subclass')['token_count'].sum().compute() / total_tokens
                top10 = subclass_dist.nlargest(10)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(top10.index, top10.values)
                ax.set_xlabel('比例')
                ax.set_title('Top 10 distribution of subclass combinations')
                for i, v in enumerate(top10.values):
                    ax.text(v + 0.005, i, f'{v:.1%}', va='center')
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"绘图出错: {str(e)}")
    
    # 显示数据摘要
    st.divider()
    st.subheader("🔍 数据摘要")
    st.write(f"**文件数量**: {len(df_paths)}")
    st.write(f"**实际总Token数**: {actual_total_tokens / 1e9:.2f} B (10亿)")
    
else:
    st.info("👈 请在左侧输入数据集路径并点击'加载数据集'")
    st.image("https://docs.streamlit.io/images/brand/streamlit-mark-color.png", width=300)

# 内存监控
if st.sidebar.checkbox("🔍 显示内存使用情况"):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    st.sidebar.info(f"当前内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
    if client:
        st.sidebar.info(f"Dask集群状态: {client.dashboard_link}")
