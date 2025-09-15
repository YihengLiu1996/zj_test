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
import concurrent.futures
from threading import Thread
import queue
import mmap

# 配置页面
st.set_page_config(layout="wide", page_title="数据配比工具")
st.title("📊 数据配比分析与调整工具")

# 全局常量
TOKEN_BINS = [
    (0, 4000, "0-4k"),
    (4000, 8000, "4k-8k"),
    (8000, 16000, "8k-16k"),
    (16000, 32000, "16k-32k"),
    (32000, float('inf'), ">32k")
]
GB = 1024 * 1024 * 1024  # 1GB in bytes

# 工具函数
def get_token_bin(token_count):
    """确定token_count所属区间"""
    for low, high, label in TOKEN_BINS:
        if low < token_count <= high:
            return label
    return ">32k"

def calculate_distribution(df, column, weights=None):
    """计算加权分布"""
    if weights is None:
        weights = df['token_count']
    total = weights.sum()
    dist = df.groupby(column).apply(lambda x: np.sum(weights[x.index]) / total)
    return dist.sort_values(ascending=False)

def ipf_solver(df, target_ratios, target_total, max_iter=50, tol=0.01):
    """
    IPF迭代比例拟合求解器
    :param df: 数据DataFrame
    :param target_ratios: 目标比例字典 {维度: {类别: 比例}}
    :param target_total: 目标总token数
    :param max_iter: 最大迭代次数
    :param tol: 误差容忍度(1%)
    :return: 采样权重数组, 实际分布, 是否收敛
    """
    # 初始化权重
    weights = np.ones(len(df))
    total_tokens = df['token_count'].sum()
    
    # 检查目标比例可行性
    for dim, targets in target_ratios.items():
        dim_total = 0
        for cat, ratio in targets.items():
            # 检查该类别在原始数据中是否存在
            if cat not in df[dim].values:
                st.error(f"错误：维度 '{dim}' 中不存在类别 '{cat}'")
                return None, None, False
            
            # 检查目标比例是否超过原始数据最大可能
            orig_ratio = (df[df[dim] == cat]['token_count'].sum() / total_tokens)
            if ratio > orig_ratio * 1.05:  # 允许5%缓冲（IPF可微调）
                st.warning(f"警告：'{dim}'中'{cat}'目标比例({ratio:.2%})超过原始比例({orig_ratio:.2%})，可能无法精确满足")
        
        # 检查维度内比例和
        dim_sum = sum(targets.values())
        if not (0.99 <= dim_sum <= 1.01):
            st.error(f"错误：维度 '{dim}' 的目标比例和({dim_sum:.2%})不在[99%, 101%]范围内")
            return None, None, False
    
    # 开始IPF迭代
    for iter in range(max_iter):
        prev_weights = weights.copy()
        max_error = 0
        
        # 按维度迭代调整
        for dim, targets in target_ratios.items():
            for cat, target_ratio in targets.items():
                # 计算当前维度类别的加权比例
                mask = (df[dim] == cat)
                current_ratio = np.sum(weights[mask] * df.loc[mask, 'token_count']) / np.sum(weights * df['token_count'])
                
                # 计算调整因子（避免除零）
                if current_ratio > 1e-5:
                    factor = target_ratio / current_ratio
                    weights[mask] *= factor
                
                # 记录最大误差
                error = abs(current_ratio - target_ratio)
                max_error = max(max_error, error)
        
        # 检查收敛
        if max_error < tol:
            break
            
        # 检查权重变化
        weight_change = np.mean(np.abs(weights - prev_weights) / (prev_weights + 1e-5))
        if weight_change < 1e-4:
            break
    
    # 缩放至目标总量
    current_total = np.sum(weights * df['token_count'])
    if current_total > 0:
        weights *= (target_total / current_total)
    
    # 计算实际分布（用于验证）
    actual_dist = {}
    for dim in target_ratios.keys():
        actual_dist[dim] = {}
        for cat in target_ratios[dim].keys():
            mask = (df[dim] == cat)
            actual_dist[dim][cat] = np.sum(weights[mask] * df.loc[mask, 'token_count']) / target_total
    
    return weights, actual_dist, (max_error < tol)

def sample_dataset(df, weights, target_total):
    """根据权重进行伯努利采样"""
    # 生成保留概率（截断到[0,1]）
    probs = np.minimum(weights, 1.0)
    
    # 伯努利采样
    retained = np.random.random(len(df)) < probs
    
    # 计算实际采样总量
    sampled_tokens = np.sum(df.loc[retained, 'token_count'])
    
    # 调整采样（确保接近目标总量）
    if sampled_tokens < target_total * 0.95:  # 低于95%时补充
        additional = target_total - sampled_tokens
        remaining = df[~retained].copy()
        remaining['prob'] = (additional * remaining['token_count'] / 
                            remaining['token_count'].sum() / 
                            remaining['token_count'])
        retained[~retained] = np.random.random(len(remaining)) < np.minimum(remaining['prob'], 1.0)
    
    return df[retained].copy()

def export_shards(df, output_path, shard_size_gb=1):
    """分片导出JSONL文件"""
    os.makedirs(output_path, exist_ok=True)
    shard_size_bytes = shard_size_gb * GB
    current_size = 0
    shard_idx = 1
    buffer = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        # 计算当前样本字节数
        sample_bytes = len(row['text'].encode('utf-8')) + 1  # +1 for newline
        
        # 如果当前分片已满，写入文件
        if current_size + sample_bytes > shard_size_bytes and buffer:
            shard_path = os.path.join(output_path, f"shard_{shard_idx:04d}.jsonl")
            with open(shard_path, 'w', encoding='utf-8') as f:
                f.write("".join(buffer))
            buffer = []
            current_size = 0
            shard_idx += 1
        
        # 添加样本到缓冲区
        buffer.append(json.dumps({
            'source': row['source'],
            'category': row['category'],
            'domain': row['domain'],
            'language': row['language'],
            'token_count': row['token_count'],
            'text': row['text']
        }, ensure_ascii=False) + '\n')
        current_size += sample_bytes
        
        # 更新进度
        if idx % 1000 == 0:
            progress = (idx + 1) / len(df)
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"处理样本 {idx+1}/{len(df)} | 当前分片: {shard_idx}")
    
    # 写入最后一个分片
    if buffer:
        shard_path = os.path.join(output_path, f"shard_{shard_idx:04d}.jsonl")
        with open(shard_path, 'w', encoding='utf-8') as f:
            f.write("".join(buffer))
    
    progress_bar.empty()
    status_text.empty()
    st.success(f"导出完成！共 {shard_idx} 个分片，路径: {output_path}")

# ========== 优化的文件加载函数 ==========
def load_file_batched(file, batch_size=1000):
    """批量加载文件（结合内存映射和批量处理）"""
    file_data = []
    try:
        file_size = os.path.getsize(file)
        
        # 对于大文件使用内存映射
        if file_size > 100 * 1024 * 1024:  # >100MB
            with open(file, 'r', encoding='utf-8') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    batch = []
                    valid_count = 0
                    
                    for line in iter(mm.readline, b""):
                        try:
                            sample = json.loads(line.decode('utf-8', errors='replace'))
                            required_fields = ['source', 'category', 'domain', 'language', 'token_count', 'text']
                            if all(k in sample for k in required_fields):
                                try:
                                    token_count = int(float(sample['token_count']))
                                    batch.append({
                                        'source': str(sample['source']),
                                        'category': str(sample['category']),
                                        'domain': str(sample['domain']),
                                        'language': str(sample['language']),
                                        'token_count': token_count,
                                        'text': str(sample['text'])
                                    })
                                    valid_count += 1
                                except (ValueError, TypeError):
                                    pass
                        except json.JSONDecodeError:
                            pass
                        
                        # 批量处理
                        if len(batch) >= batch_size:
                            file_data.extend(batch)
                            batch = []
                    
                    # 处理剩余批次
                    if batch:
                        file_data.extend(batch)
        else:
            # 小文件使用普通读取
            with open(file, 'r', encoding='utf-8') as f:
                batch = []
                valid_count = 0
                
                for line_num, line in enumerate(f):
                    try:
                        sample = json.loads(line)
                        required_fields = ['source', 'category', 'domain', 'language', 'token_count', 'text']
                        if all(k in sample for k in required_fields):
                            try:
                                token_count = int(float(sample['token_count']))
                                batch.append({
                                    'source': str(sample['source']),
                                    'category': str(sample['category']),
                                    'domain': str(sample['domain']),
                                    'language': str(sample['language']),
                                    'token_count': token_count,
                                    'text': str(sample['text'])
                                })
                                valid_count += 1
                            except (ValueError, TypeError):
                                pass
                    except json.JSONDecodeError:
                        pass
                    
                    # 批量处理
                    if len(batch) >= batch_size:
                        file_data.extend(batch)
                        batch = []
                
                # 处理剩余批次
                if batch:
                    file_data.extend(batch)
                    
    except Exception as e:
        pass  # 静默处理错误，避免影响其他文件
    
    return file, file_data, len(file_data)

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
        items = os.listdir(data_path)
        st.sidebar.info(f"包含 {len(items)} 个项目")
        # 显示前5个JSONL文件
        jsonl_count = sum(1 for item in items if item.lower().endswith('.jsonl'))
        st.sidebar.info(f"JSONL文件: {jsonl_count} 个")
    else:
        st.sidebar.error("❌ 路径不存在或无效")

# 加载数据按钮（优化版本）
if st.sidebar.button("📁 加载数据集 (多线程+批量优化)", type="primary"):
    if not data_path:
        st.sidebar.error("❌ 请先输入路径")
    else:
        # 关键修复：规范化路径
        data_path = os.path.normpath(data_path)
        
        start_time = time.time()
        with st.spinner("🔍 正在扫描数据集文件..."):
            try:
                # 扫描文件
                jsonl_files = []
                total_size = 0
                for root, _, files in os.walk(data_path):
                    for file in files:
                        if file.lower().endswith('.jsonl'):
                            full_path = os.path.join(root, file)
                            jsonl_files.append(full_path)
                            total_size += os.path.getsize(full_path)
                
                st.sidebar.info(f"📁 找到 {len(jsonl_files)} 个JSONL文件 | 总大小: {total_size/(1024**3):.1f}GB")
                
                if not jsonl_files:
                    st.sidebar.warning("⚠️ 未找到JSONL文件，请检查：")
                    st.sidebar.caption("- 路径是否正确")
                    st.sidebar.caption("- 文件后缀是否为.jsonl")
                    st.stop()
                
                # 显示文件预览（仅第一个文件）
                sample_file = jsonl_files[0]
                try:
                    with open(sample_file, 'r', encoding='utf-8') as f:
                        sample_lines = [next(f).strip() for _ in range(3)]
                    st.sidebar.caption(f"📄 预览 {os.path.basename(sample_file)}:")
                    for line in sample_lines:
                        st.sidebar.caption(f"`{line[:100]}...`")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ 无法读取示例文件: {str(e)}")
                
                # 多线程并行加载（核心优化）
                all_data = []
                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()
                
                # 动态调整线程数（根据文件数量）
                max_workers = min(32, max(4, len(jsonl_files)))
                st.sidebar.info(f"⚡ 使用 {max_workers} 个线程并行加载")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有任务
                    future_to_file = {executor.submit(load_file_batched, file): file for file in jsonl_files}
                    
                    # 收集结果
                    completed = 0
                    total_valid_samples = 0
                    for future in concurrent.futures.as_completed(future_to_file):
                        file, file_data, valid_count = future.result()
                        all_data.extend(file_data)
                        total_valid_samples += valid_count
                        completed += 1
                        status_text.text(f"✅ 完成 {completed}/{len(jsonl_files)} 文件 | 有效样本: {total_valid_samples:,}")
                        progress_bar.progress(completed / len(jsonl_files))
                
                progress_bar.empty()
                status_text.empty()
                
                if all_data:
                    # 转为DataFrame
                    df = pd.DataFrame(all_data)
                    total_tokens = df['token_count'].sum()
                    
                    # 存储到session state
                    st.session_state.df = df
                    st.session_state.total_tokens = total_tokens
                    st.session_state.token_bins = [get_token_bin(tc) for tc in df['token_count']]
                    
                    # 计算加载性能
                    elapsed = time.time() - start_time
                    speed = len(all_data) / elapsed if elapsed > 0 else 0
                    
                    st.sidebar.success(f"🎉 加载成功！共 {len(df):,} 个有效样本，{total_tokens/1e9:.2f}B tokens")
                    st.sidebar.info(f"⏱️ 耗时: {elapsed:.1f}秒 | 速度: {speed:.0f} 样本/秒")
                else:
                    st.sidebar.error("❌ 未找到有效数据，请检查文件格式")
                    st.sidebar.info("有效JSONL样本示例:")
                    st.sidebar.code('''{"source": "CCI4", "category": "book", "domain": "science", "language": "CN", "token_count": 1234, "text": "示例文本..."}''')
                    st.stop()
                    
            except Exception as e:
                st.sidebar.error(f"加载失败: {str(e)}")
                st.stop()

# 保留原始串行加载作为备选
if st.sidebar.button("📁 加载数据集 (串行-兼容模式)"):
    if not data_path:
        st.sidebar.error("❌ 请先输入路径")
    else:
        data_path = os.path.normpath(data_path)
        
        with st.spinner("🔍 正在扫描数据集文件..."):
            try:
                # 扫描文件
                jsonl_files = []
                for root, _, files in os.walk(data_path):
                    for file in files:
                        if file.lower().endswith('.jsonl'):
                            jsonl_files.append(os.path.join(root, file))
                
                st.sidebar.info(f"📁 找到 {len(jsonl_files)} 个JSONL文件")
                
                if not jsonl_files:
                    st.sidebar.warning("⚠️ 未找到JSONL文件...")
                    st.stop()
                
                # 原始串行加载逻辑
                all_data = []
                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()
                
                for i, file in enumerate(jsonl_files):
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            valid_count = 0
                            for line_num, line in enumerate(f):
                                try:
                                    sample = json.loads(line)
                                    required_fields = ['source', 'category', 'domain', 'language', 'token_count', 'text']
                                    if all(k in sample for k in required_fields):
                                        try:
                                            token_count = int(float(sample['token_count']))
                                            all_data.append({
                                                'source': str(sample['source']),
                                                'category': str(sample['category']),
                                                'domain': str(sample['domain']),
                                                'language': str(sample['language']),
                                                'token_count': token_count,
                                                'text': str(sample['text'])
                                            })
                                            valid_count += 1
                                        except (ValueError, TypeError):
                                            st.sidebar.warning(f"⚠️ 文件 {os.path.basename(file)} 第{line_num}行: token_count非数字")
                                    else:
                                        missing = [k for k in required_fields if k not in sample]
                                        st.sidebar.warning(f"⚠️ 文件 {os.path.basename(file)} 第{line_num}行: 缺少字段 {missing}")
                                except json.JSONDecodeError:
                                    st.sidebar.warning(f"⚠️ 文件 {os.path.basename(file)} 第{line_num}行: JSON解析失败")
                        
                        status_text.text(f"✅ {os.path.basename(file)}: {valid_count} 有效样本")
                    except Exception as e:
                        st.sidebar.error(f"❌ 跳过 {os.path.basename(file)}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(jsonl_files))
                
                progress_bar.empty()
                status_text.empty()
                
                if all_data:
                    df = pd.DataFrame(all_data)
                    total_tokens = df['token_count'].sum()
                    
                    st.session_state.df = df
                    st.session_state.total_tokens = total_tokens
                    st.session_state.token_bins = [get_token_bin(tc) for tc in df['token_count']]
                    
                    st.sidebar.success(f"🎉 加载成功！共 {len(df):,} 个有效样本，{total_tokens/1e9:.2f}B tokens")
                else:
                    st.sidebar.error("❌ 未找到有效数据...")
                    st.stop()
                    
            except Exception as e:
                st.sidebar.exception(f"_fatal error_: {str(e)}")
                st.stop()

# 检查数据是否已加载
if 'df' in st.session_state:
    df = st.session_state.df
    total_tokens = st.session_state.total_tokens
    
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
    
    # 动态生成各维度配比输入
    dimensions = ['source', 'category', 'domain', 'language', 'token_bin']
    target_ratios = {}
    
    for dim in dimensions:
        st.sidebar.subheader(f"{dim.capitalize()} 配比")
        
        # 获取该维度的唯一值
        if dim == 'token_bin':
            values = pd.Series(st.session_state.token_bins).unique()
        else:
            values = df[dim].unique()
        
        # 计算当前分布
        if dim == 'token_bin':
            current_dist = df.groupby(pd.Series(st.session_state.token_bins))['token_count'].sum() / total_tokens
        else:
            current_dist = df.groupby(dim)['token_count'].sum() / total_tokens
        
        # 为每个类别创建输入框
        target_ratios[dim] = {}
        total_ratio = 0.0
        cols = st.sidebar.columns(len(values))
        
        for i, val in enumerate(values):
            current_ratio = current_dist.get(val, 0.0)
            with cols[i % len(cols)]:
                ratio = st.number_input(
                    f"{val}", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=float(current_ratio),
                    step=0.01,
                    key=f"{dim}_{val}"
                )
                target_ratios[dim][val] = ratio
                total_ratio += ratio
        
        # 显示维度内比例和
        st.sidebar.caption(f"当前和: {total_ratio:.2%}")
        if not (0.99 <= total_ratio <= 1.01):
            st.sidebar.warning("比例和应接近100%")
    
    # 应用配比按钮
    if st.sidebar.button("🎯 应用配比", type="primary"):
        with st.spinner("正在计算配比方案..."):
            # 运行IPF求解器
            weights, actual_dist, converged = ipf_solver(
                df, 
                target_ratios, 
                target_total,
                tol=0.01  # 1%误差
            )
            
            if weights is not None:
                # 存储采样结果
                sampled_df = sample_dataset(df, weights, target_total)
                st.session_state.sampled_df = sampled_df
                
                # 显示采样结果
                st.sidebar.success("配比方案已生成！")
                st.sidebar.info(f"实际总量: {sampled_df['token_count'].sum()/1e9:.2f}B tokens")
                
                # 显示关键维度误差
                for dim in ['language', 'domain']:
                    if dim in actual_dist:
                        max_error = 0
                        for cat in actual_dist[dim]:
                            target = target_ratios[dim].get(cat, 0)
                            actual = actual_dist[dim].get(cat, 0)
                            error = abs(target - actual)
                            max_error = max(max_error, error)
                        st.sidebar.caption(f"{dim}: 最大误差 {max_error:.1%}")
    
    # ========== 导出配置 ==========
    st.sidebar.header("📤 导出设置")
    output_path = st.sidebar.text_input("导出路径", value="./balanced_datasets")
    shard_size = st.sidebar.number_input("分片大小 (GB)", min_value=0.1, value=1.0, step=0.1)
    
    if st.sidebar.button("💾 导出配比数据集", type="primary"):
        if 'sampled_df' not in st.session_state:
            st.sidebar.error("请先应用配比方案")
        else:
            with st.spinner("正在导出分片..."):
                export_shards(st.session_state.sampled_df, output_path, shard_size)
    
    # ========== 右侧图表展示 ==========
    st.header("📊 数据分布分析")
    
    # 创建图表布局
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    
    # 1. Source 配比图
    with col1:
        st.subheader("数据来源 (Source) 分布")
        source_dist = calculate_distribution(df, 'source')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(source_dist, labels=source_dist.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
    # 2. Category 配比图
    with col2:
        st.subheader("数据类别 (Category) 分布")
        category_dist = calculate_distribution(df, 'category')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(category_dist, labels=category_dist.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
    # 3. Domain 配比图
    with col3:
        st.subheader("数据领域 (Domain) 分布")
        domain_dist = calculate_distribution(df, 'domain')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(domain_dist, labels=domain_dist.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
    # 4. Language 配比图
    with col4:
        st.subheader("语言 (Language) 分布")
        lang_dist = calculate_distribution(df, 'language')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(lang_dist, labels=lang_dist.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    
    # 5. Token Count 配比图
    with col5:
        st.subheader("Token长度分布")
        df['token_bin'] = st.session_state.token_bins
        token_dist = calculate_distribution(df, 'token_bin')
        
        # 确保所有分组都存在
        for _, _, label in TOKEN_BINS:
            if label not in token_dist:
                token_dist[label] = 0.0
        
        token_dist = token_dist.reindex([label for _, _, label in TOKEN_BINS])
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(token_dist.index, token_dist.values)
        ax.set_ylabel('Ratio')
        ax.set_title('Token length distribution')
        for i, v in enumerate(token_dist.values):
            ax.text(i, v + 0.01, f'{v:.1%}', ha='center')
        st.pyplot(fig)
    
    # 6. 子类分布图
    with col6:
        st.subheader("子类组合分布 (Top 50)")
        # 创建子类组合
        df['subclass'] = df['source'] + "+" + df['category'] + "+" + df['domain'] + "+" + df['language']
        subclass_dist = calculate_distribution(df, 'subclass')
        
        # 取Top 50
        top50 = subclass_dist.head(50)
        
        fig, ax = plt.subplots(figsize=(50, 5))
        ax.barh(top50.index, top50.values)
        ax.set_xlabel('Ratio')
        ax.set_title('Top 50 distribution of subclass combinations')
        
        # 添加比例标签
        for i, v in enumerate(top50.values):
            ax.text(v + 0.005, i, f'{v:.1%}', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # 显示数据摘要
    st.divider()
    st.subheader("🔍 数据摘要")
    st.write(f"**总样本数**: {len(df):,}")
    st.write(f"**总Token数**: {total_tokens/1e9:.2f} B (10亿)")
    st.write(f"**平均Token长度**: {total_tokens/len(df):.0f}")
    
    # 如果有采样数据，显示采样质量
    if 'sampled_df' in st.session_state:
        st.subheader("🎯 采样质量报告")
        sampled_df = st.session_state.sampled_df
        sampled_tokens = sampled_df['token_count'].sum()
        
        st.write(f"**采样总量**: {sampled_tokens/1e9:.2f} B tokens")
        st.write(f"**采样比例**: {len(sampled_df)/len(df):.1%}")
        
        # 比较关键维度
        col1, col2, col3 = st.columns(3)
        for i, dim in enumerate(['language', 'domain', 'source']):
            orig_dist = calculate_distribution(df, dim)
            sampled_dist = calculate_distribution(sampled_df, dim)
            
            # 计算最大误差
            max_error = 0
            for cat in orig_dist.index:
                orig = orig_dist.get(cat, 0)
                sampled = sampled_dist.get(cat, 0)
                error = abs(orig - sampled)
                max_error = max(max_error, error)
            
            if i == 0:
                col1.metric(f"{dim.capitalize()} 最大误差", f"{max_error:.1%}")
            elif i == 1:
                col2.metric(f"{dim.capitalize()} 最大误差", f"{max_error:.1%}")
            else:
                col3.metric(f"{dim.capitalize()} 最大误差", f"{max_error:.1%}")
else:
    st.info("👈 请在左侧输入数据集路径并点击'加载数据集'")
    # 用本地SVG替代网络图片
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
        <svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300">
            <rect width="100%" height="100%" fill="#ffffff"/>
            <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" 
                  font-family="Arial" font-size="20px" fill="#000000">
                数据配比工具
            </text>
            <text x="50%" y="65%" dominant-baseline="middle" text-anchor="middle" 
                  font-family="Arial" font-size="14px" fill="#666666">
                输入数据集路径并点击"加载数据集"
            </text>
        </svg>
    </div>
    """, unsafe_allow_html=True)
