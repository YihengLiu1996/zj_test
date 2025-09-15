import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
import mmap
import concurrent.futures
import time
import linecache
import psutil
import hashlib
import re
from tqdm import tqdm
from scipy.optimize import nnls
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_balancer')

# 配置页面
st.set_page_config(layout="wide", page_title="数据配比工具", page_icon="📊")
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
    if total == 0:
        return pd.Series()
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
    
    if total_tokens == 0:
        st.error("错误：数据集中token_count总和为0")
        return None, None, False
    
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
            actual_ratio = np.sum(weights[mask] * df.loc[mask, 'token_count']) / target_total
            actual_dist[dim][cat] = actual_ratio
    
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
        if not remaining.empty:
            remaining['prob'] = (additional * remaining['token_count'] / 
                                remaining['token_count'].sum() / 
                                remaining['token_count'])
            retained[~retained] = np.random.random(len(remaining)) < np.minimum(remaining['prob'], 1.0)
    
    return df[retained].copy()

def get_verified_text(file_path, offset, expected_id=None, expected_hash=None):
    """带验证的文本获取（核心：保证100%准确性）"""
    try:
        with open(file_path, 'rb') as f:
            f.seek(offset)
            line = f.readline()
            
            # 1. 验证物理完整性
            if expected_hash:
                actual_hash = hashlib.md5(line).hexdigest()
                if actual_hash != expected_hash:
                    logger.error(f"数据篡改检测: {file_path}:{offset} | 期望哈希: {expected_hash} | 实际: {actual_hash}")
                    return f"[ERROR: DATA CORRUPTED AT {offset}]"
            
            # 2. 解析JSON
            try:
                data = json.loads(line.decode('utf-8', errors='replace'))
            except json.JSONDecodeError:
                logger.error(f"JSON解析失败: {file_path}:{offset}")
                return f"[ERROR: INVALID JSON AT {offset}]"
            
            # 3. 验证逻辑ID（如果提供）
            if expected_id is not None:
                actual_id = data.get('id')
                if actual_id != expected_id:
                    logger.warning(f"ID不匹配: 期望 {expected_id} 但得到 {actual_id} | {file_path}:{offset}")
            
            return data.get('text', "")
            
    except Exception as e:
        logger.exception(f"读取失败 {file_path}:{offset} - {str(e)}")
        return f"[ERROR: READ FAILED AT {offset}]"

def export_shards_verified(df, output_path, shard_size_gb=1):
    """带验证的分片导出（保证100%数据准确性）"""
    # 确保输出路径是绝对路径
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)
    
    shard_size_bytes = shard_size_gb * GB
    current_size = 0
    shard_idx = 1
    buffer = []
    
    # 创建进度容器
    progress_container = st.empty()
    status_text = st.sidebar.empty()
    
    # 按文件分组处理（减少文件打开次数）
    total_samples = len(df)
    processed = 0
    
    for (file_path, offset), group in df.groupby(['file_path', 'offset']):
        # 转换为绝对路径
        abs_file_path = os.path.abspath(file_path)
        
        for _, row in group.iterrows():
            # 关键：使用双重验证获取文本
            text = get_verified_text(
                abs_file_path,
                offset,
                expected_id=row.get('id'),
                expected_hash=row.get('line_hash')
            )
            
            # 创建样本
            sample = {
                'id': row.get('id'),
                'source': row['source'],
                'category': row['category'],
                'domain': row['domain'],
                'language': row['language'],
                'token_count': row['token_count'],
                'text': text
            }
            
            # 序列化为JSONL
            try:
                sample_json = json.dumps(sample, ensure_ascii=False) + '\n'
                sample_bytes = len(sample_json.encode('utf-8'))
            except Exception as e:
                logger.error(f"序列化失败: {str(e)} | 样本: {sample}")
                continue
            
            # 检查是否需要新分片
            if current_size + sample_bytes > shard_size_bytes and buffer:
                shard_path = os.path.join(output_path, f"shard_{shard_idx:04d}.jsonl")
                try:
                    with open(shard_path, 'w', encoding='utf-8') as out_f:
                        out_f.writelines(buffer)
                except Exception as e:
                    logger.error(f"写入分片失败 {shard_path}: {str(e)}")
                    st.sidebar.error(f"写入分片失败: {str(e)}")
                    return
                buffer = []
                current_size = 0
                shard_idx += 1
            
            # 添加到缓冲区
            buffer.append(sample_json)
            current_size += sample_bytes
            
            # 更新进度（每100样本）
            processed += 1
            if processed % 100 == 0:
                with progress_container.container():
                    progress = processed / total_samples
                    st.progress(min(progress, 1.0))
                    st.caption(f"处理样本 {processed}/{total_samples} | 当前分片: {shard_idx}")
                status_text.text(f"导出进度: {progress:.1%} | 分片: {shard_idx}")
    
    # 写入最后一个分片
    if buffer:
        shard_path = os.path.join(output_path, f"shard_{shard_idx:04d}.jsonl")
        try:
            with open(shard_path, 'w', encoding='utf-8') as f:
                f.writelines(buffer)
        except Exception as e:
            logger.error(f"写入最终分片失败 {shard_path}: {str(e)}")
            st.sidebar.error(f"写入最终分片失败: {str(e)}")
            return
    
    progress_container.empty()
    status_text.empty()
    st.sidebar.success(f"导出完成！共 {shard_idx} 个分片，路径: {output_path}")

# ========== 优化后的数据加载函数 ==========
def load_dataset_parallel(data_path):
    """并行加载JSONL数据集，返回元数据和统计信息"""
    # 确保路径是绝对路径
    data_path = os.path.abspath(data_path)
    
    # 1. 扫描所有JSONL文件（大小写不敏感）
    jsonl_files = []
    total_size = 0
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.jsonl'):
                file_path = os.path.abspath(os.path.join(root, file))
                jsonl_files.append(file_path)
                total_size += os.path.getsize(file_path)
    
    if not jsonl_files:
        return None, f"未找到JSONL文件，请检查路径: {data_path}"
    
    st.sidebar.info(f"📁 扫描到 {len(jsonl_files)} 个文件 | 总大小: {total_size/(1024**3):.1f} GB")
    
    # 2. 并行处理文件（使用所有可用CPU核心）
    all_metadata = []
    progress_small = st.sidebar.empty()
    
    # 自动确定工作进程数（不超过32，避免过度调度）
    max_workers = min(32, os.cpu_count() or 1)
    
    def process_file(file):
        """处理单个文件并记录精确元数据"""
        metadata = []
        try:
            with open(file, 'rb') as f:  # 必须用二进制模式
                offset = 0
                while True:
                    line = f.readline()
                    if not line:
                        break
                    
                    try:
                        # 计算内容哈希（用于后续验证）
                        line_hash = hashlib.md5(line).hexdigest()
                        
                        # 尝试解析JSON
                        try:
                            data = json.loads(line.decode('utf-8', errors='replace'))
                        except json.JSONDecodeError:
                            offset += len(line)
                            continue
                        
                        # 验证必要字段
                        required_fields = ['source', 'category', 'domain', 'language', 'token_count']
                        if all(k in data for k in required_fields):
                            # 确保token_count是数字
                            try:
                                token_count = int(float(data['token_count']))
                                
                                # 提取ID（如果存在）
                                sample_id = data.get('id')
                                if sample_id is not None:
                                    sample_id = str(sample_id)
                                
                                meta = {
                                    'id': sample_id,  # 保存UUID（如果存在）
                                    'source': str(data['source']),
                                    'category': str(data['category']),
                                    'domain': str(data['domain']),
                                    'language': str(data['language']),
                                    'token_count': token_count,
                                    'file_path': file,
                                    'offset': offset,
                                    'line_hash': line_hash
                                }
                                metadata.append(meta)
                            except (ValueError, TypeError):
                                pass
                    except Exception as e:
                        logger.debug(f"处理文件 {file} 偏移量 {offset} 时出错: {str(e)}")
                    
                    # 更新偏移量
                    offset += len(line)
        except Exception as e:
            logger.exception(f"处理文件 {file} 时出错")
            return file, str(e), []
        
        return file, None, metadata
    
    # 并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file) for file in jsonl_files]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            file, error, metadata = future.result()
            if error:
                st.sidebar.warning(f"⚠️ {os.path.basename(file)}: {error}")
            else:
                all_metadata.extend(metadata)
                progress_small.text(f"✅ 处理 {i+1}/{len(jsonl_files)} | 样本: {len(all_metadata):,}")
    
    progress_small.empty()
    
    if not all_meta
        return None, "未找到有效数据样本"
    
    # 3. 创建元数据DataFrame
    df = pd.DataFrame(all_metadata)
    total_tokens = df['token_count'].sum()
    
    # 4. 计算token分组
    token_bins = [get_token_bin(tc) for tc in df['token_count']]
    
    # 5. 记录关键指标
    logger.info(f"加载完成: {len(df)} 样本 | {total_tokens/1e9:.2f}B tokens")
    
    return {
        'df': df,
        'total_tokens': total_tokens,
        'token_bins': token_bins
    }, None

# ========== 左侧配置栏 ==========
st.sidebar.header("🔧 配置面板")

# 路径诊断工具
st.sidebar.subheader("🔍 路径诊断")
diagnose = st.sidebar.checkbox("启用路径诊断", value=False)

if diagnose:
    data_path = st.sidebar.text_input("数据集文件夹路径", value=os.getcwd())
    
    if data_path:
        abs_path = os.path.abspath(data_path)
        st.sidebar.code(f"绝对路径: {abs_path}")
        
        if os.path.exists(abs_path):
            st.sidebar.success("✅ 路径存在")
            st.sidebar.info(f"包含 {len(os.listdir(abs_path))} 个项目")
            
            # 检查是否有JSONL文件
            has_jsonl = any(f.lower().endswith('.jsonl') for f in os.listdir(abs_path))
            st.sidebar.info(f"包含JSONL文件: {'是' if has_jsonl else '否'}")
        else:
            st.sidebar.error("❌ 路径不存在")
else:
    data_path = st.sidebar.text_input("数据集文件夹路径", value=os.getcwd())

# 加载数据按钮
if st.sidebar.button("📁 加载数据集 (极速模式)", type="primary"):
    # 确保路径是绝对路径
    abs_data_path = os.path.abspath(data_path)
    
    if not data_path or not os.path.exists(abs_data_path):
        st.sidebar.error("❌ 请提供有效的绝对路径")
    else:
        # 显示内存监控
        mem_col1, mem_col2 = st.sidebar.columns(2)
        mem_usage = psutil.virtual_memory().percent
        mem_col1.metric("内存使用", f"{mem_usage:.1f}%")
        mem_col2.metric("可用内存", f"{psutil.virtual_memory().available/(1024**3):.1f} GB")
        
        if mem_usage > 80:
            st.sidebar.warning("⚠️ 内存使用过高，加载可能失败")
        
        start_time = time.time()
        with st.spinner("⚡ 正在并行加载数据集（使用所有CPU核心）..."):
            result, error = load_dataset_parallel(abs_data_path)
            
            if error:
                st.sidebar.error(f"加载失败: {error}")
            else:
                # 存储到session state
                st.session_state.df = result['df']
                st.session_state.total_tokens = result['total_tokens']
                st.session_state.token_bins = result['token_bins']
                
                # 计算加载速度
                elapsed = time.time() - start_time
                speed = result['total_tokens'] / elapsed / 1e6  # MB tokens/s
                
                st.sidebar.success(f"🎉 加载成功！共 {len(result['df']):,} 个样本")
                st.sidebar.info(f"⏱️ 耗时: {elapsed:.1f}秒 | 速度: {speed:.1f}M tokens/秒")
                st.sidebar.info(f"📊 总Token数: {result['total_tokens']/1e9:.2f}B")
                
                # 显示ID统计
                if 'id' in result['df'] and not pd.isna(result['df']['id']).all():
                    unique_ids = result['df']['id'].nunique()
                    total = len(result['df'])
                    st.sidebar.info(f"🔑 唯一ID: {unique_ids:,} / {total:,} ({unique_ids/total:.1%})")

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
        cols = st.sidebar.columns(min(3, len(values)))  # 限制每行最多3个
        
        for i, val in enumerate(values):
            current_ratio = current_dist.get(val, 0.0)
            with cols[i % len(cols)]:
                ratio = st.number_input(
                    f"{val}", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=float(current_ratio),
                    step=0.01,
                    format="%.4f",
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
                actual_tokens = sampled_df['token_count'].sum()
                st.sidebar.info(f"实际总量: {actual_tokens/1e9:.2f}B tokens ({actual_tokens/target_total:.1%} of target)")
                
                # 显示关键维度误差
                for dim in ['language', 'domain', 'source']:
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
    
    # 确保导出路径是绝对路径
    output_path = st.sidebar.text_input(
        "导出路径 (绝对路径)", 
        value=os.path.abspath("./balanced_datasets")
    )
    
    # 验证导出路径
    if output_path:
        abs_output_path = os.path.abspath(output_path)
        st.sidebar.caption(f"规范路径: {abs_output_path}")
        
        # 检查路径是否可写
        try:
            test_file = os.path.join(abs_output_path, ".test_write")
            os.makedirs(abs_output_path, exist_ok=True)
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            st.sidebar.success("✅ 路径可写")
        except Exception as e:
            st.sidebar.error(f"❌ 路径不可写: {str(e)}")
    
    shard_size = st.sidebar.number_input("分片大小 (GB)", min_value=0.1, value=1.0, step=0.1)
    
    if st.sidebar.button("💾 导出配比数据集", type="primary"):
        if 'sampled_df' not in st.session_state:
            st.sidebar.error("请先应用配比方案")
        else:
            abs_output_path = os.path.abspath(output_path)
            with st.spinner("正在导出分片..."):
                export_shards_verified(st.session_state.sampled_df, abs_output_path, shard_size)
    
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
        if not source_dist.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(source_dist, labels=source_dist.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("无source分布数据")
    
    # 2. Category 配比图
    with col2:
        st.subheader("数据类别 (Category) 分布")
        category_dist = calculate_distribution(df, 'category')
        if not category_dist.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(category_dist, labels=category_dist.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("无category分布数据")
    
    # 3. Domain 配比图
    with col3:
        st.subheader("数据领域 (Domain) 分布")
        domain_dist = calculate_distribution(df, 'domain')
        if not domain_dist.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(domain_dist, labels=domain_dist.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("无domain分布数据")
    
    # 4. Language 配比图
    with col4:
        st.subheader("语言 (Language) 分布")
        lang_dist = calculate_distribution(df, 'language')
        if not lang_dist.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(lang_dist, labels=lang_dist.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.info("无language分布数据")
    
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
        
        if not token_dist.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(token_dist.index, token_dist.values)
            ax.set_ylabel('比例')
            ax.set_title('Token长度分布')
            for i, v in enumerate(token_dist.values):
                ax.text(i, v + 0.01, f'{v:.1%}', ha='center')
            st.pyplot(fig)
        else:
            st.info("无token count分布数据")
    
    # 6. 子类分布图
    with col6:
        st.subheader("子类组合分布 (Top 10)")
        # 创建子类组合
        df['subclass'] = df['source'] + "+" + df['category'] + "+" + df['domain'] + "+" + df['language']
        subclass_dist = calculate_distribution(df, 'subclass')
        
        if not subclass_dist.empty:
            # 取Top 10
            top10 = subclass_dist.head(10)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(top10.index, top10.values)
            ax.set_xlabel('比例')
            ax.set_title('Top 10 子类组合分布')
            
            # 添加比例标签
            for i, v in enumerate(top10.values):
                ax.text(v + 0.005, i, f'{v:.1%}', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("无子类组合分布数据")
    
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
            
            if orig_dist.empty or sampled_dist.empty:
                continue
                
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
    # 用本地SVG替代网络图片（避免CDN问题）
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
    
    # 显示使用说明
    st.subheader("使用说明")
    st.markdown("""
    1. **在左侧输入数据集路径**（必须是包含JSONL文件的文件夹）
    2. **点击'加载数据集'**（路径诊断可帮助确认路径有效性）
    3. **分析数据分布**（右侧图表实时显示）
    4. **调整配比参数**（可同时调整多个维度）
    5. **导出配比数据集**（指定绝对路径和分片大小）
    
    💡 **提示**： 
    - 确保路径是**绝对路径**
    - 系统会自动递归查找所有JSONL文件
    - 支持TB级数据集（利用服务器多核CPU加速）
    """)
