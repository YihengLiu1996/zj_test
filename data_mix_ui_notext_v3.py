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
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import math
import hashlib
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_balancer')

# 配置页面
st.set_page_config(layout="wide", page_title="数据配比工具")
st.title("📊 数据配比分析与调整工具")

# 全局常量
TOKEN_BINS = [
    (0, 4096, "0-4k"),
    (4096, 8192, "4k-8k"),
    (8192, 16384, "8k-16k"),
    (16384, 32768, "16k-32k"),
    (32768, float('inf'), ">32k")
]
GB = 1024 * 1024 * 1024  # 1GB in bytes

# ========== 核心工具函数 ==========

def get_token_bin(token_count):
    """确定token_count所属区间"""
    for low, high, label in TOKEN_BINS:
        if low <= token_count < high:
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

# ========== 新增：用于缓存的哈希生成器 ==========
def get_dataframe_hash(df):
    """
    生成DataFrame的哈希值，用于图表缓存。
    使用前几行和总行数、总token数来生成一个轻量级哈希。
    """
    sample_str = str(df.head(10).to_dict()) + str(len(df)) + str(df['token_count'].sum())
    return hashlib.md5(sample_str.encode('utf-8')).hexdigest()

@st.cache_data(show_spinner=False)
def calculate_distribution_cached_wrapper(_df, column, weights=None):
    """
    包装器函数，用于缓存分布计算。
    _df 前缀告诉 Streamlit 不要尝试哈希这个参数（我们自己处理）。
    """
    return calculate_distribution(_df, column, weights)

# ========== 改造后的图表函数，全部加入缓存 ==========
@st.cache_data(show_spinner=False)
def get_cached_pie_chart_data(_df, column, cache_key, data_version):
    """缓存饼图数据"""
    return calculate_distribution_cached_wrapper(_df, column)

@st.cache_data(show_spinner=False)
def get_cached_bar_chart_data(_df, column, cache_key, data_version):
    """缓存柱状图数据"""
    return calculate_distribution_cached_wrapper(_df, column)

@st.cache_data(show_spinner=False)
def get_cached_subclass_data(_df, cache_key, data_version):
    """缓存子类组合数据"""
    _df['subclass'] = _df['source'] + "+" + _df['category'] + "+" + _df['domain'] + "+" + _df['language']
    subclass_dist = calculate_distribution_cached_wrapper(_df, 'subclass')
    return subclass_dist.head(10) if not subclass_dist.empty else pd.Series()

def advanced_ipf_solver(df, target_ratios, target_total, priority_order, max_iter=100, tol=0.005):
    """
    改进的IPF求解器 - 支持多维度同时优化、优先级排序，并在迭代中考虑目标总量
    :param df: 数据DataFrame (仅包含元数据)
    :param target_ratios: 目标比例字典 {维度: {类别: 比例}}
    :param target_total: 目标总token数
    :param priority_order: 优先级顺序列表
    :param max_iter: 最大迭代次数
    :param tol: 误差容忍度(0.5%)
    :return: 采样权重数组, 实际分布, 是否收敛
    """
    # 初始化权重: 考虑到目标总量远小于原始总量，初始权重应远小于1
    total_tokens = df['token_count'].sum()
    if total_tokens == 0:
        st.error("错误：数据集中token_count总和为0")
        return None, None, False

    initial_scale_guess = target_total / total_tokens if total_tokens > 0 else 1.0
    weights = np.full(len(df), initial_scale_guess) # 使用 initial_scale_guess 初始化所有权重

    # 检查目标比例可行性
    for dim, targets in target_ratios.items():
        for cat, ratio in targets.items():
            # 检查该类别在原始数据中是否存在
            if cat not in df[dim].values:
                st.error(f"错误：维度 '{dim}' 中不存在类别 '{cat}'")
                return None, None, False
            # 检查目标比例是否超过原始数据最大可能
            orig_ratio = (df[df[dim] == cat]['token_count'].sum() / total_tokens)
            if ratio > orig_ratio * 1.05:  # 允许5%缓冲
                st.warning(f"警告：'{dim}'中'{cat}'目标比例({ratio:.2%})可能超过原始比例({orig_ratio:.2%})")
        # 检查维度内比例和
        dim_sum = sum(targets.values())
        if not (0.99 <= dim_sum <= 1.01):
            st.error(f"错误：维度 '{dim}' 的目标比例和({dim_sum:.2%})不在[99%, 101%]范围内")
            return None, None, False

    # 检查优先级顺序是否完整且唯一
    if set(priority_order) != set(target_ratios.keys()):
        st.error(f"错误：优先级顺序必须包含所有维度且不能重复")
        return None, None, False

    # 开始IPF迭代 (移除维度冻结逻辑，每次都检查所有维度)
    all_dims = list(target_ratios.keys()) # 使用列表保持顺序，虽然这里不关键
    for iter in range(max_iter):
        prev_weights = weights.copy()
        max_errors = {}
        # 计算当前加权总和
        current_total = np.sum(weights * df['token_count'])
        # 计算总量误差因子 (避免除零)
        total_factor = (target_total / current_total) if current_total > 1e-5 else 1.0
        # 限制总量调整幅度，防止剧烈波动
        total_factor = max(0.8, min(1.2, total_factor))

        # 按优先级顺序迭代调整维度
        for dim in priority_order:
            targets = target_ratios[dim]
            dim_max_error = 0
            for cat, target_ratio in targets.items():
                # 计算当前维度类别的加权比例
                mask = (df[dim] == cat)
                # 使用当前的 weights 计算 current_ratio
                current_ratio = np.sum(weights[mask] * df.loc[mask, 'token_count']) / current_total if current_total > 1e-5 else 0.0
                # 计算比例调整因子（避免除零）
                if current_ratio > 1e-5 and target_ratio > 0:
                    ratio_factor = target_ratio / current_ratio
                    # 限制比例调整幅度
                    ratio_factor = max(0.7, min(1.4, ratio_factor))
                    # 更新权重：结合比例因子和总量因子
                    # 这里是关键修改：权重更新同时考虑了比例和总量
                    combined_factor = ratio_factor * total_factor
                    weights[mask] *= combined_factor
                # 记录最大误差
                error = abs(current_ratio - target_ratio)
                dim_max_error = max(dim_max_error, error)
            max_errors[dim] = dim_max_error
            # 注意：不再将维度加入 converged_dims 集合

        # 检查所有维度是否都收敛 (在每次迭代后都检查)
        if all(error < tol for error in max_errors.values()):
            st.info(f"✅ 所有维度在第 {iter+1} 轮迭代后收敛")
            break

        # 检查权重变化
        weight_change = np.mean(np.abs(weights - prev_weights) / (prev_weights + 1e-5))
        if weight_change < 1e-5:
            st.info(f"⚠️ 权重变化过小，在第 {iter+1} 轮迭代后停止")
            break

    # 迭代结束后，进行一次最终的总量校准 (可选，但通常是个好主意)
    # 因为迭代中的 total_factor 是一个近似值
    current_total = np.sum(weights * df['token_count'])
    if current_total > 0:
        final_scale_factor = target_total / current_total
        weights *= final_scale_factor
        # 更新 current_total 以用于后续计算
        current_total = target_total

    # 计算实际分布（用于验证）
    actual_dist = {}
    final_errors = {}
    # 使用最终校准后的 current_total (即 target_total) 来计算实际比例
    for dim in target_ratios.keys():
        actual_dist[dim] = {}
        dim_max_error = 0
        for cat in target_ratios[dim].keys():
            mask = (df[dim] == cat)
            # 使用最终的 weights 和 target_total 计算实际比例
            actual_ratio = np.sum(weights[mask] * df.loc[mask, 'token_count']) / current_total
            actual_dist[dim][cat] = actual_ratio
            target_ratio = target_ratios[dim][cat]
            error = abs(actual_ratio - target_ratio)
            dim_max_error = max(dim_max_error, error)
        final_errors[dim] = dim_max_error

    # 显示各维度误差 (按优先级顺序显示)
    st.subheader("📊 各维度配比误差")
    for dim in priority_order: # 按优先级顺序显示
        error = final_errors[dim]
        if error <= tol:
            st.success(f"✅ {dim}: 最大误差 {error:.3f} ({error*100:.1f}%)")
        else:
            st.warning(f"⚠️ {dim}: 最大误差 {error:.3f} ({error*100:.1f}%)")

    is_converged = all(error <= tol for error in final_errors.values())
    return weights, actual_dist, is_converged

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
        if len(remaining) > 0:
            remaining_prob = (additional * remaining['token_count'] / 
                             remaining['token_count'].sum() if remaining['token_count'].sum() > 0 else 0)
            remaining['prob'] = remaining_prob
            retained[~retained] = np.random.random(len(remaining)) < np.minimum(remaining['prob'], 1.0)

    return df[retained].copy()

# ========== 关键改造：带验证的文本获取与导出 ==========

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
    """带验证的分片导出（保证100%数据准确性） - 支持分片并行写入"""
    os.makedirs(output_path, exist_ok=True)
    shard_size_bytes = shard_size_gb * GB
    current_size = 0
    shard_idx = 1
    buffer = []
    shard_data_list = []  # 存储每个分片的数据列表

    # 创建进度容器
    progress_container = st.empty()
    status_text = st.sidebar.empty()

    total_samples = len(df)
    processed = 0

    # 第一阶段：数据分组（串行，确保顺序和分片大小准确）
    for (file_path, offset), group in df.groupby(['file_path', 'offset']):
        for _, row in group.iterrows():
            # 关键：使用双重验证获取文本
            text = get_verified_text(
                file_path,
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
                # 将当前缓冲区的数据和分片路径加入列表
                shard_path = os.path.join(output_path, f"shard_{shard_idx:04d}.jsonl")
                shard_data_list.append({
                    'shard_path': shard_path,
                    'data_lines': buffer.copy()  # 复制当前缓冲区
                })
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
                    st.caption(f"分组样本 {processed}/{total_samples} | 当前分片: {shard_idx}")
                status_text.text(f"分组进度: {progress:.1%} | 分片: {shard_idx}")

    # 处理最后一个分片
    if buffer:
        shard_path = os.path.join(output_path, f"shard_{shard_idx:04d}.jsonl")
        shard_data_list.append({
            'shard_path': shard_path,
            'data_lines': buffer
        })

    progress_container.empty()
    status_text.empty()
    st.sidebar.info(f"数据分组完成！共需创建 {len(shard_data_list)} 个分片")

    # ========== 第二阶段：并行写入分片 ==========
    if not shard_data_list:
        st.sidebar.warning("无数据可导出")
        return

    # 创建新的进度条用于写入阶段
    write_progress = st.sidebar.progress(0)
    write_status = st.sidebar.empty()

    # 定义单个分片的写入函数
    def write_single_shard(shard_info):
        """写入单个分片文件"""
        shard_path = shard_info['shard_path']
        data_lines = shard_info['data_lines']
        try:
            with open(shard_path, 'w', encoding='utf-8') as f:
                f.writelines(data_lines)
            return True, shard_path, None
        except Exception as e:
            error_msg = f"写入分片失败 {shard_path}: {str(e)}"
            logger.error(error_msg)
            return False, shard_path, error_msg

    # 使用线程池并行写入
    # 线程数设置为 min(32, CPU核心数 * 2)，这是一个经验值，可根据服务器调整
    max_workers = min(32, (os.cpu_count() or 1) * 2)
    success_count = 0
    failed_shards = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有分片写入任务
        future_to_shard = {
            executor.submit(write_single_shard, shard_info): shard_info['shard_path']
            for shard_info in shard_data_list
        }

        # 处理完成的任务
        for i, future in enumerate(as_completed(future_to_shard)):
            success, shard_path, error_msg = future.result()
            if success:
                success_count += 1
            else:
                failed_shards.append(error_msg)

            # 更新写入进度
            progress = (i + 1) / len(shard_data_list)
            write_progress.progress(progress)
            write_status.text(f"写入进度: {i+1}/{len(shard_data_list)} | 成功: {success_count}")

    # 清理进度条
    write_progress.empty()
    write_status.empty()

    # 报告最终结果
    if failed_shards:
        st.sidebar.warning(f"导出完成！成功: {success_count}, 失败: {len(failed_shards)}")
        for error in failed_shards[:5]:  # 只显示前5个错误
            st.sidebar.error(error)
        if len(failed_shards) > 5:
            st.sidebar.error(f"... 还有 {len(failed_shards) - 5} 个错误")
    else:
        st.sidebar.success(f"🎉 导出完成！共 {success_count} 个分片，路径: {output_path}")

# ========== 独立的文件处理函数（修复pickle错误） ==========

def process_file_for_parallel_load(file_path):
    """
    独立的文件处理函数，用于并行加载。
    此函数必须在模块顶层定义，以便被pickle序列化。
    """
    metadata = []
    try:
        with open(file_path, 'rb') as f:  # 必须用二进制模式
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

                            # 只存储元数据和定位信息，不存储text
                            meta = {
                                'id': sample_id,  # 保存UUID（如果存在）
                                'source': str(data['source']),
                                'category': str(data['category']),
                                'domain': str(data['domain']),
                                'language': str(data['language']),
                                'token_count': token_count,
                                'file_path': file_path,  # 记录文件路径
                                'offset': offset,   # 记录文件偏移量
                                'line_hash': line_hash # 记录行哈希，用于验证
                            }
                            metadata.append(meta)
                        except (ValueError, TypeError):
                            pass
                except Exception as e:
                    logger.debug(f"处理文件 {file_path} 偏移量 {offset} 时出错: {str(e)}")

                # 更新偏移量
                offset += len(line)
    except Exception as e:
        logger.exception(f"处理文件 {file_path} 时出错")
        return file_path, str(e), []

    return file_path, None, metadata

# ========== 数据加载函数（改造核心） ==========

def load_dataset_parallel(data_path):
    """并行加载JSONL数据集，仅返回元数据和统计信息（不加载text字段）"""
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
    # 创建进度条和状态文本
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    # 自动确定工作进程数（不超过32，避免过度调度）
    max_workers = min(32, os.cpu_count() or 1)

    # 并行处理 - 使用全局定义的 process_file_for_parallel_load
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        futures = [executor.submit(process_file_for_parallel_load, file) for file in jsonl_files]
        for i, future in enumerate(as_completed(futures)):
            file, error, metadata = future.result()
            if error:
                st.sidebar.warning(f"⚠️ {os.path.basename(file)}: {error}")
            else:
                all_metadata.extend(metadata)
                # 更新进度条和状态文本
                progress = (i + 1) / len(jsonl_files)
                progress_bar.progress(progress)
                status_text.text(f"✅ 处理 {i+1}/{len(jsonl_files)} | 样本: {len(all_metadata):,}")

    # 清理进度条
    progress_bar.empty()
    status_text.empty()

    if not all_metadata:
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
data_path = st.sidebar.text_input("数据集文件夹路径", value="/path/to/datasets")

# 添加路径诊断工具
if st.sidebar.checkbox("🔍 启用路径诊断", value=False):
    st.sidebar.subheader("路径诊断")
    abs_path = os.path.abspath(data_path) if data_path else ""
    st.sidebar.code(f"绝对路径: {abs_path}")
    if data_path and os.path.exists(data_path):
        st.sidebar.success("✅ 路径存在")
        st.sidebar.info(f"包含 {len(os.listdir(data_path))} 个项目")
    else:
        st.sidebar.error("❌ 路径不存在或无效")

# 加载数据按钮
if st.sidebar.button("📁 加载数据集", type="primary"):
    if not data_path:
        st.sidebar.error("❌ 请先输入路径")
    else:
        # 关键修复：规范化路径（解决Windows大小写问题）
        data_path = os.path.normpath(data_path)
        with st.spinner("🔍 正在扫描数据集文件..."):
            try:
                # 调用改造后的加载函数
                result, error = load_dataset_parallel(data_path)
                if error:
                    st.sidebar.error(f"加载失败: {error}")
                else:
                    # 存储到session state
                    st.session_state.df = result['df']
                    st.session_state.total_tokens = result['total_tokens']
                    st.session_state.token_bins = result['token_bins']
                    st.session_state.df['token_bin'] = st.session_state.token_bins

                    # 新增：递增数据版本号，使图表缓存失效
                    if 'data_version' not in st.session_state:
                        st.session_state.data_version = 0
                    st.session_state.data_version += 1

                    st.sidebar.success(f"🎉 加载成功！共 {len(result['df']):,} 个有效样本，{result['total_tokens']/1e9:.2f}B tokens")

                    # 显示ID统计
                    if 'id' in result['df'] and not pd.isna(result['df']['id']).all():
                        unique_ids = result['df']['id'].nunique()
                        total = len(result['df'])
                        st.sidebar.info(f"🔑 唯一ID: {unique_ids:,} / {total:,} ({unique_ids/total:.1%})")

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

    # 定义维度
    dimensions = ['source', 'category', 'domain', 'language', 'token_bin']

    # 优先级排序输入 (新增)
    st.sidebar.subheader("📌 优先级排序")
    st.sidebar.info("请按重要性从高到低拖拽维度")

    # 初始化 session_state 存储优先级顺序
    if 'priority_order' not in st.session_state:
        st.session_state.priority_order = dimensions.copy()

    # 创建可拖拽的优先级排序组件
    # 使用 selectbox 模拟排序（Streamlit 原生不支持拖拽）
    priority_placeholders = {}
    temp_priority_order = st.session_state.priority_order.copy()
    for i in range(len(dimensions)):
        with st.sidebar.container():
            cols = st.sidebar.columns([1, 4])
            cols[0].markdown(f"**{i+1}.**")
            # 创建一个下拉列表，包含所有未被选择的维度
            available_dims = [dim for dim in dimensions if dim not in temp_priority_order[:i]]
            selected_dim = cols[1].selectbox(
                f"选择第 {i+1} 优先级维度",
                options=available_dims,
                index=available_dims.index(temp_priority_order[i]) if temp_priority_order[i] in available_dims else 0,
                key=f"priority_{i}"
            )
            temp_priority_order[i] = selected_dim

    # 更新 session_state 中的优先级顺序
    if temp_priority_order != st.session_state.priority_order:
        st.session_state.priority_order = temp_priority_order
        st.rerun() # 重新运行以更新UI

    st.sidebar.caption(f"当前优先级顺序: {' > '.join(st.session_state.priority_order)}")

    # 动态生成各维度配比输入
    target_ratios = {}

    # 初始化 session_state 存储目标比例
    if 'target_ratios' not in st.session_state:
        st.session_state.target_ratios = {}

    # 获取 token_bin 顺序
    token_bin_order = [label for _, _, label in TOKEN_BINS]

    for dim in dimensions:
        st.sidebar.subheader(f"{dim.capitalize()} 配比")

        # 获取该维度的唯一值（按正确顺序排列）
        if dim == 'token_bin':
            values = sorted(df['token_bin'].unique(), key=lambda x: token_bin_order.index(x) if x in token_bin_order else len(token_bin_order))
        else:
            values = sorted(df[dim].unique())

        # 计算当前分布
        if dim == 'token_bin':
            current_dist = df.groupby('token_bin')['token_count'].sum() / total_tokens
        else:
            current_dist = df.groupby(dim)['token_count'].sum() / total_tokens

        # 为每个类别创建输入框
        if dim not in st.session_state.target_ratios:
            st.session_state.target_ratios[dim] = {}

        target_ratios[dim] = {}
        total_ratio = 0.0

        # 每行最多放 3 个输入框
        items_per_row = 3
        for i_start in range(0, len(values), items_per_row):
            cols = st.sidebar.columns(items_per_row)
            for i_offset, val in enumerate(values[i_start:i_start + items_per_row]):
                current_ratio = current_dist.get(val, 0.0)
                with cols[i_offset]:
                    ratio = st.number_input(
                        label=f"{val}",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(current_ratio),
                        step=0.01,
                        format="%.3f",  # 显示3位小数
                        key=f"{dim}_{val}"
                    )
                    st.session_state.target_ratios[dim][val] = ratio
                    target_ratios[dim][val] = ratio
                    total_ratio += ratio

        # 显示维度内比例和
        st.sidebar.caption(f"当前和: {total_ratio:.2%}")
        if not (0.99 <= total_ratio <= 1.01):
            st.sidebar.warning("比例和应接近100%")

    # 应用配比按钮
    if st.sidebar.button("🎯 应用配比", type="primary"):
        with st.spinner("正在计算配比方案..."):
            # 从 session_state 读取最新的目标比例和优先级顺序
            target_ratios = st.session_state.target_ratios
            priority_order = st.session_state.priority_order

            # 运行改进的IPF求解器
            weights, actual_dist, converged = advanced_ipf_solver(
                df, 
                target_ratios, 
                target_total,
                priority_order, # 传入优先级顺序
                max_iter=100,  # 增加迭代次数
                tol=0.005      # 降低误差容忍度到0.5%
            )

            if weights is not None:
                # 存储采样结果（此时df中仍不包含text字段）
                sampled_df = sample_dataset(df, weights, target_total)
                st.session_state.sampled_df = sampled_df

                # 显示采样结果
                st.sidebar.success("配比方案已生成！")
                st.sidebar.info(f"实际总量: {sampled_df['token_count'].sum()/1e9:.2f}B tokens")

                # 显示收敛状态
                if converged:
                    st.sidebar.success("✅ 所有维度配比均已满足！")
                else:
                    st.sidebar.warning("⚠️ 部分维度配比未完全满足，请检查误差报告")

    # ========== 导出配置 ==========
    st.sidebar.header("📤 导出设置")
    output_path = st.sidebar.text_input("导出路径", value="./balanced_datasets")
    shard_size = st.sidebar.number_input("分片大小 (GB)", min_value=0.1, value=1.0, step=0.1)

    if st.sidebar.button("💾 导出配比数据集", type="primary"):
        if 'sampled_df' not in st.session_state:
            st.sidebar.error("请先应用配比方案")
        else:
            with st.spinner("正在导出分片..."):
                # 调用带验证的导出函数
                export_shards_verified(st.session_state.sampled_df, output_path, shard_size)

    # ========== 右侧图表展示 ==========

    st.header("📊 数据分布分析")

    # 为当前数据生成缓存键
    cache_key = get_dataframe_hash(df)
    # 获取数据版本号
    data_version = st.session_state.get('data_version', 0) # 新增：获取版本号

    # 创建图表布局
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)

    # 1. Source 配比图
    with col1:
        st.subheader("数据来源 (Source) 分布")
        source_dist = get_cached_pie_chart_data(df, 'source', cache_key, data_version)
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
        category_dist = get_cached_pie_chart_data(df, 'category', cache_key, data_version)
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
        domain_dist = get_cached_pie_chart_data(df, 'domain', cache_key, data_version)
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
        lang_dist = get_cached_pie_chart_data(df, 'language', cache_key, data_version)
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
        token_dist = get_cached_bar_chart_data(df, 'token_bin', cache_key, data_version)
        # 确保所有分组都存在并按正确顺序排列
        ordered_labels = [label for _, _, label in TOKEN_BINS]
        for label in ordered_labels:
            if label not in token_dist:
                token_dist[label] = 0.0
        token_dist = token_dist.reindex(ordered_labels)

        if not token_dist.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(token_dist.index, token_dist.values)
            ax.set_ylabel('Ratio')
            ax.set_title('Token length distribution')
            for i, v in enumerate(token_dist.values):
                ax.text(i, v + 0.01, f'{v:.1%}', ha='center')
            st.pyplot(fig)
        else:
            st.info("无token count分布数据")

    # 6. 子类分布图
    with col6:
        st.subheader("子类组合分布 (Top 10)")
        top10 = get_cached_subclass_data(df, cache_key, data_version)
        if not top10.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(top10.index, top10.values)
            ax.set_xlabel('比例')
            ax.set_title('Top 10 distribution of subclass combinations')
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
        st.subheader("📈 原始配比与目标配比偏离分析")
        comparison_cols = st.columns(len(['language', 'domain', 'category', 'token_bin']))
        for i, dim in enumerate(['language', 'domain', 'category', 'token_bin']):
            with comparison_cols[i]:
                orig_dist = get_cached_pie_chart_data(df, dim, cache_key, data_version)
                sampled_dist = get_cached_pie_chart_data(sampled_df, dim, get_dataframe_hash(sampled_df))
                # 计算最大误差
                max_error = 0
                for cat in orig_dist.index:
                    orig = orig_dist.get(cat, 0)
                    sampled = sampled_dist.get(cat, 0)
                    error = abs(orig - sampled)
                    max_error = max(max_error, error)
                st.metric(f"{dim.capitalize()}", f"{max_error:.1%}", "最大偏离")
else:
    st.info("👈 请在左侧输入数据集路径并点击'加载数据集'")
    st.image("https://docs.streamlit.io/images/brand/streamlit-mark-color.png  ", width=300)
