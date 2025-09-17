import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import tempfile
import shutil

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

class LargeDataSampler:
    """处理大容量数据的采样器"""
    def __init__(self, data_path, chunk_size=50000):
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

    def calculate_statistics(self):
        """计算数据统计信息（分块处理）"""
        st.info("正在计算数据统计信息...")
        # 初始化统计信息
        stats = {
            'total_samples': 0,
            'total_tokens': 0,
            'dimensions': defaultdict(lambda: defaultdict(int))  # {dim: {value: count}}
        }
        progress_bar = st.progress(0)
        status_text = st.empty()
        # 分块读取并统计
        for i, file_path in enumerate(self.jsonl_files):
            try:
                # 分块读取文件
                chunk_iter = pd.read_json(file_path, lines=True, chunksize=self.chunk_size)
                if not hasattr(chunk_iter, '__iter__'):
                    chunk_iter = [chunk_iter]
                for chunk in chunk_iter:
                    required_fields = ['source', 'category', 'domain', 'language', 'token_count', 'text']
                    if all(col in chunk.columns for col in required_fields):
                        # 数据清洗
                        chunk['token_count'] = pd.to_numeric(chunk['token_count'], errors='coerce')
                        chunk.dropna(subset=['token_count'], inplace=True)
                        chunk['token_count'] = chunk['token_count'].astype(int)
                        # 统计信息
                        stats['total_samples'] += len(chunk)
                        stats['total_tokens'] += chunk['token_count'].sum()
                        # 维度统计
                        for dim in ['source', 'category', 'domain', 'language']:
                            dim_counts = chunk[dim].value_counts()
                            for val, count in dim_counts.items():
                                stats['dimensions'][dim][str(val)] += count
                        # token_bin统计
                        chunk['token_bin'] = chunk['token_count'].apply(self.get_token_bin)
                        bin_counts = chunk['token_bin'].value_counts()
                        for val, count in bin_counts.items():
                            stats['dimensions']['token_bin'][val] += count
                status_text.text(f"已处理 {i+1}/{len(self.jsonl_files)} 个文件")
                progress_bar.progress((i + 1) / len(self.jsonl_files))
            except Exception as e:
                st.warning(f"处理文件 {file_path} 时出错: {str(e)}")
                continue
        progress_bar.empty()
        status_text.empty()
        # 转换为比例
        for dim in stats['dimensions']:
            total = sum(stats['dimensions'][dim].values())
            for val in stats['dimensions'][dim]:
                stats['dimensions'][dim][val] = stats['dimensions'][dim][val] / total
        self.stats = stats
        return stats

    def get_token_bin(self, token_count):
        """确定token_count所属区间"""
        for low, high, label in TOKEN_BINS:
            if low <= token_count < high:
                return label
        return ">32k"

    def sample_data_streaming(self, target_ratios, target_total_tokens, output_path, shard_size_gb=1):
        """流式采样大数据集"""
        st.info("开始流式采样...")
        # 计算每个维度的采样权重
        weights = self.calculate_sampling_weights(target_ratios)
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        # 流式处理和采样
        total_written_tokens = 0
        shard_idx = 1
        current_shard_data = []
        current_shard_size = 0
        shard_size_bytes = shard_size_gb * GB
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_files = len(self.jsonl_files)
        for file_idx, file_path in enumerate(self.jsonl_files):
            try:
                # 分块读取文件
                chunk_iter = pd.read_json(file_path, lines=True, chunksize=self.chunk_size)
                if not hasattr(chunk_iter, '__iter__'):
                    chunk_iter = [chunk_iter]
                for chunk in chunk_iter:
                    required_fields = ['source', 'category', 'domain', 'language', 'token_count', 'text']
                    if all(col in chunk.columns for col in required_fields):
                        # 数据清洗
                        chunk['token_count'] = pd.to_numeric(chunk['token_count'], errors='coerce')
                        chunk.dropna(subset=['token_count'], inplace=True)
                        chunk['token_count'] = chunk['token_count'].astype(int)
                        chunk['token_bin'] = chunk['token_count'].apply(self.get_token_bin)
                        # 为每行计算采样概率
                        chunk['sample_prob'] = chunk.apply(
                            lambda row: self.calculate_row_probability(row, weights), axis=1
                        )
                        # 采样
                        sampled_chunk = chunk[np.random.random(len(chunk)) < chunk['sample_prob']]
                        # 写入分片
                        for _, row in sampled_chunk.iterrows():
                            sample_bytes = len(row['text'].encode('utf-8')) + 1
                            # 如果当前分片已满，写入文件
                            if current_shard_size + sample_bytes > shard_size_bytes and current_shard_data:
                                shard_path = os.path.join(output_path, f"shard_{shard_idx:04d}.jsonl")
                                self.write_shard(current_shard_data, shard_path)
                                current_shard_data = []
                                current_shard_size = 0
                                shard_idx += 1
                                # 注意：这里应该是累加当前分片的数据量，而不是累加空列表
                                # 修复：正确计算已写入的token数
                                total_written_tokens += sum(item['token_count'] for item in current_shard_data)
                            # 添加到当前分片
                            current_shard_data.append({
                                'source': str(row['source']),
                                'category': str(row['category']),
                                'domain': str(row['domain']),
                                'language': str(row['language']),
                                'token_count': int(row['token_count']),
                                'text': str(row['text'])
                            })
                            current_shard_size += sample_bytes
                progress = (file_idx + 1) / total_files
                progress_bar.progress(progress)
                status_text.text(f"处理进度: {file_idx+1}/{total_files} 文件 | 已写入: {total_written_tokens/1e9:.2f}B tokens")
            except Exception as e:
                st.warning(f"处理文件 {file_path} 时出错: {str(e)}")
                continue
        # 写入最后一个分片
        if current_shard_data:
            shard_path = os.path.join(output_path, f"shard_{shard_idx:04d}.jsonl")
            self.write_shard(current_shard_data, shard_path)
            # 修复：在写入最后一个分片后，更新总写入token数
            total_written_tokens += sum(item['token_count'] for item in current_shard_data)
        progress_bar.empty()
        status_text.empty()
        st.success(f"采样完成！共写入 {total_written_tokens/1e9:.2f}B tokens，{shard_idx} 个分片")
        return total_written_tokens

    def calculate_sampling_weights(self, target_ratios):
        """计算采样权重"""
        weights = {}
        for dim, targets in target_ratios.items():
            weights[dim] = {}
            # 修复：正确访问统计信息
            current_dist = self.stats.get('dimensions', {}).get(dim, {})
            for cat, target_ratio in targets.items():
                current_ratio = current_dist.get(cat, 0)
                if current_ratio > 0:
                    # 限制调整因子范围，避免过度调整
                    factor = max(0.1, min(10.0, target_ratio / (current_ratio + 1e-8)))
                    weights[dim][cat] = factor
                else:
                    weights[dim][cat] = 0
        return weights

    def calculate_row_probability(self, row, weights):
        """计算单行的采样概率"""
        prob = 1.0
        dimensions = ['source', 'category', 'domain', 'language', 'token_bin']
        for dim in dimensions:
            if dim in weights and row[dim] in weights[dim]:
                prob *= weights[dim][row[dim]]
        # 限制概率在合理范围内
        return max(0.0, min(1.0, prob))

    def write_shard(self, data, shard_path):
        """写入分片文件"""
        try:
            with open(shard_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n') # 修复：添加换行符
        except Exception as e:
            st.error(f"写入分片 {shard_path} 失败: {str(e)}")

# 工具函数
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
    dist = df.groupby(column).apply(lambda x: np.sum(weights[x.index]) / total)
    return dist.sort_values(ascending=False)

@st.cache_data
def calculate_distribution_cached(df, column, weights=None):
    """缓存版本的分布计算"""
    return calculate_distribution(df, column, weights)

def advanced_ipf_solver(df, target_ratios, target_total, max_iter=100, tol=0.005):
    """
    改进的IPF求解器 - 支持多维度同时优化
    修复：保证所有维度都收敛再停止迭代
    """
    # 初始化权重
    weights = np.ones(len(df))
    total_tokens = df['token_count'].sum()
    # 检查目标比例可行性
    for dim, targets in target_ratios.items():
        for cat, ratio in targets.items():
            # 检查该类别在原始数据中是否存在
            if cat not in df[dim].values:
                st.error(f"错误：维度 '{dim}' 中不存在类别 '{cat}'")
                return None, None, False, {}
            # 检查目标比例是否超过原始数据最大可能
            orig_ratio = (df[df[dim] == cat]['token_count'].sum() / total_tokens)
            if ratio > orig_ratio * 1.05:  # 允许5%缓冲
                st.warning(f"警告：'{dim}'中'{cat}'目标比例({ratio:.2%})超过原始比例({orig_ratio:.2%})，可能无法精确满足")
        # 检查维度内比例和
        dim_sum = sum(targets.values())
        if not (0.99 <= dim_sum <= 1.01):
            st.error(f"错误：维度 '{dim}' 的目标比例和({dim_sum:.2%})不在[99%, 101%]范围内")
            return None, None, False, {}
    # 开始IPF迭代 - 修复：保证所有维度都收敛再停止
    all_dims = set(target_ratios.keys())
    converged_dims = set()  # 记录已收敛的维度
    for iter in range(max_iter):
        prev_weights = weights.copy()
        max_errors = {}
        iteration_converged_dims = set()  # 本轮迭代中收敛的维度
        # 按维度迭代调整
        for dim, targets in target_ratios.items():
            dim_max_error = 0
            for cat, target_ratio in targets.items():
                # 计算当前维度类别的加权比例
                mask = (df[dim] == cat)
                current_ratio = np.sum(weights[mask] * df.loc[mask, 'token_count']) / np.sum(weights * df['token_count'])
                # 计算调整因子（避免除零）
                if current_ratio > 1e-5 and target_ratio > 0:
                    factor = target_ratio / (current_ratio + 1e-8)
                    # 限制调整幅度，避免过度调整
                    factor = max(0.5, min(2.0, factor))
                    weights[mask] *= factor
                # 记录最大误差
                error = abs(current_ratio - target_ratio)
                dim_max_error = max(dim_max_error, error)
            max_errors[dim] = dim_max_error
            # 检查该维度是否收敛
            if dim_max_error < tol:
                iteration_converged_dims.add(dim)
        # 更新收敛维度集合
        converged_dims.update(iteration_converged_dims)
        # 检查所有维度是否都收敛
        if len(converged_dims) == len(all_dims):
            st.info(f"✅ 所有维度在第 {iter+1} 轮迭代后收敛")
            break
        # 检查权重变化
        weight_change = np.mean(np.abs(weights - prev_weights) / (prev_weights + 1e-5))
        if weight_change < 1e-5:
            st.info(f"⚠️ 权重变化过小，在第 {iter+1} 轮迭代后停止")
            break
    # 缩放至目标总量
    current_total = np.sum(weights * df['token_count'])
    if current_total > 0:
        weights *= (target_total / current_total)
    # 计算实际分布（用于验证）
    actual_dist = {}
    final_errors = {}
    for dim in target_ratios.keys():
        actual_dist[dim] = {}
        dim_max_error = 0
        for cat in target_ratios[dim].keys():
            mask = (df[dim] == cat)
            actual_ratio = np.sum(weights[mask] * df.loc[mask, 'token_count']) / target_total
            actual_dist[dim][cat] = actual_ratio
            target_ratio = target_ratios[dim][cat]
            error = abs(actual_ratio - target_ratio)
            dim_max_error = max(dim_max_error, error)
        final_errors[dim] = dim_max_error
    # 显示各维度误差
    st.subheader("📊 各维度配比误差")
    for dim, error in final_errors.items():
        if error <= tol:
            st.success(f"✅ {dim}: 最大误差 {error:.3f} ({error*100:.1f}%)")
        else:
            st.warning(f"⚠️ {dim}: 最大误差 {error:.3f} ({error*100:.1f}%)")
    is_converged = all(error <= tol for error in final_errors.values())
    return weights, actual_dist, is_converged, final_errors

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
            remaining_token_sum = remaining['token_count'].sum()
            if remaining_token_sum > 0:
                remaining_prob = (additional * remaining['token_count'] / remaining_token_sum)
                # 确保概率不超过1
                remaining['prob'] = np.minimum(remaining_prob, 1.0)
                retained[~retained] = np.random.random(len(remaining)) < remaining['prob']
    return df[retained].copy()

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
                }, ensure_ascii=False) + '\n') # 修复：添加换行符
        return True, shard_path
    except Exception as e:
        return False, f"Error writing {shard_path}: {str(e)}"

def export_shards_parallel(df, output_path, shard_size_gb=1, max_workers=4):
    """并行分片导出JSONL文件"""
    os.makedirs(output_path, exist_ok=True)
    shard_size_bytes = shard_size_gb * GB
    # 计算需要多少个分片
    total_bytes = df['text'].apply(lambda x: len(x.encode('utf-8')) + 1).sum()
    num_shards = math.ceil(total_bytes / shard_size_bytes)
    st.info(f"需要创建 {num_shards} 个分片文件")
    # 将数据分组到分片中
    shards_data = []
    current_shard = []
    current_size = 0
    shard_idx = 1
    progress_bar = st.progress(0)
    status_text = st.empty()
    for idx, row in df.iterrows():
        # 计算当前样本字节数
        sample_bytes = len(row['text'].encode('utf-8')) + 1  # +1 for newline
        # 如果当前分片已满，保存当前分片
        if current_size + sample_bytes > shard_size_bytes and current_shard:
            shard_path = os.path.join(output_path, f"shard_{shard_idx:04d}.jsonl")
            shards_data.append((current_shard.copy(), shard_path))
            current_shard = []
            current_size = 0
            shard_idx += 1
        # 添加样本到当前分片
        current_shard.append(row.to_dict())
        current_size += sample_bytes
        # 更新进度
        if idx % 1000 == 0:
            progress = min(idx / len(df), 1.0)
            progress_bar.progress(progress)
            status_text.text(f"分片准备中: {idx+1}/{len(df)} | 当前分片: {shard_idx}")
    # 保存最后一个分片
    if current_shard:
        shard_path = os.path.join(output_path, f"shard_{shard_idx:04d}.jsonl")
        shards_data.append((current_shard, shard_path))
    progress_bar.empty()
    status_text.empty()
    # 并行写入分片文件
    st.info(f"开始并行写入 {len(shards_data)} 个分片文件...")
    success_count = 0
    failed_files = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有写入任务
        future_to_shard = {executor.submit(write_shard_batch, rows, path): (i, path) 
                          for i, (rows, path) in enumerate(shards_data)}
        # 处理完成的任务
        for i, future in enumerate(as_completed(future_to_shard)):
            success, result = future.result()
            if success:
                success_count += 1
            else:
                failed_files.append(result)
            # 更新进度
            progress = (i + 1) / len(shards_data)
            progress_bar.progress(progress)
            status_text.text(f"写入进度: {i+1}/{len(shards_data)} | 成功: {success_count}")
    progress_bar.empty()
    status_text.empty()
    # 报告结果
    if failed_files:
        st.warning(f"导出完成！成功: {success_count}, 失败: {len(failed_files)}")
        for error in failed_files[:5]:  # 只显示前5个错误
            st.error(error)
        if len(failed_files) > 5:
            st.error(f"... 还有 {len(failed_files) - 5} 个错误")
    else:
        st.success(f"导出完成！共 {success_count} 个分片，路径: {output_path}")

def parse_jsonl_file_pandas(file_path, chunksize=50000):
    """使用pandas高效解析JSONL文件（支持分块读取）"""
    records = []
    try:
        # 分块读取以处理大文件
        chunk_iter = pd.read_json(file_path, lines=True, chunksize=chunksize)
        # 如果不是迭代器，说明文件较小，直接读取
        if not hasattr(chunk_iter, '__iter__'):
            chunk_iter = [chunk_iter]
        for chunk in chunk_iter:
            # 必需字段
            required_fields = ['source', 'category', 'domain', 'language', 'token_count', 'text']
            # 检查必需字段是否存在
            if all(col in chunk.columns for col in required_fields):
                # 只保留必需字段
                chunk = chunk[required_fields]
                # 数据类型转换和清洗
                chunk['token_count'] = pd.to_numeric(chunk['token_count'], errors='coerce')
                chunk.dropna(subset=['token_count'], inplace=True)
                chunk['token_count'] = chunk['token_count'].astype(int)
                # 确保其他字段为字符串类型
                string_fields = ['source', 'category', 'domain', 'language', 'text']
                for field in string_fields:
                    chunk[field] = chunk[field].astype(str)
                # 添加token_bin列
                chunk['token_bin'] = chunk['token_count'].apply(get_token_bin)
                # 转换为记录列表
                records.extend(chunk.to_dict(orient='records'))
            else:
                print(f"Missing required fields in {file_path}")
                continue
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return records

# ========== 左侧配置栏 ==========
st.sidebar.header("🔧 配置面板")
data_path = st.sidebar.text_input("数据集文件夹路径", value="./test_data")

# 数据处理模式选择
# 确保 session_state 中有 processing_mode
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = "内存模式（小数据）"

# 根据 session_state 中的值确定 radio 的 index
current_index = 0 if st.session_state.processing_mode == "内存模式（小数据）" else 1

# 创建 radio 按钮
processing_mode_selection = st.sidebar.radio(
    "处理模式",
    ["内存模式（小数据）", "流式模式（大数据）"],
    index=current_index, # 使用计算出的 index
    help="内存模式适用于<100GB数据，流式模式适用于>100GB数据"
)

# 添加路径诊断工具
if st.sidebar.checkbox("🔍 启用路径诊断", value=False):
    st.sidebar.subheader("路径诊断")
    abs_path = os.path.abspath(data_path) if data_path else ""
    st.sidebar.code(f"绝对路径: {abs_path}")
    if data_path and os.path.exists(data_path):
        st.sidebar.success("✅ 路径存在")
        try:
            items = os.listdir(data_path)
            st.sidebar.info(f"包含 {len(items)} 个项目")
            jsonl_files = [f for f in items if f.lower().endswith('.jsonl')]
            st.sidebar.info(f"其中 {len(jsonl_files)} 个JSONL文件")
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
        st.sidebar.info(f"正在处理路径: {data_path}")
        with st.spinner("🔍 正在扫描数据集文件..."):
            try:
                if processing_mode == "内存模式（小数据）":
                    # 原有的内存模式处理
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
                    # 并行读取所有JSONL文件（使用pandas优化版本）
                    all_data = []
                    progress_bar = st.sidebar.progress(0)
                    status_text = st.sidebar.empty()
                    # 使用线程池并行处理文件
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        future_to_file = {executor.submit(parse_jsonl_file_pandas, file, 50000): file for file in jsonl_files}
                        for i, future in enumerate(as_completed(future_to_file)):
                            result = future.result()
                            all_data.extend(result)
                            status_text.text(f"✅ 已处理 {i+1}/{len(jsonl_files)} 个文件")
                            progress_bar.progress((i + 1) / len(jsonl_files))
                    progress_bar.empty()
                    status_text.empty()
                    if all_data:
                        # 转为DataFrame
                        df = pd.DataFrame(all_data)
                        total_tokens = df['token_count'].sum()
                        # 存储到session state
                        st.session_state.df = df
                        st.session_state.total_tokens = total_tokens
                        st.session_state.processing_mode = "内存"
                        st.sidebar.success(f"🎉 加载成功！共 {len(df):,} 个有效样本，{total_tokens/1e9:.2f}B tokens")
                    else:
                        st.sidebar.error("❌ 未找到有效数据，请检查文件格式")
                        st.sidebar.info("有效JSONL样本示例:")
                        st.sidebar.code('''{"source": "CCI4", "category": "book", "domain": "science", "language": "CN", "token_count": 1234, "text": "示例文本..."}''')
                        st.stop()
                else:  # 流式模式
                    # 大数据流式处理
                    sampler = LargeDataSampler(data_path)
                    file_count = sampler.scan_files()
                    st.sidebar.info(f"📁 找到 {file_count} 个JSONL文件")
                    if file_count == 0:
                        st.sidebar.warning("⚠️ 未找到JSONL文件，请检查路径")
                        st.stop()
                    # 计算统计信息
                    stats = sampler.calculate_statistics()
                    # 存储到session state
                    st.session_state.sampler = sampler
                    st.session_state.stats = stats
                    st.session_state.processing_mode = "流式"
                    st.sidebar.success(f"🎉 统计完成！共 {stats['total_samples']:,} 个样本，{stats['total_tokens']/1e9:.2f}B tokens")
            except Exception as e:
                st.sidebar.exception(f"_fatal error_: {str(e)}")
                st.stop()

# 检查数据是否已加载
if 'processing_mode' in st.session_state and st.session_state.processing_mode in ["内存", "流式"]:
    processing_mode_session = st.session_state.processing_mode
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
    # 初始化 session_state 存储目标比例
    if 'target_ratios' not in st.session_state:
        st.session_state.target_ratios = {}
    # 获取 token_bin 顺序
    token_bin_order = [label for _, _, label in TOKEN_BINS]
    if processing_mode_session == "内存":
        df = st.session_state.df
        total_tokens = st.session_state.total_tokens
        # 确保token_bin列存在
        if 'token_bin' not in df.columns:
            df['token_bin'] = df['token_count'].apply(get_token_bin)
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
    else:  # streaming mode
        stats = st.session_state.stats
        for dim in dimensions:
            st.sidebar.subheader(f"{dim.capitalize()} 配比")
            # 获取该维度的唯一值
            if dim == 'token_bin':
                values = sorted(stats['dimensions'][dim].keys(), key=lambda x: token_bin_order.index(x) if x in token_bin_order else len(token_bin_order))
            else:
                values = sorted(stats['dimensions'][dim].keys())
            # 获取当前分布
            current_dist = stats['dimensions'][dim]
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
                            format="%.3f",
                            key=f"{dim}_{val}_stream"
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
        if processing_mode_session == "内存":
            with st.spinner("正在计算配比方案..."):
                # 从 session_state 读取最新的目标比例
                target_ratios = st.session_state.target_ratios
                # 运行改进的IPF求解器
                weights, actual_dist, converged, final_errors = advanced_ipf_solver(
                    df, 
                    target_ratios, 
                    target_total,
                    max_iter=100,
                    tol=0.005
                )
                if weights is not None:
                    # 存储采样结果和误差信息
                    sampled_df = sample_dataset(df, weights, target_total)
                    st.session_state.sampled_df = sampled_df
                    st.session_state.final_errors = final_errors  # 存储误差信息
                    # 显示采样结果
                    st.sidebar.success("配比方案已生成！")
                    st.sidebar.info(f"实际总量: {sampled_df['token_count'].sum()/1e9:.2f}B tokens")
                    # 显示收敛状态
                    if converged:
                        st.sidebar.success("✅ 所有维度配比均已满足！")
                    else:
                        st.sidebar.warning("⚠️ 部分维度配比未完全满足，请检查误差报告")
        else:  # streaming mode
            with st.spinner("正在流式采样大数据集..."):
                sampler = st.session_state.sampler
                target_ratios = st.session_state.target_ratios
                # 获取输出路径
                output_path = st.sidebar.text_input("输出路径", value="./sampled_datasets")
                shard_size = st.sidebar.number_input("分片大小 (GB)", min_value=0.1, value=1.0, step=0.1)
                # 执行流式采样
                total_written = sampler.sample_data_streaming(
                    target_ratios, 
                    target_total, 
                    output_path, 
                    shard_size
                )
                st.sidebar.success(f"流式采样完成！共写入 {total_written/1e9:.2f}B tokens")
    # ========== 导出配置 ==========
    st.sidebar.header("📤 导出设置")
    output_path = st.sidebar.text_input("导出路径", value="./balanced_datasets")
    shard_size = st.sidebar.number_input("分片大小 (GB)", min_value=0.1, value=1.0, step=0.1)
    max_export_workers = st.sidebar.slider("导出并行线程数", min_value=1, max_value=16, value=4)
    if st.sidebar.button("💾 导出配比数据集", type="primary"):
        if 'sampled_df' not in st.session_state:
            st.sidebar.error("请先应用配比方案")
        else:
            with st.spinner("正在导出分片..."):
                export_shards_parallel(st.session_state.sampled_df, output_path, shard_size, max_export_workers)
    # ========== 右侧图表展示 ==========
    st.header("📊 数据分布分析")
    if processing_mode_session == "内存":
        # 内存模式的图表展示
        df = st.session_state.df
        total_tokens = st.session_state.total_tokens
        # 确保token_bin列存在
        if 'token_bin' not in df.columns:
            df['token_bin'] = df['token_count'].apply(get_token_bin)
        # 创建图表布局
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        col5, col6 = st.columns(2)
        # 1. Source 配比图
        with col1:
            st.subheader("数据来源 (Source) 分布")
            source_dist = calculate_distribution_cached(df, 'source')
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(source_dist, labels=source_dist.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        # 2. Category 配比图
        with col2:
            st.subheader("数据类别 (Category) 分布")
            category_dist = calculate_distribution_cached(df, 'category')
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(category_dist, labels=category_dist.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        # 3. Domain 配比图
        with col3:
            st.subheader("数据领域 (Domain) 分布")
            domain_dist = calculate_distribution_cached(df, 'domain')
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(domain_dist, labels=domain_dist.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        # 4. Language 配比图
        with col4:
            st.subheader("语言 (Language) 分布")
            lang_dist = calculate_distribution_cached(df, 'language')
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(lang_dist, labels=lang_dist.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        # 5. Token Count 配比图
        with col5:
            st.subheader("Token长度分布")
            token_dist = calculate_distribution_cached(df, 'token_bin')
            # 确保所有分组都存在并按正确顺序排列
            ordered_labels = [label for _, _, label in TOKEN_BINS]
            for label in ordered_labels:
                if label not in token_dist:
                    token_dist[label] = 0.0
            token_dist = token_dist.reindex(ordered_labels)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(token_dist.index, token_dist.values)
            ax.set_ylabel('Ratio')
            ax.set_title('Token length distribution')
            for i, v in enumerate(token_dist.values):
                ax.text(i, v + 0.01, f'{v:.1%}', ha='center')
            st.pyplot(fig)
        # 6. 子类分布图
        with col6:
            st.subheader("子类组合分布 (Top 10)")
            # 创建子类组合
            df['subclass'] = df['source'] + "+" + df['category'] + "+" + df['domain'] + "+" + df['language']
            subclass_dist = calculate_distribution_cached(df, 'subclass')
            # 取Top 10
            top10 = subclass_dist.head(10)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(top10.index, top10.values)
            ax.set_xlabel('比例')
            ax.set_title('Top 10 distribution of subclass combinations')
            # 添加比例标签
            for i, v in enumerate(top10.values):
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
            # 确保采样数据也有token_bin列
            if 'token_bin' not in sampled_df.columns:
                sampled_df['token_bin'] = sampled_df['token_count'].apply(get_token_bin)
            st.write(f"**采样总量**: {sampled_tokens/1e9:.2f} B tokens")
            st.write(f"**采样比例**: {len(sampled_df)/len(df):.1%}")
            # 比较关键维度
            st.subheader("📈 配比对比分析")
            comparison_cols = st.columns(len(['language', 'domain', 'category', 'token_bin']))
            # 获取存储的误差信息
            final_errors = st.session_state.get('final_errors', {})
            for i, dim in enumerate(['language', 'domain', 'category', 'token_bin']):
                with comparison_cols[i]:
                    if dim in final_errors:
                        # 使用IPF求解器计算出的准确误差
                        error = final_errors[dim]
                        st.metric(f"{dim.capitalize()}", f"{error*100:.1f}%", "最大误差")
                    else:
                        # 备用计算方法
                        orig_dist = calculate_distribution_cached(df, dim)
                        sampled_dist = calculate_distribution_cached(sampled_df, dim)
                        # 计算最大误差
                        max_error = 0
                        for cat in orig_dist.index:
                            orig = orig_dist.get(cat, 0)
                            sampled = sampled_dist.get(cat, 0)
                            error = abs(orig - sampled)
                            max_error = max(max_error, error)
                        st.metric(f"{dim.capitalize()}", f"{max_error*100:.1f}%", "最大误差")
    else:
        # 流式模式的统计信息展示
        stats = st.session_state.stats
        st.subheader("📊 数据统计信息")
        st.write(f"**总样本数**: {stats['total_samples']:,}")
        st.write(f"**总Token数**: {stats['total_tokens']/1e9:.2f} B (10亿)")
        st.write(f"**平均Token长度**: {stats['total_tokens']/stats['total_samples']:.0f}")
        # 显示各维度分布
        st.subheader("各维度分布")
        for dim in ['source', 'category', 'domain', 'language', 'token_bin']:
            if dim in stats['dimensions']:
                st.write(f"**{dim.capitalize()} 分布**:")
                dist_data = [(k, v) for k, v in stats['dimensions'][dim].items()]
                dist_data.sort(key=lambda x: x[1], reverse=True)
                # 显示前10个
                for key, ratio in dist_data[:10]:
                    st.write(f"  {key}: {ratio:.2%}")
                if len(dist_data) > 10:
                    st.write(f"  ... 还有 {len(dist_data) - 10} 个类别")
else:
    st.info("👈 请在左侧输入数据集路径并点击'加载数据集'")
    st.image("https://docs.streamlit.io/images/brand/streamlit-mark-color.png", width=300)
