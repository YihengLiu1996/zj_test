# utils/chunk_utils.py

import os
import argparse
from typing import Optional
from transformers import AutoTokenizer
from tqdm import tqdm
import fasttext
import json
import uuid
import re

# ========================
# 从您的“数据预处理代码.txt”中提取的核心函数
# ========================

# 语言检测模型 (假设模型文件在项目根目录)
FASTTEXT_MODEL_PATH = "./lid.176.bin"
TOKENIZER_MODEL_PATH = "./qwen_model_fold" # 假设tokenizer模型在此路径

def detect_language(text):
    """检测文本语言"""
    # 检查模型文件是否存在
    if not os.path.exists(FASTTEXT_MODEL_PATH):
        print(f"警告: 未找到语言检测模型 {FASTTEXT_MODEL_PATH}, 默认返回 'CN'")
        return "CN"
    
    text_lg = text.replace('\n', '')
    if len(text_lg) > 0:
        try:
            fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
            predictions = fasttext_model.predict(text_lg, k=1)
            language = predictions[0][0].replace('__label__', '')
            if language == "zh":
                language = "CN"
            else:
                language = "EN"
        except Exception as e:
            print(f"语言检测失败: {e}, 默认返回 'CN'")
            language = "CN"
    else:
        language = "CN"
    return language

def split_markdown_by_headings_tokenized(md_content, tokenizer, threshold_tokens=15000):
    """按标题 + token 长度切分 Markdown"""
    # ============= 第0阶段：整体判断 =============
    all_tokens = tokenizer.encode(md_content, add_special_tokens=False)
    if len(all_tokens) <= threshold_tokens:
        return [md_content]  # 整体不超限，直接返回

    # ============= 第一阶段：按标题切分 =============
    lines = md_content.splitlines(keepends=True)
    heading_chunks = []
    current_chunk = []
    for line in lines:
        if line.lstrip().startswith('#'):
            if current_chunk:
                heading_chunks.append(''.join(current_chunk))
                current_chunk = []
            current_chunk.append(line)
        else:
            current_chunk.append(line)
    if current_chunk:
        heading_chunks.append(''.join(current_chunk))

    # ============= 第二阶段：对超限标题块，按段落切分 =============
    paragraph_level_chunks = []
    for h_chunk in heading_chunks:
        h_tokens = tokenizer.encode(h_chunk, add_special_tokens=False)
        if len(h_tokens) <= threshold_tokens:
            # 标题块不超限，保留
            paragraph_level_chunks.append(h_chunk)
        else:
            # 超限 → 按段落切
            paragraphs = re.split(r'(\n\s*\n)', h_chunk)  # 保留分隔符
            current_para = ""
            para_list = []
            for part in paragraphs:
                if re.fullmatch(r'\n\s*\n', part):
                    if current_para.strip():
                        para_list.append(current_para)
                        current_para = ""
                    para_list.append(part)  # 保留空行
                else:
                    current_para += part
            if current_para.strip():
                para_list.append(current_para)

            # 检查每个段落块是否超限
            for p_chunk in para_list:
                p_tokens = tokenizer.encode(p_chunk, add_special_tokens=False)
                if len(p_tokens) <= threshold_tokens:
                    paragraph_level_chunks.append(p_chunk)
                else:
                    # 仍超限 → 进入第三级：按 token + 公式保护 切分
                    fine_segments = _split_by_token_with_formula_protection(p_chunk, tokenizer, threshold_tokens)
                    paragraph_level_chunks.extend(fine_segments)

    # ============= 第三阶段：贪心合并所有块 =============
    final_chunks = []
    current_merged = []
    current_token_count = 0
    for seg in paragraph_level_chunks:
        seg_tokens = tokenizer.encode(seg, add_special_tokens=False)
        seg_token_len = len(seg_tokens)
        # 如果当前段单独超限（如超长公式）→ 单独成块
        if seg_token_len > threshold_tokens:
            if current_merged:
                final_chunks.append(''.join(current_merged))
                current_merged = []
                current_token_count = 0
            final_chunks.append(seg)
            continue
        # 尝试合并到当前块
        if current_token_count + seg_token_len <= threshold_tokens:
            current_merged.append(seg)
            current_token_count += seg_token_len
        else:
            # 当前块已满，提交
            if current_merged:
                final_chunks.append(''.join(current_merged))
            # 开新块
            current_merged = [seg]
            current_token_count = seg_token_len
    # 提交最后一块
    if current_merged:
        final_chunks.append(''.join(current_merged))
    return final_chunks

def _split_by_token_with_formula_protection(text, tokenizer, max_tokens):
    """在 text 内部按 token 长度切分，保护公式不被切断。"""
    segments = []
    pos = 0
    text_len = len(text)
    # Step 1: 按公式切分成 segments（公式作为原子单元）
    while pos < text_len:
        inline_match = re.search(r'\$[^$].*?\$', text[pos:], re.DOTALL)
        block_match = re.search(r'\$\$.*?\$\$', text[pos:], re.DOTALL)
        matches = []
        if inline_match:
            matches.append((inline_match.start() + pos, inline_match.end() + pos, inline_match.group()))
        if block_match:
            matches.append((block_match.start() + pos, block_match.end() + pos, block_match.group()))
        if matches:
            start, end, formula = min(matches, key=lambda x: x[0])
            if start > pos:
                segments.append(text[pos:start])
            segments.append(formula)
            pos = end
        else:
            segments.append(text[pos:])
            break

    # Step 2: 按 token 合并 segments，不超过 max_tokens
    chunks = []
    current_chunk = ""
    current_token_count = 0
    for seg in segments:
        seg_tokens = tokenizer.encode(seg, add_special_tokens=False)
        seg_len = len(seg_tokens)
        # 如果是公式且单独超限 → 单独成块
        if (seg.startswith('$') or seg.startswith('$$')) and seg_len > max_tokens:
            if current_chunk.strip():
                chunks.append(current_chunk)
                current_chunk = ""
                current_token_count = 0
            chunks.append(seg)
            continue
        # 尝试加入当前块
        if current_token_count + seg_len <= max_tokens:
            current_chunk += seg
            current_token_count += seg_len
        else:
            # 当前块满了
            if current_chunk.strip():
                chunks.append(current_chunk)
            current_chunk = seg
            current_token_count = seg_len
    if current_chunk.strip():
        chunks.append(current_chunk)
    return chunks

# ========================
# 封装的主处理函数 (供 Dataset 类调用)
# ========================

def process_markdown_folder(
    input_dir: str,
    output_file_path: str,
    max_tokens: int = 15000,
    min_tokens: int = 2000
) -> None:
    """
    处理指定文件夹内的所有Markdown文件，并将结果输出到JSONL文件。
    这是供 `models/dataset.py` 调用的核心接口。
    
    Args:
        input_dir: 包含 .md 文件的输入目录。
        output_file_path: 输出的 .jsonl 文件路径。
        max_tokens: 每个chunk的最大token数。
        min_tokens: 每个chunk的最小token数（小于该值的chunk会被过滤掉）。
    """
    # 加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH, local_files_only=True)
    except Exception as e:
        raise RuntimeError(f"无法加载Tokenizer: {e}. 请确保模型文件在 {TOKENIZER_MODEL_PATH}")

    md_files = [f for f in os.listdir(input_dir) if f.endswith('.md')]
    total_files = len(md_files)
    
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for filename in tqdm(md_files, desc="Processing markdown files"):
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # 检测语言
            language = detect_language(content)

            # 按标题+token切分
            chunks = split_markdown_by_headings_tokenized(content, tokenizer, max_tokens)

            # 过滤并构造输出记录
            for chunk in chunks:
                tokens = tokenizer.encode(chunk, add_special_tokens=False)
                token_count = len(tokens)
                if token_count < min_tokens:
                    continue

                record = {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "text_length": token_count,  # 实际是 token 数
                    "language": language,
                    "file_path": file_path  # 新增字段
                }
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Processed {total_files} markdown files. Output saved to {output_file_path}")
