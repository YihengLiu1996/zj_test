import os
import json
import uuid
import re
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import fasttext


# ========================
# 配置与模型加载
# ========================
fasttext_model_dir = "./lid.176.bin"
fasttext_model = fasttext.load_model(fasttext_model_dir)


# ========================
# 语言检测
# ========================
def detect_language(text):
    text_lg = text.replace('\n', '')
    if len(text_lg) > 0:
        predictions = fasttext_model.predict(text_lg, k=1)
        language = predictions[0][0].replace('__label__', '')
        if language == "zh":
            language = "CN"
        else:
            language = "EN"
    else:
        language = "CN"
    return language


# ========================
# 按标题 + token 长度切分 Markdown（核心修改：用 tokenizer 计算长度）
# ========================
def split_markdown_by_headings_tokenized(md_content, tokenizer, threshold_tokens=15000):
    """
    Split markdown by headings, then by token count (not char count).
    Preserves LaTeX formulas as atomic units.
    Then greedily merge segments to form chunks as close as possible to threshold_tokens (without exceeding).
    """
    lines = md_content.splitlines(keepends=True)
    chunks = []
    current_chunk = []

    # 第一阶段：按标题切分
    for line in lines:
        if line.lstrip().startswith('#'):
            if current_chunk:
                chunks.append(''.join(current_chunk))
                current_chunk = []
            current_chunk.append(line)
        else:
            current_chunk.append(line)
    if current_chunk:
        chunks.append(''.join(current_chunk))

    # 第二阶段：对每个 chunk 拆分成 segments（公式保护）
    all_segments = []
    for chunk in chunks:
        segments = []
        pos = 0
        chunk_len = len(chunk)
        while pos < chunk_len:
            # 查找下一个 LaTeX 公式（$...$ 或 $$...$$）
            inline_match = re.search(r'\$[^$].*?\$', chunk[pos:], re.DOTALL)
            block_match = re.search(r'\$\$.*?\$\$', chunk[pos:], re.DOTALL)

            matches = []
            if inline_match:
                matches.append((inline_match.start() + pos, inline_match.end() + pos, inline_match.group()))
            if block_match:
                matches.append((block_match.start() + pos, block_match.end() + pos, block_match.group()))

            if matches:
                # 取最早出现的公式
                start, end, formula = min(matches, key=lambda x: x[0])
                # 添加公式前的文本
                if start > pos:
                    segments.append(chunk[pos:start])
                # 添加公式（作为原子单元）
                segments.append(formula)
                pos = end
            else:
                # 无公式，添加剩余文本
                segments.append(chunk[pos:])
                break
        all_segments.extend(segments)

    # 第三阶段：贪心合并 segments，使每个块尽可能接近 threshold_tokens
    final_chunks = []
    current_chunk_segments = []
    current_token_count = 0

    for seg in all_segments:
        seg_tokens = tokenizer.encode(seg, add_special_tokens=False)
        seg_token_len = len(seg_tokens)

        # 如果当前段单独就超过阈值 → 单独成块（即使超限，也保留完整性，特别是公式）
        if seg_token_len > threshold_tokens:
            # 先提交当前块（如果有）
            if current_chunk_segments:
                final_chunks.append(''.join(current_chunk_segments))
                current_chunk_segments = []
                current_token_count = 0
            # 再提交这个超大段（即使超限）
            final_chunks.append(seg)
            continue

        # 尝试加入当前块
        if current_token_count + seg_token_len <= threshold_tokens:
            current_chunk_segments.append(seg)
            current_token_count += seg_token_len
        else:
            # 当前块满了，提交
            if current_chunk_segments:
                final_chunks.append(''.join(current_chunk_segments))
            # 开启新块，放入当前段
            current_chunk_segments = [seg]
            current_token_count = seg_token_len

    # 提交最后一个块
    if current_chunk_segments:
        final_chunks.append(''.join(current_chunk_segments))

    return final_chunks


# ========================
# 主处理函数
# ========================
def process_markdown_folder(input_dir, output_file_path, tokenizer, max_tokens=15000, min_tokens=2000):
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

            # 过滤小于 min_tokens 的块
            for chunk in chunks:
                tokens = tokenizer.encode(chunk, add_special_tokens=False)
                token_count = len(tokens)
                if token_count < min_tokens:
                    continue

                # 构造输出记录
                record = {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "text_length": token_count,  # 实际是 token 数
                    "language": language,
                    "file_path": file_path  # 新增字段
                }
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Processed {total_files} markdown files. Output saved to {output_file_path}")


# ========================
# 主程序入口
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process markdown files into tokenized chunks.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .md files")
    parser.add_argument("--output_file", type=str, required=True, help="Output .jsonl file path")
    parser.add_argument("--max_tokens", type=int, default=15000, help="Max tokens per chunk")
    parser.add_argument("--min_tokens", type=int, default=2000, help="Min tokens per chunk (filter out smaller)")

    args = parser.parse_args()
    model_dir = "./qwen_model_fold"
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # 处理文件夹
    process_markdown_folder(
        input_dir=args.input_dir,
        output_file_path=args.output_file,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens
    )
