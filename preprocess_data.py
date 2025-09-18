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

    # 第二阶段：按 token 数进一步切分超长块
    final_chunks = []
    for chunk in chunks:
        # 如果当前块 token 数小于阈值，直接保留
        tokens = tokenizer.encode(chunk, add_special_tokens=False)
        if len(tokens) <= threshold_tokens:
            final_chunks.append(chunk)
            continue

        # 否则，按公式保护 + token 切分
        segments = []
        pos = 0
        while pos < len(chunk):
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

        # 合并 segments 成符合 token 阈值的 chunks
        current_sub = []
        current_token_count = 0

        for seg in segments:
            seg_tokens = tokenizer.encode(seg, add_special_tokens=False)
            seg_token_len = len(seg_tokens)

            # 情况1：当前段可加入当前块
            if current_token_count + seg_token_len <= threshold_tokens:
                current_sub.append(seg)
                current_token_count += seg_token_len

            # 情况2：当前段是公式且单独超限 → 单独成块
            elif seg.startswith('$'):
                if current_sub:
                    final_chunks.append(''.join(current_sub))
                    current_sub = []
                    current_token_count = 0
                final_chunks.append(seg)  # 公式单独成块，即使超限

            # 情况3：当前段是普通文本且无法加入 → 切分当前段
            else:
                if current_sub:
                    final_chunks.append(''.join(current_sub))
                    current_sub = []
                    current_token_count = 0

                # 对大段普通文本按 token 切分
                start_idx = 0
                while start_idx < len(seg_tokens):
                    end_idx = min(start_idx + threshold_tokens, len(seg_tokens))
                    # 尝试在自然边界切分（换行或空格）
                    sub_tokens = seg_tokens[start_idx:end_idx]
                    sub_text = tokenizer.decode(sub_tokens, skip_special_tokens=True)

                    # 如果还能继续切分，尝试找最后一个换行或空格
                    if end_idx < len(seg_tokens):
                        # 优先找换行
                        last_newline = sub_text.rfind('\n')
                        if last_newline != -1:
                            # 截断到换行处
                            sub_text_cut = sub_text[:last_newline + 1]
                            cut_tokens = tokenizer.encode(sub_text_cut, add_special_tokens=False)
                            end_idx = start_idx + len(cut_tokens)
                        else:
                            # 找空格
                            last_space = sub_text.rfind(' ')
                            if last_space != -1:
                                sub_text_cut = sub_text[:last_space + 1]
                                cut_tokens = tokenizer.encode(sub_text_cut, add_special_tokens=False)
                                end_idx = start_idx + len(cut_tokens)

                    final_chunks.append(tokenizer.decode(seg_tokens[start_idx:end_idx], skip_special_tokens=True))
                    start_idx = end_idx

        # 添加最后一块
        if current_sub:
            final_chunks.append(''.join(current_sub))

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
