import os
import fasttext
import json
import multiprocessing as mp
import pickle
import sys
import re
import uuid
import time
from tqdm import tqdm
import psutil
import argparse
import signal
from pylatexenc.latexwalker import LatexWalker, LatexMathNode
from latex_filter import LatexFormulaQualityFilter

# 修复导入路径
sys.path.append(os.path.join(os.path.dirname(__file__), './english_data_clean_pipeline/filter'))
sys.path.append(os.path.join(os.path.dirname(__file__), './chinese_data_clean_pipeline/filter'))

from english_data_clean_pipeline.filter.util_with_try import filter_one_line
from chinese_data_clean_pipeline.filter.filter import filter_simple

# FastText 模型路径
fasttext_model_dir = "/mnt/sizjwb25c1g7/nanhu_lyh/code_tx/data_process_code/MAP-NEO-main/Matrix/lid.176.bin"

# 全局初始化锁，避免多个进程同时加载模型
_init_lock = mp.Lock()

# ================= 工具函数 =================

def remove_whitespace(input_str):
    """移除字符串中所有空白字符（空格、制表符、换行符等）"""
    return ''.join(input_str.split())

def filter_markdown(text):
    if not text:
        return "", {"before_first_heading": "", "references_section": ""}
    
    lines = text.splitlines()
    
    removed_content = {
        "before_first_heading": "",
        "references_section": ""
    }
    
    start_index = None
    for i, line in enumerate(lines):
        if re.match(r'^#{1,6}\s', line.strip()):
            start_index = i
            break
    
    if start_index is None:
        removed_content["before_first_heading"] = text
        return "", removed_content
    
    removed_content["before_first_heading"] = "\n".join(lines[:start_index])
    filtered_lines = lines[start_index:]
    
    last_headers = []
    for i in range(len(filtered_lines)-1, -1, -1):
        line = filtered_lines[i].strip()
        if re.match(r'^#{1,6}\s', line):
            header_level = len(re.match(r'^(#+)', line).group(1))
            header_text = re.sub(r'^#{1,6}\s*', '', line).strip()
            last_headers.append((i, header_text, header_level))
            if len(last_headers) >= 3:
                break
    
    ref_index = None
    ref_level = None
    for idx, header_text, header_level in last_headers:
        if "references" in header_text.lower() or "参考文献" in remove_whitespace(header_text):
            ref_index = idx
            ref_level = header_level
            break
    
    if ref_index is not None:
        next_header_index = None
        for i in range(ref_index + 1, len(filtered_lines)):
            line = filtered_lines[i].strip()
            if re.match(r'^#{1,6}\s', line):
                current_level = len(re.match(r'^(#+)', line).group(1))
                if current_level <= ref_level:
                    next_header_index = i
                    break
        
        if next_header_index is not None:
            removed_content["references_section"] = "\n".join(filtered_lines[ref_index:next_header_index])
            filtered_lines = filtered_lines[:ref_index] + filtered_lines[next_header_index:]
        else:
            removed_content["references_section"] = "\n".join(filtered_lines[ref_index:])
            filtered_lines = filtered_lines[:ref_index]
    
    return "\n".join(filtered_lines), removed_content

def read_jsonl(file_path) -> list:
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"解析 JSON 错误: {e}")
    return data_list

def tree(filepath, ignore_dir_names=None, ignore_file_names=None):
    if ignore_dir_names is None:
        ignore_dir_names = []
    if ignore_file_names is None:
        ignore_file_names = []
    ret_list = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print("路径不存在")
            return None, None
        elif os.path.isfile(filepath) and os.path.basename(filepath) not in ignore_file_names:
            return [filepath], [os.path.basename(filepath)]
        elif os.path.isdir(filepath) and os.path.basename(filepath) not in ignore_dir_names:
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                if os.path.isfile(fullfilepath) and os.path.basename(fullfilepath) not in ignore_file_names:
                    ret_list.append(fullfilepath)
                if os.path.isdir(fullfilepath) and os.path.basename(fullfilepath) not in ignore_dir_names:
                    ret_list.extend(tree(fullfilepath, ignore_dir_names, ignore_file_names)[0])
    return ret_list, [os.path.basename(p) for p in ret_list]

from bs4 import BeautifulSoup

def extract_and_convert_tables(md_text):
    def _convert_html_table(html_table):
        soup = BeautifulSoup(html_table, 'html.parser')
        table = soup.find('table')
        if not table:
            return html_table

        rows = table.find_all(['tr'])
        if not rows:
            return html_table

        max_cols = 0
        for tr in rows:
            cols_count = 0
            for cell in tr.find_all(['th', 'td']):
                cols_count += int(cell.get('colspan', 1))
            if cols_count > max_cols:
                max_cols = cols_count
        if max_cols == 0:
            return html_table

        matrix = [['' for _ in range(max_cols)] for _ in range(len(rows))]
        rowspans = [0] * max_cols

        for i, tr in enumerate(rows):
            col_idx = 0
            cells = tr.find_all(['th', 'td'])

            while col_idx < max_cols and rowspans[col_idx] > 0:
                rowspans[col_idx] -= 1
                col_idx += 1

            for cell in cells:
                while col_idx < max_cols and matrix[i][col_idx]:
                    col_idx += 1
                if col_idx >= max_cols:
                    break

                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                text = cell.get_text().strip()

                matrix[i][col_idx] = text

                for r in range(i, i + rowspan):
                    for c in range(col_idx, col_idx + colspan):
                        if r == i and c == col_idx:
                            continue
                        if r < len(matrix) and c < max_cols:
                            matrix[r][c] = ' '

                if rowspan > 1:
                    for c in range(col_idx, col_idx + colspan):
                        if c < len(rowspans):
                            rowspans[c] = rowspan - 1
                col_idx += colspan

        md_lines = []
        for r, row in enumerate(matrix):
            md_row = '| ' + ' | '.join(row) + ' |'
            md_lines.append(md_row)
            if r == 0:
                separator = '| ' + ' | '.join(['---'] * len(row)) + ' |'
                md_lines.append(separator)

        return '\n'.join(md_lines)

    table_pattern = re.compile(
        r'(?:<html[^>]*>\s*<body[^>]*>\s*)?'
        r'<table[^>]*>.*?</table>'
        r'\s*(?:</body>\s*</html>)?',
        re.DOTALL | re.IGNORECASE
    )

    last_end = 0
    output_lines = []
    for match in table_pattern.finditer(md_text):
        output_lines.append(md_text[last_end:match.start()])
        converted = _convert_html_table(match.group(0))
        output_lines.append(converted)
        last_end = match.end()
    output_lines.append(md_text[last_end:])
    return ''.join(output_lines)

def split_markdown_by_headings(md_content, threshold=2048):
    lines = md_content.splitlines(keepends=True)
    chunks = []
    current_chunk = []
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

    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= threshold:
            final_chunks.append(chunk)
            continue

        segments = []
        pos = 0
        while pos < len(chunk):
            inline_match = re.search(r'\$[^$].*?\$', chunk[pos:], re.DOTALL)
            block_match = re.search(r'\$\$.*?\$\$', chunk[pos:], re.DOTALL)
            matches = []
            if inline_match:
                matches.append(inline_match)
            if block_match:
                matches.append(block_match)
            if matches:
                match = min(matches, key=lambda m: m.start())
                start, end = match.span()
                start += pos
                end += pos
                if start > pos:
                    segments.append(chunk[pos:start])
                segments.append(chunk[start:end])
                pos = end
            else:
                segments.append(chunk[pos:])
                break

        current_sub = []
        current_length = 0
        for seg in segments:
            seg_len = len(seg)
            if current_length + seg_len <= threshold:
                current_sub.append(seg)
                current_length += seg_len
            elif seg.startswith('$$') or seg.startswith('$'):
                if current_sub:
                    final_chunks.append(''.join(current_sub))
                    current_sub = []
                    current_length = 0
                final_chunks.append(seg)
            else:
                if current_sub:
                    final_chunks.append(''.join(current_sub))
                    current_sub = []
                    current_length = 0
                start = 0
                while start < seg_len:
                    end = min(start + threshold, seg_len)
                    if end < seg_len:
                        line_break = seg.rfind('\n', start, end)
                        if line_break != -1:
                            end = line_break + 1
                        else:
                            space_pos = seg.rfind(' ', start, end)
                            if space_pos != -1:
                                end = space_pos + 1
                    final_chunks.append(seg[start:end])
                    start = end
        if current_sub:
            final_chunks.append(''.join(current_sub))

    return final_chunks

def remove_md_images(markdown_text):
    pattern = r'!\[[^\]]*\]\s*(?:\([^)]*\)|\[[^\]]*\])'
    return re.sub(pattern, '', markdown_text)

def replace_formulas(text, placeholder=""):
    try:
        from pylatexenc.latexwalker import LatexWalker, LatexMathNode
        walker = LatexWalker(text)
        nodelist, _, _ = walker.get_latex_nodes()
        formula_positions = []

        stack = [nodelist]
        while stack:
            nodes = stack.pop()
            if not nodes:
                continue
            for node in nodes:
                if isinstance(node, LatexMathNode):
                    formula_positions.append((node.pos, node.pos + node.len))
                if hasattr(node, 'nodelist') and node.nodelist:
                    stack.append(node.nodelist)
                if hasattr(node, 'nodeargd') and node.nodeargd:
                    for arg in node.nodeargd.argnlist:
                        if hasattr(arg, 'nodelist') and arg.nodelist:
                            stack.append(arg.nodelist)

        if formula_positions:
            result = list(text)
            for start, end in sorted(formula_positions, reverse=True, key=lambda x: x[0]):
                result[start:end] = placeholder
            return "".join(result)
    except Exception:
        pass
    return remove_formulas_safe(text, placeholder)

def remove_formulas_safe(text, placeholder=""):
    patterns = [
        r'\\begin\{.*?\}.*?\\end\{.*?\}',
        r'\$\$.*?\$\$',
        r'\\\[.*?\\\]',
        r'\\\(.*?\\\)',
        r'\$.*?\$'
    ]
    for pattern in patterns:
        text = re.sub(pattern, placeholder, text, flags=re.DOTALL)
    return text

# ================= 多进程相关 =================

def init_process(model_path):
    """子进程初始化：加载模型"""
    global _process_model
    process_name = mp.current_process().name
    print(f"[{process_name}] 正在加载语言检测模型...")
    try:
        with _init_lock:
            _process_model = fasttext.load_model(model_path)
        print(f"[{process_name}] 模型加载成功")
    except Exception as e:
        print(f"[{process_name}] 模型加载失败: {e}")
        raise e

def write_log_file(log_path, false_list, latex_false_list, doc_id):
    """写入日志文件"""
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write("="*70 + "\n")
        log_file.write(f"文档ID: {doc_id}\n")
        log_file.write("="*70 + "\n\n")
        
        log_file.write("="*50 + "\n")
        log_file.write("文本过滤未通过部分 (false_list):\n")
        log_file.write("="*50 + "\n\n")
        if false_list:
            for i, item in enumerate(false_list, 1):
                log_file.write(f"条目 #{i}:\n")
                log_file.write(f"过滤标记: {json.dumps(item['flag'], ensure_ascii=False)}\n")
                log_file.write(f"原始内容:\n{item['text']}\n")
                log_file.write("-"*50 + "\n\n")
        else:
            log_file.write("无未通过文本\n\n")
         
        log_file.write("="*50 + "\n")
        log_file.write("LaTeX公式过滤未通过部分 (latex_false_list):\n")
        log_file.write("="*50 + "\n\n")
        if latex_false_list:
            for i, item in enumerate(latex_false_list, 1):
                log_file.write(f"条目 #{i}:\n")
                log_file.write(f"过滤标记: {json.dumps(item['flag'], ensure_ascii=False)}\n")
                log_file.write(f"原始内容:\n{item['text']}\n")
                log_file.write("-"*50 + "\n\n")
        else:
            log_file.write("无未通过公式\n\n")
        log_file.write("\n" + "="*70 + "\n\n")

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("单文档处理超时")

def filter_func_with_timeout(params, timeout=300):
    """带超时保护的过滤函数"""
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
    
    try:
        result = filter_func_core(params)
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return result
    except TimeoutError:
        data_json = params[0]
        print(f"⚠️  文档 {data_json.get('id', 'unknown')} 处理超时，已跳过")
        return {
            "false_list": [], 
            "latex_false_list": [], 
            "data_json": data_json
        }
    except Exception as e:
        data_json = params[0]
        print(f"⚠️  文档 {data_json.get('id', 'unknown')} 处理异常: {e}")
        return {
            "false_list": [], 
            "latex_false_list": [], 
            "data_json": data_json
        }

def filter_func_core(params):
    """核心过滤逻辑"""
    data_json, input_dir, output_dir, log_dir = params
    global _process_model
    model = _process_model

    if 'id' not in data_json:
        data_json["id"] = str(uuid.uuid4())
    doc_id = data_json["id"]

    md_content = data_json['text']
    md_content = remove_md_images(md_content)
    md_content = extract_and_convert_tables(md_content)
    data = split_markdown_by_headings(md_content)

    false_list = []
    for i, item in enumerate(data):
        text_no_latex = replace_formulas(item)
        text_lg = text_no_latex.replace('\n', '')
        if len(text_lg) > 0:
            try:
                predictions = model.predict(text_lg, k=1)
                language = predictions[0][0].replace('__label__', '')
                if language == "zh" and predictions[1][0] > 0.8:
                    result, flag = filter_simple(text_no_latex, model)
                    if not result and "total character count" not in flag:
                        false_list.append({"text": item, "flag": flag})
                        data[i] = ""
                elif language == "en" and predictions[1][0] > 0.8:
                    result, flag = filter_one_line(text_no_latex, model)
                    if not result and "total character count" not in flag:
                        false_list.append({"text": item, "flag": flag})
                        data[i] = ""
                else:
                    data[i] = ""
            except Exception as e:
                print(f"语言过滤异常: {e}")
                data[i] = ""
        else:
            data[i] = ""

    md_content_filter = "".join(data)

    try:
        filter_obj = LatexFormulaQualityFilter()
        latex_filtered_text, latex_false_list = filter_obj.filter_low_quality_formulas(md_content_filter, replace_with="")
    except Exception as e:
        print(f"LaTeX过滤异常: {e}")
        latex_filtered_text = md_content_filter
        latex_false_list = []

    data_json['text'] = latex_filtered_text

    # 写日志
    process_id = mp.current_process().name
    log_file_path = os.path.join(log_dir, f"process_{process_id}_log.txt")
    if false_list or latex_false_list:
        write_log_file(log_file_path, false_list, latex_false_list, doc_id)

    return {"false_list": false_list, "latex_false_list": latex_false_list, "data_json": data_json}

# ================= 主函数 =================

def main():
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='数据过滤脚本')
    parser.add_argument('--input_dir', type=str, required=True, help='输入文件夹路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--processes', type=int, default=64, help='使用的进程数量')
    parser.add_argument('--batch_size', type=int, default=10, help='每次读取的文件数量（控制内存）')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    max_workers = args.processes
    file_batch_size = args.batch_size

    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    # 获取所有待处理文件
    file_list, _ = tree(input_dir)
    jsonl_files = [
        f for f in file_list 
        if f.endswith(".jsonl") and not f.endswith("_F.jsonl")
    ]

    print(f"📥 找到 {len(jsonl_files)} 个待处理的JSONL文件")
    print(f"⚙️  使用 {max_workers} 个进程，每批处理 {file_batch_size} 个文件")

    # 创建进程池（只创建一次！）
    pool = mp.Pool(
        processes=max_workers,
        initializer=init_process,
        initargs=(fasttext_model_dir,),
        maxtasksperchild=50  # 处理50个任务后重启子进程，释放内存
    )

    all_results = []
    total_docs_processed = 0

    try:
        # 分批处理文件
        for i in range(0, len(jsonl_files), file_batch_size):
            batch_files = jsonl_files[i:i + file_batch_size]
            print(f"\n📦 正在处理第 {i//file_batch_size + 1} 批文件（共 {len(batch_files)} 个文件）")

            # 读取本批次数据
            batch_data = []
            file_mapping = {}
            for filepath in batch_files:
                rel_path = os.path.relpath(filepath, input_dir)
                output_filepath = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                file_mapping[filepath] = output_filepath

                items = read_jsonl(filepath)
                for item in items:
                    item['_original_file'] = filepath
                    batch_data.append(item)

            if not batch_data:
                continue

            print(f"📋 本批次共 {len(batch_data)} 个文档，开始投递任务...")

            # 投递任务到同一个 Pool
            futures = []
            for item in batch_data:
                params = (item, input_dir, output_dir, log_dir)
                future = pool.apply_async(filter_func_with_timeout, (params, 300))
                futures.append(future)

            # 收集结果
            batch_results = []
            with tqdm(total=len(futures), desc="⏳ 本批次处理进度", unit="docs") as pbar:
                for j, future in enumerate(futures):
                    try:
                        result = future.get(timeout=360)  # 单任务最长等待6分钟
                        batch_results.append(result)
                        all_results.append(result)
                        total_docs_processed += 1
                    except mp.TimeoutError:
                        print(f"\n❌ 任务超时，跳过文档 {j}")
                        dummy_result = {
                            "false_list": [], 
                            "latex_false_list": [], 
                            "data_json": batch_data[j] if j < len(batch_data) else {}
                        }
                        batch_results.append(dummy_result)
                        all_results.append(dummy_result)
                        total_docs_processed += 1
                    except Exception as e:
                        print(f"\n❌ 任务执行错误: {e}")
                        dummy_result = {
                            "false_list": [], 
                            "latex_false_list": [], 
                            "data_json": batch_data[j] if j < len(batch_data) else {}
                        }
                        batch_results.append(dummy_result)
                        all_results.append(dummy_result)
                        total_docs_processed += 1
                    pbar.update(1)

            # 保存本批次结果
            file_results = {}
            for result_item in batch_results:
                data_json = result_item["data_json"]
                original_file = data_json.get('_original_file', '')
                if original_file not in file_results:
                    file_results[original_file] = []
                file_results[original_file].append(data_json)

            for original_file, items in file_results.items():
                if original_file in file_mapping:
                    output_filepath = file_mapping[original_file]
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        for item in items:
                            if '_original_file' in item:
                                del item['_original_file']
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"✅ 本批次处理完成，已保存到输出目录")

    except KeyboardInterrupt:
        print("\n🛑 用户中断，正在清理...")
    except Exception as e:
        print(f"\n🔥 严重错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔒 正在关闭进程池...")
        pool.close()
        pool.join()

    # 保存汇总结果
    filter_pkl = os.path.join(output_dir, "zh_en_filter_result.pkl")
    with open(filter_pkl, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n🎉 全部处理完成!")
    print(f"📊 总计处理文档: {total_docs_processed}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📝 日志目录: {log_dir}")
    print(f"💾 汇总结果: {filter_pkl}")

if __name__ == "__main__":
    main()
