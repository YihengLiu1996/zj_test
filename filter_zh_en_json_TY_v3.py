import os
import fasttext
import fasttext.util
import json
import multiprocessing as mp
import pickle
import concurrent.futures
import sys
import re
from pylatexenc.latexwalker import LatexWalker, LatexMathNode
from latex_filter import LatexFormulaQualityFilter
from my_utils import md_to_jsonl
import uuid
import time
from tqdm import tqdm  # 新增进度条库
import psutil
import argparse  # 新增参数解析库
  
sys.path.append(os.path.join(os.path.dirname(__file__), './english_data_clean_pipeline/filter'))
sys.path.append(os.path.join(os.path.dirname(__file__), './chinese_data_clean_pipeline/filter'))
  
from english_data_clean_pipeline.filter.util_with_try import filter_one_line
from chinese_data_clean_pipeline.filter.filter import filter_simple
  
# 移除全局模型加载 - 每个进程将独立加载
fasttext_model_dir = "/mnt/sizjwb25c1g7/nanhu_lyh/code_tx/data_process_code/MAP-NEO-main/Matrix/lid.176.bin"
  
def remove_whitespace(input_str):
    """移除字符串中所有空白字符（空格、制表符、换行符等）"""
    return ''.join(input_str.split())
  
def filter_markdown(text):
    if not text:
        return "", {"before_first_heading": "", "references_section": ""}
      
    lines = text.splitlines()
      
    # 记录被过滤的内容
    removed_content = {
        "before_first_heading": "",
        "references_section": ""
    }
      
    # 步骤1：删除第一个标题前的所有内容
    start_index = None
    for i, line in enumerate(lines):
        if re.match(r'^#{1,6}\s', line.strip()):
            start_index = i
            break
      
    if start_index is None:
        # 如果没有标题，整个文本视为"before_first_heading"
        removed_content["before_first_heading"] = text
        return "", removed_content
      
    # 记录并删除第一个标题前的内容
    removed_content["before_first_heading"] = "\n".join(lines[:start_index])
    filtered_lines = lines[start_index:]
      
    # 步骤2：处理参考文献部分
    # 收集最后三个标题及其行号
    last_headers = []
    for i in range(len(filtered_lines)-1, -1, -1):
        line = filtered_lines[i].strip()
        if re.match(r'^#{1,6}\s', line):
            # 提取标题文本和级别
            header_level = len(re.match(r'^(#+)', line).group(1))
            header_text = re.sub(r'^#{1,6}\s*', '', line).strip()
            last_headers.append((i, header_text, header_level))
            if len(last_headers) >= 3:
                break
      
    # 检查最后三个标题是否包含"references"
    ref_index = None
    ref_level = None
    for idx, header_text, header_level in last_headers:
        if "references" in header_text.lower() or "参考文献" in remove_whitespace(header_text):
            ref_index = idx
            ref_level = header_level
            break
      
    # 如果找到参考文献标题，只删除该章节内容
    if ref_index is not None:
        # 查找下一个同级或更高级别的标题
        next_header_index = None
        for i in range(ref_index + 1, len(filtered_lines)):
            line = filtered_lines[i].strip()
            if re.match(r'^#{1,6}\s', line):
                current_level = len(re.match(r'^(#+)', line).group(1))
                # 如果找到同级或更高级别的标题
                if current_level <= ref_level:
                    next_header_index = i
                    break
          
        # 确定删除范围
        if next_header_index is not None:
            # 记录被删除的参考文献章节
            removed_content["references_section"] = "\n".join(filtered_lines[ref_index:next_header_index])
            # 删除参考文献章节
            filtered_lines = filtered_lines[:ref_index] + filtered_lines[next_header_index:]
        else:
            # 记录被删除的参考文献章节
            removed_content["references_section"] = "\n".join(filtered_lines[ref_index:])
            # 如果没有后续标题，只删除参考文献标题及之后内容
            filtered_lines = filtered_lines[:ref_index]
      
    return "\n".join(filtered_lines), removed_content
  
  
def read_jsonl(file_path)->list:
    """
    读取jsonl文件，返回json列表
    """
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"解析 JSON 错误: {e}")
    return data_list
  
  
def detect_language(text):
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]+', text))
    return 'zh' if chinese_chars / max(1, len(text)) > 0.3 else 'en'
  
def tree(filepath, ignore_dir_names=None, ignore_file_names=None):
    """返回两个列表，第一个列表为 filepath 下全部文件的完整路径, 第二个为对应的文件名"""
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
  
  
def check_files_in_directory(directory):
    # 获取指定目录下的所有文件和文件夹
    files = os.listdir(directory)
  
    # 检查是否存在以 'content_list_F.json' 结尾的文件
    for file in files:
        if file.endswith('content_list_F.json'):
            return True
    return False
  
  
from bs4 import BeautifulSoup
  
def extract_and_convert_tables(md_text):
    """
    从Markdown文本中提取HTML表格，转换为Markdown格式并替换回原位置
    - 处理合并单元格：拆分成独立单元格，仅左上角保留内容，其他设为空格
    - 适配两种表格格式：直接<table>标签和<html><body>包裹的表格
    """
  
    def _convert_html_table(html_table):
        # 解析整个HTML片段（可能包含<html><body>）
        soup = BeautifulSoup(html_table, 'html.parser')
  
        # 查找第一个表格（忽略外层标签）
        table = soup.find('table')
        if not table:
            return html_table
  
        # 获取所有行
        rows = table.find_all(['tr'])
        if not rows:
            return html_table
  
        # 计算最大列数
        max_cols = 0
        for tr in rows:
            cols_count = 0
            for cell in tr.find_all(['th', 'td']):
                cols_count += int(cell.get('colspan', 1))
            if cols_count > max_cols:
                max_cols = cols_count
        if max_cols == 0:
            return html_table
  
        # 创建填充矩阵
        matrix = [['' for _ in range(max_cols)] for _ in range(len(rows))]
        rowspans = [0] * max_cols  # 跟踪每列的剩余rowspan
  
        # 填充矩阵内容
        for i, tr in enumerate(rows):
            col_idx = 0
            cells = tr.find_all(['th', 'td'])
  
            # 跳过被上方rowspan占用的列
            while col_idx < max_cols and rowspans[col_idx] > 0:
                rowspans[col_idx] -= 1
                col_idx += 1
  
            for cell in cells:
                # 跳过已填充的列（由colspan导致）
                while col_idx < max_cols and matrix[i][col_idx]:
                    col_idx += 1
                if col_idx >= max_cols:
                    break
  
                # 获取单元格属性
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))
                text = cell.get_text().strip()
  
                # 填充左上角单元格
                matrix[i][col_idx] = text
  
                # 标记合并区域（除左上角外设为空格）
                for r in range(i, i + rowspan):
                    for c in range(col_idx, col_idx + colspan):
                        if r == i and c == col_idx:
                            continue
                        if r < len(matrix) and c < max_cols:
                            matrix[r][c] = ' '  # 非左上角设为空格
  
                # 更新rowspan跟踪器
                if rowspan > 1:
                    for c in range(col_idx, col_idx + colspan):
                        if c < len(rowspans):
                            rowspans[c] = rowspan - 1
                col_idx += colspan
  
        # 构建Markdown表格
        md_lines = []
        for r, row in enumerate(matrix):
            # 构建行内容
            md_row = '| ' + ' | '.join(row) + ' |'
            md_lines.append(md_row)
  
            # 添加表头分隔线（在首行之后）
            if r == 0:
                separator = '| ' + ' | '.join(['---'] * len(row)) + ' |'
                md_lines.append(separator)
  
        return '\n'.join(md_lines)
  
    # 修改后的正则表达式：匹配两种格式的表格
    table_pattern = re.compile(
        r'(?:<html[^>]*>\s*<body[^>]*>\s*)?'  # 开头的<html><body>（可选）
        r'<table[^>]*>.*?</table>'  # 表格内容
        r'\s*(?:</body>\s*</html>)?',  # 结尾的</body></html>（可选）
        re.DOTALL | re.IGNORECASE
    )
  
    last_end = 0
    output_lines = []
    for match in table_pattern.finditer(md_text):
        # 添加表格前的内容
        output_lines.append(md_text[last_end:match.start()])
  
        # 转换表格并添加
        converted = _convert_html_table(match.group(0))
        output_lines.append(converted)
  
        last_end = match.end()
  
    # 添加最后一段内容
    output_lines.append(md_text[last_end:])
  
    return ''.join(output_lines)
  
def split_markdown_by_headings(md_content, threshold=2048):
    # First pass: split by headings
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
      
    # Second pass: further split chunks that exceed the threshold
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= threshold:
            final_chunks.append(chunk)
            continue
              
        # Split into segments while preserving LaTeX formulas
        segments = []
        pos = 0
        while pos < len(chunk):
            # Find next LaTeX formula (either $$...$$ or $...$)
            inline_match = re.search(r'\$[^$].*?\$', chunk[pos:], re.DOTALL)
            block_match = re.search(r'\$\$.*?\$\$', chunk[pos:], re.DOTALL)
              
            # Find closest formula match
            matches = []
            if inline_match:
                matches.append(inline_match)
            if block_match:
                matches.append(block_match)
              
            if matches:
                # Find the earliest match
                match = min(matches, key=lambda m: m.start())
                start, end = match.span()
                start += pos
                end += pos
                  
                # Add text before formula
                if start > pos:
                    segments.append(chunk[pos:start])
                  
                # Add formula as a separate segment
                segments.append(chunk[start:end])
                pos = end
            else:
                # No more formulas, add remaining text
                segments.append(chunk[pos:])
                break
          
        # Merge segments into chunks respecting threshold
        current_sub = []
        current_length = 0
        for seg in segments:
            seg_len = len(seg)
              
            # Case 1: Segment fits in current chunk
            if current_length + seg_len <= threshold:
                current_sub.append(seg)
                current_length += seg_len
              
            # Case 2: Segment is formula and exceeds threshold alone
            elif seg.startswith('$$') or seg.startswith('$'):
                # Flush current chunk if exists
                if current_sub:
                    final_chunks.append(''.join(current_sub))
                    current_sub = []
                    current_length = 0
                  
                # Add formula as standalone chunk (even if over threshold)
                final_chunks.append(seg)
              
            # Case 3: Segment doesn't fit and is regular text
            else:
                # Flush current chunk
                if current_sub:
                    final_chunks.append(''.join(current_sub))
                    current_sub = []
                    current_length = 0
                  
                # Split large text segment
                start = 0
                while start < seg_len:
                    end = min(start + threshold, seg_len)
                    # Try to split at natural boundaries
                    if end < seg_len:
                        # Prefer splitting at line breaks
                        line_break = seg.rfind('\n', start, end)
                        if line_break != -1:
                            end = line_break + 1
                        # Otherwise split at last space
                        else:
                            space_pos = seg.rfind(' ', start, end)
                            if space_pos != -1:
                                end = space_pos + 1
                      
                    final_chunks.append(seg[start:end])
                    start = end
          
        # Add last sub-chunk if any
        if current_sub:
            final_chunks.append(''.join(current_sub))
      
    return final_chunks
  
  
def remove_md_images(markdown_text):
    pattern = r'!\[[^\]]*\]\s*(?:\([^)]*\)|\[[^\]]*\])'
    return re.sub(pattern, '', markdown_text)
  
  
def replace_formulas(text, placeholder=""):
    """
    更健壮的公式替换方案，结合解析器和正则回退
    """
    # 尝试使用latexwalker解析（精确模式）
    try:
        from pylatexenc.latexwalker import LatexWalker, LatexMathNode  # 修正导入路径
          
        walker = LatexWalker(text)
        nodelist, _, _ = walker.get_latex_nodes()
        formula_positions = []
          
        # 使用栈替代递归防止栈溢出
        stack = [nodelist]
        while stack:
            nodes = stack.pop()
            if not nodes:
                continue
                  
            for node in nodes:
                if isinstance(node, LatexMathNode):
                    formula_positions.append((node.pos, node.pos + node.len))
                  
                # 处理子节点
                if hasattr(node, 'nodelist') and node.nodelist:
                    stack.append(node.nodelist)
                  
                # 处理参数节点
                if hasattr(node, 'nodeargd') and node.nodeargd:
                    for arg in node.nodeargd.argnlist:
                        if hasattr(arg, 'nodelist') and arg.nodelist:
                            stack.append(arg.nodelist)
          
        # 从后向前替换
        if formula_positions:
            result = list(text)
            for start, end in sorted(formula_positions, reverse=True, key=lambda x: x[0]):
                result[start:end] = placeholder
            return "".join(result)
      
    except Exception:
        pass  # 解析失败时回退到正则方案
      
    # 正则回退方案（处理解析失败的情况）
    return remove_formulas_safe(text, placeholder)
  
def remove_formulas_safe(text, placeholder=""):
    """
    安全删除公式的正则方案，处理极端情况
    """
    # 公式环境模式（按优先级排序）
    patterns = [
        r'\\begin\{.*?\}.*?\\end\{.*?\}',  # 各种环境
        r'\$\$.*?\$\$',                    # 块公式
        r'\\\[.*?\\\]',                    # 块公式
        r'\\\(.*?\\\)',                    # 行内公式
        r'\$.*?\$'                         # 行内公式（最后处理）
    ]
      
    # 分阶段替换，避免嵌套问题
    for pattern in patterns:
        text = re.sub(
            pattern, 
            placeholder, 
            text, 
            flags=re.DOTALL  # 匹配换行符
        )
      
    return text
  
# 移除全局模型变量，改为在子进程中初始化
def init_process(model_path):
    """子进程初始化函数，用于加载FastText模型"""
    global _process_model
    print(f"Loading model in process {mp.current_process().name}")
    _process_model = fasttext.load_model(model_path)
    print(f"Model loaded in process {mp.current_process().name}")

def write_log_file(log_path, false_list, latex_false_list, doc_id):
    """写入日志文件"""
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write("="*70 + "\n")
        log_file.write(f"文档ID: {doc_id}\n")
        log_file.write("="*70 + "\n\n")
        
        # 写入文本过滤未通过部分
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
         
        # 写入LaTeX公式过滤未通过部分
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

def filter_func(params):
    """处理单个文档的过滤函数"""
    data_json, input_dir, output_dir, log_dir = params
      
    # 使用全局模型变量（在子进程中已初始化）
    global _process_model
    model = _process_model
      
    true_list = []
    false_list = []
    md_content = data_json['text']
    if 'id' not in data_json:
        data_json["id"] = str(uuid.uuid4())
    doc_id = data_json["id"]
    md_content = remove_md_images(md_content)  #去除图片链接
    md_content = extract_and_convert_tables(md_content)  #转换table为markdown格式
    # md_content, remove_content = filter_markdown(md_content)  #去除论文的开头和参考文献
    # if len(remove_content["before_first_heading"]) > 0:
    #     false_list.append({"text": remove_content["before_first_heading"], "flag": "before_first_heading"})
    # if len(remove_content["references_section"]) > 0:
    #     false_list.append({"text": remove_content["references_section"], "flag": "references_section"})
    data = split_markdown_by_headings(md_content)
    for i, item in enumerate(data):
        text_no_latex = replace_formulas(item)
        text_lg = text_no_latex.replace('\n', '')
        if len(text_lg) > 0:
            predictions = model.predict(text_lg, k=1)  # k=1 表示返回最可能的语言
            language = predictions[0][0].replace('__label__', '')
            if language == "zh" and predictions[1][0] > 0.8:
                result, flag = filter_simple(text_no_latex, model)
                if result:
                    true_list.append({"text": item, "flag": flag})
                else:
                    if "total character count" not in flag:
                        false_list.append({"text": item, "flag": flag})
                        data[i] = ""
            elif language == "en" and predictions[1][0] > 0.8:
                result, flag = filter_one_line(text_no_latex, model)
                if result:
                    true_list.append({"text": item, "flag": flag})
                else:
                    if "total character count" not in flag:
                        false_list.append({"text": item, "flag": flag})
                        data[i] = ""
            else:
                data[i] = ""
  
    md_content_filter = "".join(data)
  
    # latex公式专用过滤
    filter = LatexFormulaQualityFilter()
    latex_filtered_text, latex_false_list = filter.filter_low_quality_formulas(md_content_filter, replace_with="")
  
    data_json['text'] = latex_filtered_text
    
    # 获取当前进程ID，用于创建独立的日志文件
    process_id = mp.current_process().name
    log_file_path = os.path.join(log_dir, f"process_{process_id}_log.txt")
    
    # 写入日志（追加模式）
    if false_list or latex_false_list:
        write_log_file(log_file_path, false_list, latex_false_list, doc_id)
    
    return {"false_list": false_list, "latex_false_list": latex_false_list, "data_json": data_json}
 
 
  
if __name__ == "__main__":
    # 设置多进程启动方法为spawn
    mp.set_start_method('spawn', force=True)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数据过滤脚本')
    parser.add_argument('--input_dir', type=str, required=True, help='输入文件夹路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--processes', type=int, default=64, help='使用的进程数量')
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    max_workers = args.processes
    
    # 创建输出目录和日志目录
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取所有jsonl文件
    file_list, file_name = tree(input_dir)
    file_process_list = []
    data_process_list = []
    file_mapping = {}  # 存储原始文件路径到输出文件路径的映射
    
    for i, filepath in enumerate(file_list):
        if os.path.basename(filepath).endswith(".jsonl") and not os.path.basename(filepath).endswith("_F.jsonl"):
            file_process_list.append(filepath)
            
            # 计算相对路径并创建对应的输出目录结构
            rel_path = os.path.relpath(filepath, input_dir)
            output_filepath = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            file_mapping[filepath] = output_filepath
            
            # 读取文件内容
            data_items = read_jsonl(filepath)
            for item in data_items:
                # 添加原始文件路径信息，用于后续保存
                item['_original_file'] = filepath
                data_process_list.append(item)
      
    print(f"找到 {len(data_process_list)} 个待处理文档，开始并行处理...")
     
    # 计算合理的进程数
    cpu_count = mp.cpu_count()
    memory_available = psutil.virtual_memory().available
    model_size = 100 * 1024 * 1024  # 100MB
     
    print(f"使用 {max_workers} 个进程进行并行处理 (CPU核心数: {cpu_count}, 可用内存: {memory_available / (1024**3):.2f}GB)")
     
    # 准备任务参数
    task_params = [(data_item, input_dir, output_dir, log_dir) for data_item in data_process_list]
     
    results = []
     
    # 使用进程池，但添加超时和错误处理机制
    with mp.Pool(
        processes=max_workers,
        initializer=init_process,
        initargs=(fasttext_model_dir,),
        maxtasksperchild=1000  # 每个子进程最多处理1000个任务后重启，避免资源泄漏
    ) as pool:
        try:
            # 使用imap_unordered，它比imap更高效
            with tqdm(total=len(task_params), desc="文档处理进度", unit="docs") as pbar:
                # 设置超时时间为10小时
                timeout = 36000
                start_time = time.time()
                 
                # 分批处理，避免一次性提交太多任务
                batch_size = 12800
                for i in range(0, len(task_params), batch_size):
                    batch_params = task_params[i:i+batch_size]
                     
                    # 提交批次任务
                    futures = []
                    for param in batch_params:
                        future = pool.apply_async(filter_func, (param,))
                        futures.append(future)
                     
                    # 等待批次任务完成
                    for j, future in enumerate(futures):
                        try:
                            # 设置单个任务的超时时间
                            result = future.get(timeout=timeout - (time.time() - start_time))
                            results.append(result)
                        except mp.TimeoutError:
                            print(f"\n任务超时，跳过该任务")
                            # 添加一个空结果以保持索引一致
                            results.append({
                                "false_list": [], 
                                "latex_false_list": [], 
                                "data_json": batch_params[j][0] if j < len(batch_params) else {}
                            })
                        except Exception as e:
                            print(f"\n任务执行错误: {e}")
                            # 添加一个空结果以保持索引一致
                            results.append({
                                "false_list": [], 
                                "latex_false_list": [], 
                                "data_json": batch_params[j][0] if j < len(batch_params) else {}
                            })
                         
                        pbar.update(1)
                         
                        # 检查总超时
                        if time.time() - start_time > timeout:
                            print(f"\n总处理时间超时，终止处理")
                            break
                     
                    # 检查总超时
                    if time.time() - start_time > timeout:
                        break
                         
        except Exception as e:
            print(f"\n进程池执行错误: {e}")
        finally:
            # 确保进程池正确关闭
            pool.close()
            pool.join()
     
    # 按原始文件分组保存结果
    file_results = {}
    for result_item in results:
        data_json = result_item["data_json"]
        original_file = data_json.get('_original_file', '')
        if original_file not in file_results:
            file_results[original_file] = []
        file_results[original_file].append(data_json)
    
    # 保存每个文件的结果
    for original_file, items in file_results.items():
        if original_file in file_mapping:
            output_filepath = file_mapping[original_file]
            with open(output_filepath, 'w', encoding='utf-8') as jsonl_file:
                for item in items:
                    # 移除临时添加的字段
                    if '_original_file' in item:
                        del item['_original_file']
                    jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存汇总的pkl文件
    filter_pkl = os.path.join(output_dir, "zh_en_filter_result.pkl")
    with open(filter_pkl, 'wb') as f:
        pickle.dump(results, f)
      
    print(f"\n处理完成! 保存结果到:")
    print(f"- 输出目录: {output_dir}")
    print(f"- 日志目录: {log_dir}")
    print(f"- 过滤结果: {filter_pkl}")
    print(f"共处理 {len(results)} 个文档")
