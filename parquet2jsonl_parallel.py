import os
import multiprocessing as mp
from pathlib import Path
import pyarrow.parquet as pq
import orjson
import gc

def parquet_to_jsonl_task(args):
    """
    单文件转换任务（关闭 GC，极速模式）
    """
    parquet_path, output_root, input_root = args
    try:
        # 关闭子进程 GC，提升性能
        gc.disable()

        # 计算输出路径（保持结构）
        rel_path = os.path.relpath(parquet_path, input_root)
        jsonl_rel_path = Path(rel_path).with_suffix('.jsonl')
        output_path = Path(output_root) / jsonl_rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用 PyArrow 读取整表（内存充足，直接全读）
        table = pq.read_table(parquet_path)
        # 转为 RecordBatch（更高效）
        batches = table.to_batches()
        lines = []

        # 批量序列化为 JSONL 行
        for batch in batches:
            for i in range(batch.num_rows):
                row_dict = {}
                for col in batch.column_names:
                    val = batch[col][i].as_py()
                    row_dict[col] = val
                # orjson 序列化 + 换行
                line = orjson.dumps(row_dict, option=orjson.OPT_APPEND_NEWLINE)
                lines.append(line)

        # 一次性写入（最大化 I/O 吞吐）
        with open(output_path, 'wb') as f:
            f.writelines(lines)

        # 重新启用 GC（可选，进程即将退出）
        gc.enable()
        return f"✅ {parquet_path}"

    except Exception as e:
        return f"❌ {parquet_path} | {e}"


def find_parquet_files_fast(root_dir):
    """
    使用 os.scandir 递归查找 .parquet 文件（更快）
    """
    parquet_files = []
    stack = [root_dir]
    while stack:
        current_dir = stack.pop()
        try:
            with os.scandir(current_dir) as it:
                for entry in it:
                    if entry.is_file() and entry.name.lower().endswith('.parquet'):
                        parquet_files.append(entry.path)
                    elif entry.is_dir():
                        stack.append(entry.path)
        except PermissionError:
            continue
    return parquet_files


def main():
    input_folder = input("📥 请输入包含 parquet 文件的文件夹路径: ").strip()
    output_folder = input("📤 请输入输出 jsonl 文件的文件夹路径: ").strip()

    if not os.path.exists(input_folder):
        print("❌ 输入文件夹不存在！")
        return

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    print("🔍 正在扫描 parquet 文件...")
    parquet_files = find_parquet_files_fast(input_folder)

    if not parquet_files:
        print("⚠️  未找到任何 .parquet 文件！")
        return

    total_files = len(parquet_files)
    print(f"🚀 找到 {total_files} 个文件，启动 {mp.cpu_count()} 进程并行转换...")

    # 构建任务参数
    tasks = [(f, output_folder, input_folder) for f in parquet_files]

    # 创建进程池（默认 CPU 核数）
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(parquet_to_jsonl_task, tasks)

    # 打印结果摘要（避免逐行 print 影响性能）
    success_count = sum(1 for r in results if r.startswith('✅'))
    print(f"\n🎉 转换完成！成功: {success_count} / {total_files}")
    if success_count < total_files:
        print("❗ 以下文件转换失败:")
        for r in results:
            if r.startswith('❌'):
                print(r)


if __name__ == "__main__":
    mp.freeze_support()  # Windows 兼容
    main()
