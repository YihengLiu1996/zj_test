import os
import multiprocessing as mp
from pathlib import Path
import pyarrow.parquet as pq
import orjson
import gc
from tqdm import tqdm  # ← 新增！

# ========== 可调参数 ==========
BATCH_SIZE = 100_000  # 每批处理行数
# =============================

def parquet_to_jsonl_task(args):
    """
    单文件转换任务（分批读取 + 分批写入，内存安全）
    """
    parquet_path, output_root, input_root = args
    try:
        gc.disable()  # 关闭 GC 提速

        # 计算输出路径
        rel_path = os.path.relpath(parquet_path, input_root)
        jsonl_rel_path = Path(rel_path).with_suffix('.jsonl')
        output_path = Path(output_root) / jsonl_rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 打开 Parquet 文件（不全加载）
        parquet_file = pq.ParquetFile(parquet_path)

        # 使用 iter_batches 分批读取
        with open(output_path, 'wb') as f:
            for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
                lines = []
                for i in range(batch.num_rows):
                    row_dict = {}
                    for col in batch.schema.names:
                        val = batch[col][i].as_py()
                        row_dict[col] = val
                    line = orjson.dumps(row_dict, option=orjson.OPT_APPEND_NEWLINE)
                    lines.append(line)
                f.writelines(lines)
                del lines, batch  # 显式释放

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

    tasks = [(f, output_folder, input_folder) for f in parquet_files]

    # 👇 使用 imap_unordered + tqdm 实现进度条
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # tqdm 包裹迭代器，显示进度
        results = list(tqdm(
            pool.imap_unordered(parquet_to_jsonl_task, tasks),
            total=total_files,
            desc="🔄 转换进度",
            unit="文件",
            ncols=80,
            colour='green'
        ))

    # 统计结果
    success_count = sum(1 for r in results if r.startswith('✅'))
    print(f"\n🎉 转换完成！成功: {success_count} / {total_files}")
    if success_count < total_files:
        print("❗ 以下文件转换失败:")
        for r in results:
            if r.startswith('❌'):
                print(r)


if __name__ == "__main__":
    mp.freeze_support()
    main()
