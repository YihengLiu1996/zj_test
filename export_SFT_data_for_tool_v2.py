import json
import os
import glob
from tqdm import tqdm


def convert_jsonl_folder(input_folder, output_path):
    # 获取所有 .jsonl 文件
    jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {input_folder}")
        return

    total_lines = 0
    # 先统计总行数用于 tqdm 进度条
    for file_path in jsonl_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines += sum(1 for _ in f)

    print(f"Found {len(jsonl_files)} JSONL files, total {total_lines} lines.")
    print(f"Converting and merging into {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        pbar = tqdm(total=total_lines, desc="Processing All Files")

        for file_path in jsonl_files:
            with open(file_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    try:
                        data = json.loads(line.strip())
                        original_text = data.get("text", "")
                        messages = data.get("messages", [])
                        file_path_orig = data.get("file_path", "")

                        # 转换消息格式
                        new_messages = []
                        for msg in messages:
                            if msg["role"] == "assistant":
                                think = msg.get("think", "")
                                answer = msg.get("answer", "")
                                content = f'<think>\n{think}\n</think>\n\n{answer}'
                                new_messages.append({
                                    "role": "assistant",
                                    "content": content
                                })
                            else:
                                new_messages.append({
                                    "role": msg["role"],
                                    "content": msg.get("content", "")
                                })

                        # 构建新数据结构
                        new_data = {
                            "messages": new_messages,
                            "query_source": "multi_distiller",
                            "answer_source": "multi_distiller",
                            "file_path": file_path_orig,
                            "source_chunk": original_text,
                            # 注意：原代码中 "file_path" 重复了，这里保留一次即可
                        }

                        # 写入输出文件
                        f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')

                    except Exception as e:
                        print(f"Error processing line in {file_path}: {str(e)}")
                        continue
                    finally:
                        pbar.update(1)

        pbar.close()

    print(f"Conversion and merging complete! Output saved to {output_path}")


if __name__ == "__main__":
    # 配置参数
    input_folder = "/mnt/ht1_nas3/nanhu_lyh/train_data/sft_data/propulsion_sft_data/3_score_filter/"  # 输入文件夹路径
    output_jsonl = "/mnt/ht1_nas3/nanhu_lyh/train_data/sft_data/propulsion_sft_data/3_score_filter/propulsion_sft_score_8_trans2tool_merged.jsonl"  # 输出合并文件路径

    convert_jsonl_folder(input_folder, output_jsonl)
