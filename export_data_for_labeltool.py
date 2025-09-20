import os
import json
import uuid
import argparse
import re
from collections import OrderedDict

# --- Script Configuration ---
DEFAULT_INPUT_PATH = "/mnt/sizjwb25c1g7/nanhu_lyh/train_data/sft_data/zhongzi/htqjgysj/sft_data_zhongzi_htqjgysj_filter_8_sharegpt.jsonl"
DEFAULT_OUTPUT_PATH = "/mnt/sizjwb25c1g7/nanhu_lyh/train_data/sft_data/zhongzi/htqjgysj/sft_data_zhongzi_htqjgysj_filter_8_tool.jsonl"
DEFAULT_MAX_OBJECTS_PER_FILE = 0
# --- End of Configuration ---

# --- Set the judge prompt ---
PROMPT = "请对以下的问题回答对进行评价，"

# --- Field Mapping Configuration ---
FIELD_MAPPING = {
    "conversations_key": "messages",
    "role_key": "role",
    "content_key": "content",
    "user_role_value": "user",
    "assistant_role_value": "assistant",
}
# --- End of Field Mapping ---

def format_assistant_content(content):
    """
    将 <think>xxx</think>yyy 格式的助手回答转换为：
    # 思考过程：
    xxx
    # 最终回答：
    yyy
    """
    pattern = r'<think>(.*?)</think>(.*)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        thought = match.group(1).strip()
        final_answer = match.group(2).strip()
        return f"# 思考过程：\n\n{thought}\n\n# 最终回答：\n\n{final_answer}"
    else:
        # 不符合格式的，原样返回
        return content


def convert_sharegpt_to_labelllm(sharegpt_obj, mapping):
    """
    Converts a single ShareGPT JSON object to a LabelLLM JSON object using the provided field mapping.
    Supports multi-turn conversations and formats assistant's <think>...</think> content.
    """
    conversations_key = mapping["conversations_key"]
    role_key = mapping["role_key"]
    content_key = mapping["content_key"]
    user_role_value = mapping["user_role_value"]
    assistant_role_value = mapping["assistant_role_value"]

    labelllm_obj = {
        "prompt": PROMPT,
        "conversation": [],
        "custom": {
            "query_source": sharegpt_obj.get("query_source", ""),
            "answer_source": sharegpt_obj.get("answer_source", ""),
            "file_path": sharegpt_obj.get("file_path", ""),
            "source_chunk": sharegpt_obj.get("source_chunk", "")
        }
    }

    conversations = sharegpt_obj.get(conversations_key, [])
    if not conversations:
        return labelllm_obj

    parent_id = None  # 初始消息无父节点

    for turn in conversations:
        role = turn.get(role_key)
        content = turn.get(content_key, "")
        if not content:
            continue

        # 如果是助手回复，尝试格式化思考+回答结构
        if role == assistant_role_value:
            content = format_assistant_content(content)

        message_id = str(uuid.uuid4())
        message_type = "send" if role == user_role_value else "receive"
        user_id = sharegpt_obj.get("query_source" if role == user_role_value else "answer_source", "")

        labelllm_obj["conversation"].append({
            "message_id": message_id,
            "content": content,
            "message_type": message_type,
            "user_id": user_id,
            "parent_id": parent_id
        })

        parent_id = message_id  # 链式更新父ID

    return labelllm_obj


def write_merged_data(merged_data, output_path, max_objects):
    """
    Writes the merged data to one or more .jsonl files, splitting them based on the number of objects.
    """
    if not merged_data:
        print("No data to write.")
        return

    max_objects_per_file = max_objects if max_objects > 0 else float('inf')
    part_num = 1
    object_count = 0
    outfile = None
    base_output_filename = "merged_output"

    def open_new_file():
        nonlocal outfile, part_num
        if outfile:
            outfile.close()
        output_file_path = output_path
        print(f"Creating new output file: {output_file_path}")
        outfile = open(output_file_path, 'w', encoding='utf-8')
        part_num += 1
        return outfile

    outfile = open_new_file()

    for labelllm_obj in merged_data.values():
        if object_count > 0 and object_count >= max_objects_per_file:
            outfile = open_new_file()
            object_count = 0

        line = json.dumps(labelllm_obj, ensure_ascii=False) + '\n'
        outfile.write(line)
        object_count += 1

    if outfile:
        outfile.close()
    print("\nFinished writing all data.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert and merge ShareGPT-like format jsonl files to LabelLLM format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_path", default=DEFAULT_INPUT_PATH, help="Directory containing input .jsonl files (searched recursively).")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH, help="Directory to save the output file(s).")
    parser.add_argument("--max_objects", type=int, default=0, help="Maximum number of JSON objects per output file. 0 for no limit.")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    if not input_path or not output_path:
        print("Error: Output directories must be specified either in the script or via command-line arguments.")
        return

    merged_data = OrderedDict()

    conversations_key = FIELD_MAPPING["conversations_key"]
    role_key = FIELD_MAPPING["role_key"]
    content_key = FIELD_MAPPING["content_key"]
    user_role_value = FIELD_MAPPING["user_role_value"]
    assistant_role_value = FIELD_MAPPING["assistant_role_value"]

    print(f"Processing {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                sharegpt_obj = json.loads(line)
                conversations = sharegpt_obj.get(conversations_key, [])
                if not conversations:
                    continue

                first_user_turn = next((c for c in conversations if c.get(role_key) == user_role_value), None)
                if not first_user_turn:
                    continue
                user_query = first_user_turn.get(content_key)
                if not user_query:
                    continue

                if user_query not in merged_data:
                    merged_data[user_query] = convert_sharegpt_to_labelllm(sharegpt_obj, FIELD_MAPPING)
                else:
                    existing_obj = merged_data[user_query]
                    existing_conv = existing_obj["conversation"]

                    last_message = existing_conv[-1] if existing_conv else None
                    parent_id = last_message["message_id"] if last_message else None

                    for turn in conversations:
                        role = turn.get(role_key)
                        content = turn.get(content_key, "")
                        if not content:
                            continue

                        # 如果是助手回复，格式化思考过程
                        if role == assistant_role_value:
                            content = format_assistant_content(content)

                        # 简单去重：跳过已有相同内容的消息
                        if any(msg["content"] == content for msg in existing_conv):
                            continue

                        message_id = str(uuid.uuid4())
                        message_type = "send" if role == user_role_value else "receive"
                        user_id = sharegpt_obj.get("query_source" if role == user_role_value else "answer_source", "")

                        existing_conv.append({
                            "message_id": message_id,
                            "content": content,
                            "message_type": message_type,
                            "user_id": user_id,
                            "parent_id": parent_id
                        })
                        parent_id = message_id

            except (json.JSONDecodeError, IndexError, KeyError) as e:
                print(f"Skipping invalid line {line_num} in {input_path}: {e} | Line: {line.strip()}")

    write_merged_data(merged_data, output_path, args.max_objects)

    print("Conversion and merge complete.")


if __name__ == "__main__":
    main()
