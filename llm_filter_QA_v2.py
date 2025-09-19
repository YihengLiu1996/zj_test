import os
import json
import random
import re
import asyncio
import argparse
from tqdm import tqdm
import threading
from openai import AsyncOpenAI
from typing import List, Dict, Any, Tuple, Optional
import json_repair

# ================= 工具函数 =================

def extract_think_answer(text):
    pattern = r'(.*?)(?:<think>(.*?)</think>)(.*)'
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        if '</think>' in text:
            text_list = text.split('</think>')
            answer_content = "".join(text_list[1:])
            return {"think": text_list[0].replace("<think>", "").strip(), "answer": answer_content.strip()}
        else:
            return {"think": "", "answer": text.replace("<think>", "").strip()}

    before_think = match.group(1) or ""
    think_content = match.group(2) or ""
    after_think = match.group(3) or ""

    answer_content = after_think.strip()

    return {
        "think": think_content.strip(),
        "answer": answer_content
    }

def get_uuid_jsonl(file_path: str) -> List[str]:
    """从输出目录获取已处理的UUID列表"""
    uuid_list = []
    for root, _, files in os.walk(file_path):
        for file in files:
            if file.endswith(".jsonl"):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                uuid_list.append(data.get("id", ""))
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    continue
    return uuid_list

def read_all_jsonl_in_dir(input_dir: str, output_dir: str) -> List[Dict]:
    """读取输入文件夹内所有.jsonl，跳过已处理的数据"""
    data_list = []
    processed_uuids = set(get_uuid_jsonl(output_dir))

    for filename in os.listdir(input_dir):
        if not filename.endswith(".jsonl"):
            continue
        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("id", "") not in processed_uuids:
                        data_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"解析JSON错误: {e} - 跳过该行")
    return data_list

def merge_tmp_files(tmp_dir: str, output_file_path: str) -> None:
    """合并临时文件到输出文件"""
    tmp_files = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
                if f.startswith("tmp_") and f.endswith(".jsonl")]

    if not tmp_files:
        return

    with open(output_file_path, 'a', encoding='utf-8') as outfile:
        for tmp_file in tmp_files:
            try:
                with open(tmp_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
                os.remove(tmp_file)
            except Exception as e:
                print(f"合并文件 {tmp_file} 时出错: {e}")

def convert_msg_to_str(data):
    messages = data["messages"]
    output_lines = []

    for i in range(0, len(messages), 2):
        if i + 1 >= len(messages):
            break

        user_msg = messages[i]
        assistant_msg = messages[i + 1]

        if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
            question = user_msg["content"]
            think = assistant_msg.get("think", "")
            answer = assistant_msg.get("answer", "")

            output_lines.append(f"问题{i//2 + 1}：{question}")
            output_lines.append(f"回答{i//2 + 1}：")
            output_lines.append("思考过程：")
            output_lines.append(think)
            output_lines.append("最终答案：")
            output_lines.append(answer)

    return "\n".join(output_lines)

# ================= 异步重试装饰器 =================
async def with_retry(coro_func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries + 1):
        try:
            return await coro_func()
        except Exception as e:
            if attempt == max_retries:
                raise e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"请求失败，第 {attempt + 1} 次重试，等待 {delay:.2f} 秒...")
            await asyncio.sleep(delay)

# ================= 过滤器函数 =================
class FilterManager:
    def __init__(self, prompts_loader):
        self.prompts_loader = prompts_loader

    async def llm_question_filter(self, data_item: Dict, client: AsyncOpenAI, model_name: str) -> Dict:
        prompt_template = self.prompts_loader.get_prompt("filter", "question_filter")
        question = data_item["messages"][0]["content"]
        prompt = prompt_template.format(question=question)

        async def _call():
            completion = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                timeout=36000,
            )
            return completion

        completion = await with_retry(_call, max_retries=3)
        result = extract_think_answer(completion.choices[0].message.content)
        try:
            return json_repair.loads(result['answer'])
        except:
            return {"is_filter": False, "reason": "解析异常"}

    async def llm_answer_filter(self, data_item: Dict, client: AsyncOpenAI, model_name: str) -> Dict:
        prompt_template = self.prompts_loader.get_prompt("filter", "answer_filter")
        qa_pair = convert_msg_to_str(data_item)
        prompt = prompt_template.format(qa_pair=qa_pair)

        async def _call():
            completion = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                timeout=36000,
            )
            return completion

        completion = await with_retry(_call, max_retries=3)
        result = extract_think_answer(completion.choices[0].message.content)
        try:
            return json_repair.loads(result['answer'])
        except:
            return {"is_filter": False, "reason": "解析异常"}

    async def llm_qa_score(self, data_item: Dict, client: AsyncOpenAI, model_name: str) -> Dict:
        prompt_template = self.prompts_loader.get_prompt("filter", "qa_score")
        qa_pair = convert_msg_to_str(data_item)
        ref_paper = data_item.get("text", "")
        prompt = prompt_template.format(qa_pair=qa_pair, ref_paper=ref_paper)

        async def _call():
            completion = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                timeout=36000,
            )
            return completion

        completion = await with_retry(_call, max_retries=3)
        result = extract_think_answer(completion.choices[0].message.content)
        try:
            return json_repair.loads(result['answer'])
        except:
            return {"score": 0, "reason": "解析异常"}

# ================= 核心处理函数 =================
async def process_data_item_async(
    global_idx: int,
    data_item: Dict,
    model_configs: List[Dict],
    filter_manager: FilterManager,
    tmp_dir: str,
    enabled_filters: List[str],
    log_file: str,
    log_lock: threading.Lock,
    progress_lock: threading.Lock,
    total_pbar: tqdm
) -> bool:
    try:
        # 随机选择一个模型配置组
        config = random.choice(model_configs)
        client = AsyncOpenAI(api_key=config["api_key"], base_url=config["api_path"])
        model_name = config["model_name"]

        # 依次执行启用的过滤器
        for filter_name in enabled_filters:
            if filter_name == "question_filter":
                result = await filter_manager.llm_question_filter(data_item, client, model_name)
                if result.get("is_filter", False):
                    log_message = (
                        "***********************************\n"
                        f"过滤掉的样本为：{data_item['messages']}\n\n"
                        f"---过滤理由为：{result.get('reason', '无')}\n\n"
                    )
                    with log_lock:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(log_message)
                    print(log_message, end='')
                    with progress_lock:
                        if total_pbar:
                            total_pbar.update(1)
                    return False

            elif filter_name == "answer_filter":
                result = await filter_manager.llm_answer_filter(data_item, client, model_name)
                if result.get("is_filter", False):
                    log_message = (
                        "***********************************\n"
                        f"过滤掉的样本为：{data_item['messages']}\n\n"
                        f"---过滤理由为：{result.get('reason', '无')}\n\n"
                    )
                    with log_lock:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(log_message)
                    print(log_message, end='')
                    with progress_lock:
                        if total_pbar:
                            total_pbar.update(1)
                    return False

            elif filter_name == "qa_score":
                result = await filter_manager.llm_qa_score(data_item, client, model_name)
                score = result.get("score", 0)
                if score <= 5:
                    log_message = (
                        "***********************************\n"
                        f"样本分数为：{score}\n\n"
                        f"---打分样本为：{data_item['messages']}\n\n"
                        f"---过滤理由为：{result.get('reason', '无')}\n\n"
                    )
                    with log_lock:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(log_message)
                    print(log_message, end='')
                    with progress_lock:
                        if total_pbar:
                            total_pbar.update(1)
                    return False
                else:
                    data_item['score'] = score
                    log_message = (
                        "***********************************\n"
                        f"保留的样本分数：{score}\n\n"
                        f"---样本内容为：{data_item['messages']}\n\n"
                        f"---打分理由为：{result.get('reason', '无')}\n\n"
                    )
                    with log_lock:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(log_message)
                    print(log_message, end='')

        # 通过所有过滤器，写入临时文件
        thread_id = threading.get_ident()
        tmp_file_path = os.path.join(tmp_dir, f"tmp_{thread_id}.jsonl")
        os.makedirs(tmp_dir, exist_ok=True)
        with open(tmp_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data_item, ensure_ascii=False) + '\n')

        with progress_lock:
            if total_pbar:
                total_pbar.update(1)
        return True

    except Exception as e:
        error_msg = f"处理数据项 {data_item.get('id')} 时出错: {e}\n"
        with log_lock:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(error_msg)
        print(error_msg, end='')
        with progress_lock:
            if total_pbar:
                total_pbar.update(1)
        return False

# ================= Worker 协程 =================
async def worker(
    worker_id: int,
    task_queue: asyncio.Queue,
    model_configs: List[Dict],
    filter_manager: FilterManager,
    tmp_dir: str,
    enabled_filters: List[str],
    log_file: str,
    log_lock: threading.Lock,
    progress_lock: threading.Lock,
    total_pbar: tqdm
):
    while True:
        try:
            try:
                global_idx, data_item = task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            await process_data_item_async(
                global_idx,
                data_item,
                model_configs,
                filter_manager,
                tmp_dir,
                enabled_filters,
                log_file,
                log_lock,
                progress_lock,
                total_pbar
            )

            task_queue.task_done()

        except Exception as e:
            print(f"Worker {worker_id} 处理时出错: {e}")
            import traceback
            traceback.print_exc()
            task_queue.task_done()

# ================= 主调度器 =================
async def main_task_scheduler(
    data_list: List[Dict],
    model_configs: List[Dict],
    filter_manager: FilterManager,
    tmp_dir: str,
    enabled_filters: List[str],
    log_file: str,
    log_lock: threading.Lock,
    concurrent_workers: int,
    progress_lock: threading.Lock,
    total_pbar: tqdm
):
    task_queue = asyncio.Queue()
    for global_idx, data_item in enumerate(data_list):
        task_queue.put_nowait((global_idx, data_item))

    workers = [
        asyncio.create_task(
            worker(i, task_queue, model_configs, filter_manager, tmp_dir, enabled_filters, log_file, log_lock, progress_lock, total_pbar)
        )
        for i in range(concurrent_workers)
    ]

    await task_queue.join()

    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

# ================= 配置解析工具 =================
def parse_model_configs(config_str: str) -> List[Dict[str, str]]:
    configs = []
    groups = config_str.split(';')
    for group in groups:
        group = group.strip()
        if not group:
            continue
        parts = group.split(',')
        if len(parts) != 3:
            raise ValueError(f"配置格式错误，应为 'api_path,model_name,api_key'，实际: {group}")
        api_path, model_name, api_key = parts
        configs.append({
            "api_path": api_path.strip(),
            "model_name": model_name.strip(),
            "api_key": api_key.strip()
        })
    if not configs:
        raise ValueError("至少需要提供一个配置组")
    return configs

# ================= PromptsLoader 类 =================
class PromptsLoader:
    def __init__(self, prompts_dict: Dict):
        self.prompts = prompts_dict

    def get_prompt(self, category: str, key: str) -> str:
        return self.prompts.get(category, {}).get(key, "")

# ================= 参数解析 =================
def parse_args():
    parser = argparse.ArgumentParser(description="大模型数据过滤系统")

    parser.add_argument("--input_dir", type=str, required=True, help="输入JSONL文件夹路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录路径")
    parser.add_argument("--output_filename", type=str, default="filtered_output.jsonl", help="过滤后输出文件名")
    parser.add_argument("--model_configs", type=str, required=True,
                        help='模型配置组，格式: "api_path,model_name,api_key;api_path2,model_name2,api_key2"')
    parser.add_argument("--concurrent_workers", type=int, default=3, help="总并发数")
    parser.add_argument("--enable_filters", type=str, nargs='*', default=["question_filter", "answer_filter", "qa_score"],
                        choices=["question_filter", "answer_filter", "qa_score"],
                        help="启用的过滤器列表，默认全部启用")
    parser.add_argument("--prompts_path", type=str, default="./prompts_filter.json", help="提示词配置文件路径")

    return parser.parse_args()

# ================= 主程序 =================
if __name__ == "__main__":
    args = parse_args()

    # 加载提示词
    with open(args.prompts_path, "r", encoding="utf-8") as f:
        PROMPTS = json.load(f)
    prompts_loader = PromptsLoader(PROMPTS)
    filter_manager = FilterManager(prompts_loader)

    # 构建路径
    tmp_dir = os.path.join(args.output_dir, "tmp_file")
    log_dir = os.path.join(args.output_dir, "logs")
    output_file_path = os.path.join(args.output_dir, args.output_filename)
    log_file_path = os.path.join(log_dir, "processing.log")

    # 创建目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 初始化日志文件
    open(log_file_path, 'w', encoding='utf-8').close()
    log_lock = threading.Lock()
    progress_lock = threading.Lock()

    # 合并历史临时文件
    print("合并历史临时文件...")
    merge_tmp_files(tmp_dir, output_file_path)

    # 加载数据
    print("正在加载数据...")
    data_list = read_all_jsonl_in_dir(args.input_dir, args.output_dir)
    total_items = len(data_list)

    if total_items == 0:
        print("没有新数据需要处理，程序退出")
        exit(0)

    print(f"找到 {total_items} 条待处理数据")

    # 解析模型配置
    model_configs = parse_model_configs(args.model_configs)

    # 进度条
    total_pbar = tqdm(
        total=total_items,
        desc="过滤进度",
        unit="item",
        dynamic_ncols=True,
        position=0
    )

    # 启动主调度器
    try:
        asyncio.run(
            main_task_scheduler(
                data_list,
                model_configs,
                filter_manager,
                tmp_dir,
                args.enable_filters,
                log_file_path,
                log_lock,
                args.concurrent_workers,
                progress_lock,
                total_pbar
            )
        )
    except KeyboardInterrupt:
        print("用户中断执行")
    except Exception as e:
        print(f"主程序异常: {e}")
        import traceback
        traceback.print_exc()

    total_pbar.close()

    # 合并最终结果
    print("合并临时文件到输出...")
    merge_tmp_files(tmp_dir, output_file_path)

    print(f"处理完成！结果已保存至: {output_file_path}")
