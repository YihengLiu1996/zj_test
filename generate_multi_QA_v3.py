import os
import json
import asyncio
import random
import argparse
import time
import threading
from tqdm import tqdm
from openai import AsyncOpenAI  # ✅ 异步客户端
from transformers import AutoTokenizer
from load_prompt import PromptsLoader
import re
import json_repair

"""
大模型蒸馏数据生成系统（异步 + 重试 + 参数化）
功能说明：
1. 从原始文本生成多轮对话数据（问题-思考-回答）
2. 支持三个独立模型：提问模型(questioner)、回答模型(answerer)、过滤模型(filter)
3. 每个模型使用独立的API密钥池，避免密钥混用
4. 智能控制token长度，确保不超过模型上下文限制
5. 支持固定并发数，任务完成即补位（无长尾空泡）
6. 自动重试失败请求（指数退避）
7. 所有参数通过 argparse 配置

设计原则：
- 模块化：将不同功能拆分为独立函数
- 可配置：关键参数集中管理
- 健壮性：异常处理、重试、进度跟踪
- 可扩展：便于添加新模型或功能
"""

## ================= 配置项（部分移到 argparse） ================= ##
QUES_NUM_PER = 4096  # 每QUES_NUM_PER个token提出一个问题
limit_tokens = 500    # 距离32k长度还有limit_tokens时停止生成

## ================= 工具函数 ================= ##
def append_num_to_filename(file_path: str, num: int) -> str:
    """在文件名后添加序号，用于分片存储"""
    base_path, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)
    return os.path.join(base_path, f"{name}_{num}{ext}")

def get_uuid_jsonl(file_path: str) -> List[str]:
    """从输出目录获取已处理的UUID列表"""
    uuid_list = []
    for root, _, files in os.walk(file_path):
        for file in files:
            if file.endswith(".jsonl"):
                try:
                    uuid = file.split("_")[-1].rsplit(".", 1)[0]
                    uuid_list.append(uuid)
                except (IndexError, ValueError):
                    continue
    return uuid_list

def read_jsonl(file_path: str, output_dir: str) -> List[Dict]:
    """读取JSONL文件，跳过已处理的数据"""
    data_list = []
    processed_uuids = set(get_uuid_jsonl(output_dir))
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get("id", "") not in processed_uuids:
                    data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"解析JSON错误: {e} - 跳过该行")
    return data_list

def write_single_jsonl(file_path, data, num, mode="a"):
    new_file_path = append_num_to_filename(file_path, num)
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    with open(new_file_path, mode, encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
def write_messages_to_jsonl(
    file_path: str, 
    messages: List[Dict], 
    shard_num: int, 
    mode: str = "a", 
    task_type: Optional[str] = None
) -> None:
    """将对话消息写入JSONL文件，支持分片存储"""
    new_file_path = append_num_to_filename(file_path, shard_num)
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    
    data = {"messages": messages}
    if task_type:
        data["task"] = task_type
    
    with open(new_file_path, mode, encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def cal_tokens(text: str) -> int:
    """计算文本的token数量"""
    return len(tokenizer.tokenize(text))

## ================= 核心处理函数 ================= ##
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

def generate_rounds() -> int:
    possible_rounds = [0, 1, 2, 3, 4, 5, 6]
    weights = [0.8] + [0.2 / 6] * 6
    return random.choices(possible_rounds, weights=weights, k=1)[0]

## ================= 异步 + 重试封装 ================= ##
async def with_retry(coro_func, max_retries=3, base_delay=1.0):
    """异步重试装饰器（指数退避）"""
    for attempt in range(max_retries + 1):
        try:
            return await coro_func()
        except Exception as e:
            if attempt == max_retries:
                raise e
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"请求失败，第 {attempt + 1} 次重试，等待 {delay:.2f} 秒...")
            await asyncio.sleep(delay)

## ================= 核心异步处理函数 ================= ##
async def generate_initial_questions_async(
    text: str, 
    language: str, 
    client: AsyncOpenAI,
    model_name: str
) -> Tuple[str, List[str], str, str]:
    text_token = cal_tokens(text)
    ques_num = max(1, int(text_token / QUES_NUM_PER))
    prompt = prompts_loader.get_prompt(language, "generate_1_question").format(
        text=text, 
        ques_num=str(ques_num)
    )

    async def _call():
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            timeout=36000,
        )
        return completion

    completion = await with_retry(_call, max_retries=3)

    result = extract_think_answer(completion.choices[0].message.content)
    questions = [q.strip() for q in result["answer"].strip().split("\n") if q.strip()]

    if not questions:
        raise ValueError("未能生成有效的问题列表，请检查模型输出或提示模板。")
    
    return result["answer"], questions, random.choice(questions), prompt

async def generate_initial_answer_async(
    text: str, 
    question: str, 
    language: str, 
    client: AsyncOpenAI,
    model_name: str
) -> Tuple[str, str, List[Dict]]:
    prompt = prompts_loader.get_prompt(language, "answer_1_question").format(text=text)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question}
    ]

    async def _call():
        completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            timeout=36000,
        )
        return completion

    completion = await with_retry(_call, max_retries=3)

    try:
        if hasattr(completion.choices[0].message, 'reasoning_content') and completion.choices[0].message.reasoning_content:
            result = {
                "think": completion.choices[0].message.reasoning_content,
                "answer": completion.choices[0].message.content
            }
        else:
            result = extract_think_answer(completion.choices[0].message.content)
    except:
        result = extract_think_answer(completion.choices[0].message.content)
    
    return result["answer"].strip(), result["think"].strip(), messages

async def generate_follow_up_answer_async(
    text: str, 
    question: str, 
    language: str, 
    prv_questions: List[str], 
    prv_answers: List[str], 
    client: AsyncOpenAI,
    model_name: str
) -> Tuple[str, str, List[Dict], List[Dict]]:
    prompt = prompts_loader.get_prompt(language, "answer_question").format(text=text)
    messages = [{"role": "system", "content": prompt}]
    
    for q, a in zip(prv_questions, prv_answers):
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    
    messages.append({"role": "user", "content": question})
    current_messages = [{"role": "user", "content": question}]

    async def _call():
        completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            timeout=36000,
        )
        return completion

    completion = await with_retry(_call, max_retries=3)

    result = extract_think_answer(completion.choices[0].message.content)
    return (
        result["answer"].strip(), 
        result["think"].strip(), 
        messages, 
        current_messages
    )

async def generate_follow_up_question_async(
    text: str, 
    prev_round: List[Dict], 
    language: str, 
    client: AsyncOpenAI,
    model_name: str
) -> Tuple[str, str]:
    history = "\n".join([
        f"Q: {r['question']}\nA: {r['answer']}"
        for r in prev_round
    ])
    
    prompt = prompts_loader.get_prompt(
        language, "generate_new_question"
    ).format(text=text, prev_round=history)

    async def _call():
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            timeout=36000,
        )
        return completion

    completion = await with_retry(_call, max_retries=3)

    result = extract_think_answer(completion.choices[0].message.content)
    return result['answer'].strip(), prompt

async def apply_max_filtration_async(
    language: str, 
    think: str, 
    client: AsyncOpenAI,
    model_name: str
) -> Tuple[str, List[Dict]]:
    prompt = prompts_loader.get_prompt(language, "max_user")
    messages = [{
        "role": "user",
        "content": f"{prompt}\n---\n{think}"
    }]

    async def _call():
        completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            timeout=36000,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        return completion

    completion = await with_retry(_call, max_retries=3)

    result = extract_think_answer(completion.choices[0].message.content)
    return result['answer'].strip(), messages

## ================= 数据处理核心（异步版） ================= ##
async def process_data_async(
    global_idx: int,
    data: Dict,
    output_path: str,
    clients: Dict[str, AsyncOpenAI],
    model_names: Dict[str, str],
    progress_lock: threading.Lock,
    total_pbar: Optional[tqdm] = None
) -> None:
    """
    异步处理单个数据项
    """
    try:
        text = data.get("text", "")
        language = data.get("language", "CN").upper()
        shard_num = data.get("id", "0")
        # ===== 构造输出路径 =====
        multiturn_question_path = os.path.join(args.output_dir, "multiturn_question", "multiturn_question.jsonl")
        multiturn_answer_path = os.path.join(args.output_dir, "multiturn_answer", "multiturn_answer.jsonl")
        multiturn_filter_path = os.path.join(args.output_dir, "multiturn_filter", "multiturn_filter.jsonl")
        final_output_path = os.path.join(args.output_dir, "final_output", "sft_data.jsonl")

        # ===== 阶段1: 生成初始问题 =====
        ori_questions, questions, initial_question, init_question_prompt = await generate_initial_questions_async(
            text, language, clients["questioner"], model_names["questioner"]
        )

        for initial_question in questions: 
            # ===== 阶段2: 生成初始回答 =====
            answer0, think0, answer_messages = await generate_initial_answer_async(
                text, initial_question, language, clients["answerer"], model_names["answerer"]
            )
            
            # ===== 阶段3: 优化思考过程 =====
            max_think0, filter_messages = await apply_max_filtration_async(
                language, think0, clients["filter"], model_names["filter"]
            )

            conversation = []
            question_history = []
            filter_history = []

            conversation.append({"role": "user", "content": initial_question})
            conversation.append({
                "role": "assistant",
                "think": max_think0,
                "answer": answer0
            })

            question_history.append({"role": "user", "content": init_question_prompt})
            question_history.append({"role": "assistant", "content": ori_questions})

            filter_history.extend(filter_messages)
            filter_history.append({"role": "assistant", "content": max_think0})

            # 保存中间结果
            write_messages_to_jsonl(
                os.path.join(args.output_dir, "multiturn_question.jsonl"), 
                question_history, 
                shard_num, 
                task_type='max_question'
            )
            write_messages_to_jsonl(
                os.path.join(args.output_dir, "multiturn_filter.jsonl"), 
                filter_history, 
                shard_num, 
                task_type='max_filter'
            )

            total_tokens = (
                cal_tokens(initial_question) +
                cal_tokens(answer0) +
                cal_tokens(max_think0) +
                cal_tokens(text) +
                1000
            )

            # ===== 阶段4: 生成多轮对话 =====
            num_rounds = generate_rounds()
            # print(f"数据 {global_idx} 生成 {num_rounds} 轮对话")

            if num_rounds > 0:
                for round_num in range(1, num_rounds + 1):
                    prev_round = [
                        {
                            "question": m["content"],
                            "answer": f'<think>\n{a["think"]}\n</think>\n\n{a["answer"]}'
                        }
                        for m, a in zip(
                            [m for m in conversation if m["role"] == "user"],
                            [m for m in conversation if m["role"] == "assistant"]
                        )
                    ]

                    next_question, next_question_prompt = await generate_follow_up_question_async(
                        text, prev_round, language, clients["questioner"], model_names["questioner"]
                    )

                    prv_questions = [r["question"] for r in prev_round]
                    prv_answers = [r["answer"] for r in prev_round]
                    
                    answer, think, all_messages, cur_messages = await generate_follow_up_answer_async(
                        text, next_question, language, prv_questions, prv_answers,
                        clients["answerer"], model_names["answerer"]
                    )
                    
                    max_thinking, max_think_messages = await apply_max_filtration_async(
                        language, think, clients["filter"], model_names["filter"]
                    )

                    round_tokens = (
                        cal_tokens(next_question) +
                        cal_tokens(answer) +
                        cal_tokens(max_thinking)
                    )
                    
                    if total_tokens + round_tokens > 25000 - limit_tokens:
                        print(f"数据 {global_idx} 第{round_num}轮超过token限制，终止生成")
                        break
                    
                    total_tokens += round_tokens

                    conversation.append({"role": "user", "content": next_question})
                    conversation.append({
                        "role": "assistant",
                        "think": max_thinking,
                        "answer": answer
                    })

                    question_history.append({"role": "user", "content": next_question_prompt})
                    question_history.append({"role": "assistant", "content": next_question})

                    filter_history.extend(max_think_messages)
                    filter_history.append({"role": "assistant", "content": max_thinking})

                    write_messages_to_jsonl(
                        multiturn_question_path, 
                        question_history, 
                        shard_num, 
                        task_type='max_question'
                    )
                    write_messages_to_jsonl(
                        multiturn_filter_path, 
                        filter_history, 
                        shard_num, 
                        task_type='max_filter'
                    )

                    print(f"数据 {global_idx} 完成第 {round_num} 轮，总tokens: {total_tokens}")

            # ===== 阶段5: 保存最终结果 =====
            data["messages"] = conversation
            write_messages_to_jsonl(
                multiturn_answer_path, 
                answer_messages + [
                    {"role": "assistant", "content": f'<think>\n{think0}\n</think>\n\n{answer0}'}
                ],
                shard_num,
                task_type='plus_answer'
            )
            write_single_jsonl(
                final_output_path,
                data, shard_num, mode="a"
            )

        # ===== 进度更新 =====
        with progress_lock:
            if total_pbar:
                total_pbar.update(1)

    except Exception as e:
        print(f"处理数据 {global_idx} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()

## ================= Worker 协程 ================= ##
async def worker(
    worker_id: int,
    task_queue: asyncio.Queue,
    clients_pool: List[Dict[str, AsyncOpenAI]],
    model_names: Dict[str, str],
    progress_lock: threading.Lock,
    total_pbar: tqdm
):
    while True:
        try:
            try:
                global_idx, data = task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            # 选择客户端（轮询分配）
            client_idx = global_idx % len(clients_pool)
            clients = clients_pool[client_idx]

            await process_data_async(
                global_idx,
                data,
                os.path.join(args.output_dir, "final_output", "sft_data.jsonl"),
                clients,
                model_names,
                progress_lock,
                total_pbar
            )

            task_queue.task_done()

        except Exception as e:
            print(f"Worker {worker_id} 处理时出错: {e}")
            import traceback
            traceback.print_exc()
            task_queue.task_done()

## ================= 主调度器 ================= ##
async def main_task_scheduler(
    data_list: List[Dict],
    args,
    model_names: Dict[str, str],
    progress_lock: threading.Lock,
    total_pbar: tqdm
):
    # 构建客户端池（每个API密钥一个客户端组）
    clients_pool = []
    for i in range(args.concurrent_per_api):
        clients = {
            "questioner": AsyncOpenAI(
                api_key=args.questioner_api_keys[i % len(args.questioner_api_keys)],
                base_url=args.questioner_api_path
            ),
            "answerer": AsyncOpenAI(
                api_key=args.answerer_api_keys[i % len(args.answerer_api_keys)],
                base_url=args.answerer_api_path
            ),
            "filter": AsyncOpenAI(
                api_key=args.filter_api_keys[i % len(args.filter_api_keys)],
                base_url=args.filter_api_path
            ),
        }
        clients_pool.append(clients)

    # 创建任务队列
    task_queue = asyncio.Queue()
    for global_idx, data in enumerate(data_list):
        task_queue.put_nowait((global_idx, data))

    # 启动 Workers
    workers = [
        asyncio.create_task(
            worker(i, task_queue, clients_pool, model_names, progress_lock, total_pbar)
        )
        for i in range(args.concurrent_per_api)
    ]

    await task_queue.join()

    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

## ================= 参数解析 ================= ##
def parse_args():
    parser = argparse.ArgumentParser(description="大模型蒸馏数据生成系统")

    # Questioner 模型配置
    parser.add_argument("--questioner_api_path", type=str, required=True, help="Questioner 模型 API 地址")
    parser.add_argument("--questioner_model_name", type=str, required=True, help="Questioner 模型名称")
    parser.add_argument("--questioner_api_keys", type=str, nargs='+', required=True, help="Questioner API 密钥列表")

    # Answerer 模型配置
    parser.add_argument("--answerer_api_path", type=str, required=True, help="Answerer 模型 API 地址")
    parser.add_argument("--answerer_model_name", type=str, required=True, help="Answerer 模型名称")
    parser.add_argument("--answerer_api_keys", type=str, nargs='+', required=True, help="Answerer API 密钥列表")

    # Filter 模型配置
    parser.add_argument("--filter_api_path", type=str, required=True, help="Filter 模型 API 地址")
    parser.add_argument("--filter_model_name", type=str, required=True, help="Filter 模型名称")
    parser.add_argument("--filter_api_keys", type=str, nargs='+', required=True, help="Filter API 密钥列表")

    # 并发配置
    parser.add_argument("--concurrent_per_api", type=int, default=3, help="每个API的并发数")

    # 输入输出
    parser.add_argument("--input_file_path", type=str, required=True, help="输入JSONL文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录路径")

    return parser.parse_args()

## ================= 初始化 & 主程序 ================= ##
if __name__ == "__main__":
    args = parse_args()

    # 确保输出目录结构
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "final_output"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "multiturn_question"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "multiturn_answer"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "multiturn_filter"), exist_ok=True)

    # 加载分词器
    model_dir = r"/mnt/sizjwb25c1g7/nanhu_lyh/code_tx/think_multi_turn_QA_gen/qwen_model_fold"
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # 加载提示词
    PROMPT_path = r"./prompts.json"
    with open(PROMPT_path, "r", encoding="utf-8") as f:
        PROMPTS = json.load(f)
        global prompts_loader
        prompts_loader = PromptsLoader(PROMPTS)

    # 模型名映射
    model_names = {
        "questioner": args.questioner_model_name,
        "answerer": args.answerer_model_name,
        "filter": args.filter_model_name,
    }

    # 加载数据
    start_time = time.time()
    print("正在加载数据...")
    data_list = read_jsonl(args.input_file_path, args.output_dir)
    total_items = len(data_list)

    if total_items == 0:
        print("没有新数据需要处理，程序退出")
        exit(0)

    print(f"找到 {total_items} 条待处理数据")

    # 进度条
    progress_lock = threading.Lock()
    total_pbar = tqdm(
        total=total_items,
        desc="总进度",
        unit="item",
        dynamic_ncols=True,
        position=0
    )

    # 启动主调度器
    try:
        asyncio.run(
            main_task_scheduler(data_list, args, model_names, progress_lock, total_pbar)
        )
    except KeyboardInterrupt:
        print("用户中断执行")
    except Exception as e:
        print(f"主程序异常: {e}")
        import traceback
        traceback.print_exc()

    total_pbar.close()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n处理完成! 总耗时: {execution_time:.2f}秒")
    print(f"平均速度: {total_items/execution_time:.2f} 条/秒")
    print(f"成功处理 {total_items} 条数据")
