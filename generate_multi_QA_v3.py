import os
import json
import asyncio
import random
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
from load_prompt import PromptsLoader
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from typing import List, Dict, Tuple, Any, Optional
import re
import json_repair
 
"""
大模型蒸馏数据生成系统
功能说明：
1. 从原始文本生成多轮对话数据（问题-思考-回答）
2. 支持三个独立模型：提问模型(questioner)、回答模型(answerer)、过滤模型(filter)
3. 每个模型使用独立的API密钥池，避免密钥混用
4. 智能控制token长度，确保不超过模型上下文限制
5. 支持多线程并发处理，优化进度条显示
 
设计原则：
- 模块化：将不同功能拆分为独立函数
- 可配置：关键参数集中管理
- 健壮性：异常处理和进度跟踪
- 可扩展：便于添加新模型或功能
"""
 
## ================= 配置项 ================= ##
# 基础配置
QUES_NUM_PER = 4096  # 每QUES_NUM_PER个token提出一个问题，用于计算问题总数量
limit_tokens = 500    # 距离32k长度还有limit_tokens时停止生成
 
# 模型API配置 - 每个模型独立配置
MODEL_CONFIGS = {
    "questioner": {
        "api_path": "http://jb-aionlineinferenceservice-145035963735239680-25000-nhss-job.z2120.nhss.zhejianglab.com:31080/v1",
        "model_name": "Qwen3-235B-thinking-2507",
        "api_keys": [
            '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', 
            '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM',
            '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM',
        ]
    },
    
    "answerer": {
        "api_path": "http://jb-aionlineinferenceservice-145035963735239680-25000-nhss-job.z2120.nhss.zhejianglab.com:31080/v1",
        "model_name": "Qwen3-235B-thinking-2507",
        "api_keys": [
            '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', 
            '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM',
            '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM', '1K_cnNCZ1KQBTSW2oNIZ-MjdUadAp8n5_p8QifIllxM',
        ]
    },
    "filter": {
        "api_path": "http://jb-aionlineinferenceservice-145204221050380608-30000-nhss-job.z2120.nhss.zhejianglab.com:31080/v1",
        "model_name": "Qwen3-235B-A22B",
        "api_keys": [
            'sk-svmQOvFOqwdcxJceuxVPlZD0lMEQEaPXPAPfS7eDNTwhZNGv', 
            'sk-svmQOvFOqwdcxJceuxVPlZD0lMEQEaPXPAPfS7eDNTwhZNGv', 'sk-svmQOvFOqwdcxJceuxVPlZD0lMEQEaPXPAPfS7eDNTwhZNGv', 'sk-svmQOvFOqwdcxJceuxVPlZD0lMEQEaPXPAPfS7eDNTwhZNGv', 'sk-svmQOvFOqwdcxJceuxVPlZD0lMEQEaPXPAPfS7eDNTwhZNGv',
            'sk-svmQOvFOqwdcxJceuxVPlZD0lMEQEaPXPAPfS7eDNTwhZNGv', 'sk-svmQOvFOqwdcxJceuxVPlZD0lMEQEaPXPAPfS7eDNTwhZNGv', 'sk-svmQOvFOqwdcxJceuxVPlZD0lMEQEaPXPAPfS7eDNTwhZNGv', 'sk-svmQOvFOqwdcxJceuxVPlZD0lMEQEaPXPAPfS7eDNTwhZNGv', 'sk-svmQOvFOqwdcxJceuxVPlZD0lMEQEaPXPAPfS7eDNTwhZNGv',
        ]
    },
}
 
# 每个API的最大并发数
MAX_CONCURRENT_PER_API = 3
 
# 扩展API密钥池：每个密钥支持MAX_CONCURRENT_PER_API个并发连接
for config in MODEL_CONFIGS.values():
    config["api_keys"] = config["api_keys"] * MAX_CONCURRENT_PER_API
 
# 模型和分词器配置
model_dir = r"/mnt/sizjwb25c1g7/nanhu_lyh/code_tx/think_multi_turn_QA_gen/qwen_model_fold"
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
 
# 提示词配置
PROMPT_path = r"./prompts.json"
with open(PROMPT_path, "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)
    prompts_loader = PromptsLoader(PROMPTS)
 
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
                    # 提取文件名中的UUID部分（格式：prefix_UUID.jsonl）
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
                # 跳过已处理的数据
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
    # 正则表达式匹配第一个<think>标签及其内容
    pattern = r'(.*?)(?:<think>(.*?)</think>)(.*)'
    match = re.search(pattern, text, re.DOTALL)
 
    if not match:
        # 如果没有匹配到<think>标签，整个文本作为answer
        if '</think>' in text:
            text_list = text.split('</think>')
            answer_content = "".join(text_list[1:])
            return {"think": text_list[0].replace("<think>", "").strip(), "answer": answer_content.strip()}
        else:
            return {"think": "", "answer": text.replace("<think>", "").strip()}
 
    # 提取各部分内容
    before_think = match.group(1) or ""
    think_content = match.group(2) or ""
    after_think = match.group(3) or ""
 
    # 合并标签外的内容（before_think + after_think）
    answer_content = after_think.strip()
 
    return {
        "think": think_content.strip(),
        "answer": answer_content
    }
 
def generate_rounds() -> int:
    """
    随机生成对话轮次（0-6轮）
    0轮概率80%，1-6轮均匀分配20%概率
    """
    possible_rounds = [0, 1, 2, 3, 4, 5, 6]
    weights = [0.8] + [0.2 / 6] * 6
    return random.choices(possible_rounds, weights=weights, k=1)[0]
 
def generate_initial_questions(
    text: str, 
    language: str, 
    api_key: str
) -> Tuple[str, List[str], str, str]:
    """
    生成初始问题列表（5-10个），并随机选择一个问题
    返回: (原始问题列表, 问题列表, 选中的问题, 生成提示)
    """
    text_token = cal_tokens(text)
    ques_num = max(1, int(text_token / QUES_NUM_PER))
    prompt = prompts_loader.get_prompt(language, "generate_1_question").format(
        text=text, 
        ques_num=str(ques_num)
    )
 
    client = OpenAI(
        api_key=api_key,
        base_url=MODEL_CONFIGS["questioner"]["api_path"]
    )
 
    completion = client.chat.completions.create(
        model=MODEL_CONFIGS["questioner"]["model_name"],
        messages=[{"role": "user", "content": prompt}],
        timeout=36000,
    )
     
    result = extract_think_answer(completion.choices[0].message.content)
    questions = [q.strip() for q in result["answer"].strip().split("\n") if q.strip()]
 
    if not questions:
        raise ValueError("未能生成有效的问题列表，请检查模型输出或提示模板。")
     
    return result["answer"], questions, random.choice(questions), prompt
 
def generate_initial_answer(
    text: str, 
    question: str, 
    language: str, 
    api_key: str
) -> Tuple[str, str, List[Dict]]:
    """
    生成第0轮回答（初始回答）
    返回: (回答内容, 思考过程, 消息历史)
    """
    prompt = prompts_loader.get_prompt(language, "answer_1_question").format(text=text)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question}
    ]
 
    client = OpenAI(
        api_key=api_key,
        base_url=MODEL_CONFIGS["answerer"]["api_path"]
    )
     
    completion = client.chat.completions.create(
        model=MODEL_CONFIGS["answerer"]["model_name"],
        messages=messages,
        timeout=36000,
    )
    try:
        if completion.choices[0].message.reasoning_content:
            result = {"think": completion.choices[0].message.reasoning_content, "answer": completion.choices[0].message.content}
        else:
            result = extract_think_answer(completion.choices[0].message.content)
    except:
        result = extract_think_answer(completion.choices[0].message.content)
     
    return result["answer"].strip(), result["think"].strip(), messages
 
def generate_follow_up_answer(
    text: str, 
    question: str, 
    language: str, 
    prv_questions: List[str], 
    prv_answers: List[str], 
    api_key: str
) -> Tuple[str, str, List[Dict], List[Dict]]:
    """
    生成后续轮次的回答
    返回: (回答内容, 思考过程, 完整消息历史, 当前轮次消息)
    """
    prompt = prompts_loader.get_prompt(language, "answer_question").format(text=text)
    messages = [{"role": "system", "content": prompt}]
     
    # 添加历史对话
    for q, a in zip(prv_questions, prv_answers):
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
     
    messages.append({"role": "user", "content": question})
    current_messages = [{"role": "user", "content": question}]
 
    client = OpenAI(
        api_key=api_key,
        base_url=MODEL_CONFIGS["answerer"]["api_path"]
    )
     
    completion = client.chat.completions.create(
        model=MODEL_CONFIGS["answerer"]["model_name"],
        messages=messages,
        timeout=36000,
    )
 
    result = extract_think_answer(completion.choices[0].message.content)
    return (
        result["answer"].strip(), 
        result["think"].strip(), 
        messages, 
        current_messages
    )
 
def generate_follow_up_question(
    text: str, 
    prev_round: List[Dict], 
    language: str, 
    api_key: str
) -> Tuple[str, str]:
    """
    生成后续问题
    返回: (新问题, 生成提示)
    """
    history = "\n".join([
        f"Q: {r['question']}\nA: {r['answer']}"
        for r in prev_round
    ])
     
    prompt = prompts_loader.get_prompt(
        language, "generate_new_question"
    ).format(text=text, prev_round=history)
 
    client = OpenAI(
        api_key=api_key,
        base_url=MODEL_CONFIGS["questioner"]["api_path"]
    )
 
    completion = client.chat.completions.create(
        model=MODEL_CONFIGS["questioner"]["model_name"],
        messages=[{"role": "user", "content": prompt}],
        timeout=36000,
    )
 
    result = extract_think_answer(completion.choices[0].message.content)
    return result['answer'].strip(), prompt
 
def apply_max_filtration(
    language: str, 
    think: str, 
    api_key: str
) -> Tuple[str, List[Dict]]:
    """
    应用最大过滤（优化思考过程）
    返回: (优化后的思考, 消息历史)
    """
    prompt = prompts_loader.get_prompt(language, "max_user")
    messages = [{
        "role": "user",
        "content": f"{prompt}\n---\n{think}"
    }]
 
    client = OpenAI(
        api_key=api_key,
        base_url=MODEL_CONFIGS["filter"]["api_path"]
    )
     
    completion = client.chat.completions.create(
        model=MODEL_CONFIGS["filter"]["model_name"],
        messages=messages,
        timeout=36000,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
 
    result = extract_think_answer(completion.choices[0].message.content)
    # result = completion.choices[0].message.content
    return result['answer'].strip(), messages
 
## ================= 数据处理核心 ================= ##
def process_data(
    global_idx: int,
    data: Dict,
    output_path: str,
    api_keys: Dict[str, str],
    progress_lock: threading.Lock,
    total_pbar: Optional[tqdm] = None,
    chunk_pbar: Optional[tqdm] = None
) -> None:
    """
    处理单个数据项，生成多轮对话
    参数:
        global_idx: 全局索引（用于API密钥分配）
        data: 原始数据
        output_path: 输出文件路径
        api_keys: 三个模型的API密钥字典
        progress_lock: 进度条锁
        total_pbar: 总进度条
        chunk_pbar: 分块进度条
    """
    try:
        text = data.get("text", "")
        language = data.get("language", "CN").upper()
        # shard_num = global_idx % 100  # 按100分片存储
        shard_num = data.get("id", "0")
 
        # ===== 阶段1: 生成初始问题 =====
        ori_questions, questions, initial_question, init_question_prompt = (
            generate_initial_questions(
                text, language, api_keys["questioner"]
            )
        )
 
        # 对每个初始问题生成对话（实际只处理第一个，但保留扩展性）
        for initial_question in questions:  # 通常只取第一个问题
            # ===== 阶段2: 生成初始回答 =====
            answer0, think0, answer_messages = generate_initial_answer(
                text, initial_question, language, api_keys["answerer"]
            )
             
            # ===== 阶段3: 优化思考过程 =====
            max_think0, filter_messages = apply_max_filtration(
                language, think0, api_keys["filter"]
            )
 
            # 初始化消息存储
            conversation = []
            question_history = []
            filter_history = []
 
            # 添加第0轮对话
            conversation.append({"role": "user", "content": initial_question})
            conversation.append({
                "role": "assistant",
                "think": max_think0,
                "answer": answer0
            })
 
            # 记录问题生成过程
            question_history.append({"role": "user", "content": init_question_prompt})
            question_history.append({"role": "assistant", "content": ori_questions})
 
            # 记录过滤过程
            filter_history.extend(filter_messages)
            filter_history.append({"role": "assistant", "content": max_think0})
 
            # 保存中间结果
            write_messages_to_jsonl(
                multiturn_question, 
                question_history, 
                shard_num, 
                task_type='max_question'
            )
            write_messages_to_jsonl(
                multiturn_filter, 
                filter_history, 
                shard_num, 
                task_type='max_filter'
            )
 
            # 计算当前token使用量
            total_tokens = (
                cal_tokens(initial_question) +
                cal_tokens(answer0) +
                cal_tokens(max_think0) +
                cal_tokens(text) +
                1000
            )
 
            # ===== 阶段4: 生成多轮对话 =====
            num_rounds = generate_rounds()
            print(f"数据 {global_idx} 生成 {num_rounds} 轮对话")
 
            if num_rounds > 0:
                for round_num in range(1, num_rounds + 1):  # 修正：包含num_rounds轮
                    # 构建历史对话
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
 
                    # 生成新问题
                    next_question, next_question_prompt = generate_follow_up_question(
                        text, prev_round, language, api_keys["questioner"]
                    )
 
                    # 生成回答
                    prv_questions = [r["question"] for r in prev_round]
                    prv_answers = [r["answer"] for r in prev_round]
                     
                    answer, think, all_messages, cur_messages = (
                        generate_follow_up_answer(
                            text, 
                            next_question, 
                            language, 
                            prv_questions, 
                            prv_answers, 
                            api_keys["answerer"]
                        )
                    )
                     
                    # 优化思考过程
                    max_thinking, max_think_messages = apply_max_filtration(
                        language, think, api_keys["filter"]
                    )
 
                    # 检查token限制
                    round_tokens = (
                        cal_tokens(next_question) +
                        cal_tokens(answer) +
                        cal_tokens(max_thinking)
                    )
                     
                    if total_tokens + round_tokens > 25000 - limit_tokens:
                        print(f"数据 {global_idx} 第{round_num}轮超过token限制，终止生成")
                        break
                     
                    total_tokens += round_tokens
 
                    # 更新对话历史
                    conversation.append({"role": "user", "content": next_question})
                    conversation.append({
                        "role": "assistant",
                        "think": max_thinking,
                        "answer": answer
                    })
 
                    # 更新问题历史
                    question_history.append({"role": "user", "content": next_question_prompt})
                    question_history.append({"role": "assistant", "content": next_question})
 
                    # 更新过滤历史
                    filter_history.extend(max_think_messages)
                    filter_history.append({"role": "assistant", "content": max_thinking})
 
                    # 保存中间结果
                    write_messages_to_jsonl(
                        multiturn_question, 
                        question_history, 
                        shard_num, 
                        task_type='max_question'
                    )
                    write_messages_to_jsonl(
                        multiturn_filter, 
                        filter_history, 
                        shard_num, 
                        task_type='max_filter'
                    )
 
                    print(f"数据 {global_idx} 完成第 {round_num} 轮，总tokens: {total_tokens}")
 
            # ===== 阶段5: 保存最终结果 =====
            data["messages"] = conversation
            write_messages_to_jsonl(
                multiturn_answer, 
                answer_messages + [
                    {"role": "assistant", "content": f'<think>\n{think0}\n</think>\n\n{answer0}'}
                ],
                shard_num,
                task_type='plus_answer'
            )
            write_single_jsonl(output_file_path, data, shard_num, mode="a")
 
        # ===== 进度更新 =====
        with progress_lock:
            if total_pbar:
                total_pbar.update(1)
            if chunk_pbar:
                chunk_pbar.update(1)
 
    except Exception as e:
        print(f"处理数据 {global_idx} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
 
## ================= 并发处理 ================= ##
async def process_chunk(
    chunk: List[Tuple[int, Dict]],
    output_path: str,
    progress_lock: threading.Lock,
    total_pbar: Optional[tqdm] = None,
    chunk_pbar: Optional[tqdm] = None
) -> None:
    """
    处理一个数据分块
    参数:
        chunk: (全局索引, 数据) 元组列表
        output_path: 输出路径
        progress_lock: 进度条锁
        total_pbar: 总进度条
        chunk_pbar: 分块进度条
    """
    loop = asyncio.get_running_loop()
     
    # 创建线程池（大小等于分块大小）
    with ThreadPoolExecutor(max_workers=len(chunk)) as executor:
        futures = []
        for global_idx, data in chunk:
            # 为每个模型分配API密钥
            api_keys = {
                model: config["api_keys"][global_idx % len(config["api_keys"])]
                for model, config in MODEL_CONFIGS.items()
            }
             
            # 提交任务
            future = loop.run_in_executor(
                executor,
                process_data,
                global_idx,
                data,
                output_path,
                api_keys,
                progress_lock,
                total_pbar,
                chunk_pbar
            )
            futures.append(future)
         
        # 等待所有任务完成
        await asyncio.gather(*futures)
 
## ================= 主程序 ================= ##
if __name__ == "__main__":
    # 文件路径配置
    input_file_path = '/mnt/sizjwb25c1g7/nanhu_lyh/code_tx/think_multi_turn_QA_gen/ori_data/data_system2_deal.jsonl'
    output_file_path = "/mnt/sizjwb25c1g7/nanhu_lyh/code_tx/think_multi_turn_QA_gen/result/sft_data_system2/final_output/sft_data.jsonl"
    output_file_dir = "/mnt/sizjwb25c1g7/nanhu_lyh/code_tx/think_multi_turn_QA_gen/result/sft_data_system2/final_output"
    multiturn_question = "/mnt/sizjwb25c1g7/nanhu_lyh/code_tx/think_multi_turn_QA_gen/result/sft_data_system2/multiturn_question/multiturn_question.jsonl"
    multiturn_answer = "/mnt/sizjwb25c1g7/nanhu_lyh/code_tx/think_multi_turn_QA_gen/result/sft_data_system2/multiturn_answer/multiturn_answer.jsonl"
    multiturn_filter = "/mnt/sizjwb25c1g7/nanhu_lyh/code_tx/think_multi_turn_QA_gen/result/sft_data_system2/multiturn_filter/multiturn_filter.jsonl"
 
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(multiturn_question), exist_ok=True)
    os.makedirs(os.path.dirname(multiturn_answer), exist_ok=True)
    os.makedirs(os.path.dirname(multiturn_filter), exist_ok=True)
 
    # ===== 1. 加载数据 =====
    start_time = time.time()
    print("正在加载数据...")
    data_list = read_jsonl(input_file_path, output_file_dir)
    total_items = len(data_list)
     
    if total_items == 0:
        print("没有新数据需要处理，程序退出")
        exit(0)
     
    print(f"找到 {total_items} 条待处理数据")
 
    # ===== 2. 创建进度管理 =====
    progress_lock = threading.Lock()
    total_pbar = tqdm(
        total=total_items, 
        desc="总进度", 
        unit="item",
        dynamic_ncols=True,
        position=0
    )
 
    # ===== 3. 分块处理数据 =====
    # 计算分块大小（使用最小API密钥池长度）
    min_api_pool_size = min(
        len(config["api_keys"]) 
        for config in MODEL_CONFIGS.values()
    )
    chunks = [
        list(enumerate(data_list))[i:i + min_api_pool_size]
        for i in range(0, total_items, min_api_pool_size)
    ]
 
    print(f"数据分为 {len(chunks)} 个处理块，每块最多 {min_api_pool_size} 条")
 
    # 处理每个分块
    for chunk_idx, chunk in enumerate(chunks):
        # 创建分块进度条
        chunk_pbar = tqdm(
            total=len(chunk),
            desc=f"块 {chunk_idx+1}/{len(chunks)}",
            unit="item",
            dynamic_ncols=True,
            position=1,
            leave=False
        )
         
        try:
            # 处理当前分块
            asyncio.run(process_chunk(
                chunk,
                output_file_path,
                progress_lock,
                total_pbar,
                chunk_pbar
            ))
        except Exception as e:
            print(f"处理块 {chunk_idx} 时出错: {str(e)}")
        finally:
            # 确保进度条关闭
            chunk_pbar.close()
 
    # ===== 4. 清理资源 =====
    total_pbar.close()
     
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n处理完成! 总耗时: {execution_time:.2f}秒")
    print(f"平均速度: {total_items/execution_time:.2f} 条/秒")
    print(f"成功处理 {total_items} 条数据")
