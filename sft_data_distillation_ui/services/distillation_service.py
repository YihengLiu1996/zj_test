# services/distillation_service.py

import os
import json
import asyncio
import random
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer
from openai import AsyncOpenAI
from tqdm import tqdm
import json_repair
import re

from config import settings
from services.file_service import FileService
from background_tasks.task_manager import TaskManager

class DistillationService:
    """
    数据蒸馏服务类
    封装蒸馏和过滤的核心逻辑，提供异步任务接口。
    """

    # 从您的代码中提取的常量
    QUES_NUM_PER = 4096  # 每QUES_NUM_PER个token提出一个问题
    LIMIT_TOKENS = 500   # 距离32k长度还有limit_tokens时停止生成

    def __init__(self, user):
        self.user = user
        self.tokenizer = None
        self.prompts_loader = None
        self._initialize()

    def _initialize(self):
        """初始化tokenizer和提示词加载器"""
        try:
            # 加载 tokenizer (路径需要您后续配置)
            tokenizer_path = "./qwen_model_fold"  # 请根据实际情况修改
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        except Exception as e:
            raise RuntimeError(f"无法加载Tokenizer: {e}")

        try:
            # 加载提示词 (从用户目录加载)
            prompts = {}
            for key in [
                "question_prompt_cn", "question_prompt_en",
                "followup_prompt_cn", "followup_prompt_en",
                "answer_prompt_cn", "answer_prompt_en",
                "rewrite_prompt_cn", "rewrite_prompt_en",
                "question_filter_prompt", "answer_filter_prompt", "qa_score_prompt"
            ]:
                filename = settings.PROMPT_FILE_NAMES.get(key, f"{key}.txt")
                file_path = os.path.join(self.user.prompts_path, filename)
                content = FileService.read_file(file_path)
                if content:
                    prompts[key] = content
                else:
                    prompts[key] = f"# {key} - 未配置"

            self.prompts_loader = SimplePromptsLoader(prompts)
        except Exception as e:
            raise RuntimeError(f"无法加载提示词: {e}")

    async def start_distillation_task(
        self,
        task_id: str,
        dataset_path: str,
        model_configs: List[Dict],
        distill_config: Dict,
        output_dir: str
    ) -> bool:
        """
        启动蒸馏任务的主入口。
        这是一个异步函数，应该在后台线程/进程中调用。
        """
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "final_output"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

            # 读取数据
            data_list = self._read_input_data(dataset_path, output_dir)
            if not data_list:
                TaskManager.update_task_status(task_id, "completed", "没有新数据需要处理")
                return True

            total_items = len(data_list)
            TaskManager.update_task_progress(task_id, 0, total_items)

            # 解析蒸馏配置
            question_types = distill_config.get("question_types", 1) # 题型数量，简化处理
            max_rounds = distill_config.get("max_rounds", 3)
            round_probabilities = distill_config.get("round_probabilities", [0.8, 0.2/6]*7)[:7]
            enabled_filters = distill_config.get("enabled_filters", [])

            # 启动蒸馏流程
            for global_idx, data_item in enumerate(data_list):
                try:
                    await self._process_single_data_item(
                        data_item,
                        model_configs,
                        question_types,
                        max_rounds,
                        round_probabilities,
                        enabled_filters,
                        output_dir
                    )
                    # 更新进度
                    TaskManager.update_task_progress(task_id, global_idx + 1, total_items)
                except Exception as e:
                    error_msg = f"处理数据项 {data_item.get('id', 'unknown')} 时出错: {e}"
                    TaskManager.append_task_log(task_id, error_msg)

            TaskManager.update_task_status(task_id, "completed", "蒸馏任务成功完成")
            return True

        except Exception as e:
            error_msg = f"蒸馏任务执行失败: {e}"
            TaskManager.append_task_log(task_id, error_msg)
            TaskManager.update_task_status(task_id, "failed", error_msg)
            return False

    def _read_input_data(self, input_path: str, output_dir: str) -> List[Dict]:
        """读取输入数据，跳过已处理的项"""
        data_list = []
        # 如果是目录，读取所有jsonl文件
        if os.path.isdir(input_path):
            for filename in os.listdir(input_path):
                if filename.endswith(".jsonl"):
                    file_path = os.path.join(input_path, filename)
                    items = FileService.read_jsonl_file(file_path)
                    data_list.extend(items)
        # 如果是文件，直接读取
        elif os.path.isfile(input_path) and input_path.endswith(".jsonl"):
            data_list = FileService.read_jsonl_file(input_path)
        else:
            return []

        # 过滤掉已处理的数据 (根据您的过滤代码逻辑)
        processed_uuids = set(self._get_processed_uuids(output_dir))
        filtered_data_list = [
            item for item in data_list
            if item.get("id", "") not in processed_uuids
        ]
        return filtered_data_list

    def _get_processed_uuids(self, output_dir: str) -> List[str]:
        """获取已处理的UUID列表"""
        uuid_list = []
        final_output_dir = os.path.join(output_dir, "final_output")
        for root, _, files in os.walk(final_output_dir):
            for file in files:
                if file.endswith(".jsonl"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    data = json.loads(line.strip())
                                    uuid_list.append(data.get("id", ""))
                                except json.JSONDecodeError:
                                    continue
                    except Exception:
                        continue
        return uuid_list

    async def _process_single_data_item(
        self,
        data_item: Dict,
        model_configs: List[Dict],
        question_types: int,
        max_rounds: int,
        round_probabilities: List[float],
        enabled_filters: List[str],
        output_dir: str
    ) -> None:
        """
        处理单个数据项的核心逻辑。
        整合了蒸馏和过滤的完整流程。
        """
        text = data_item.get("text", "")
        language = data_item.get("language", "CN").upper()
        item_id = data_item.get("id", str(int(time.time())))

        # 根据语言选择提示词键
        lang_key = "cn" if language == "CN" else "en"

        # 如果用户已添加问题，则跳过问题生成和过滤
        has_user_questions = (
            "messages" in data_item and
            len(data_item["messages"]) > 0 and
            data_item["messages"][0].get("role") == "user"
        )

        # ===== 阶段1: 生成初始问题 (如果需要) =====
        if not has_user_questions:
            initial_questions = await self._generate_initial_questions(
                text, language, model_configs, lang_key
            )
            # 应用问题过滤器 (如果启用)
            if "question_filter" in enabled_filters:
                filtered_questions = []
                for q in initial_questions:
                    is_valid, reason = await self._filter_question(q, model_configs)
                    if is_valid:
                        filtered_questions.append(q)
                    else:
                        # 记录被过滤的问题
                        log_msg = f"问题被过滤: '{q}' - 理由: {reason}"
                        # 这里可以写入日志文件，或通过TaskManager记录
                initial_questions = filtered_questions if filtered_questions else [initial_questions[0]] # 保底
        else:
            # 使用用户已添加的问题
            initial_questions = [
                msg["content"] for msg in data_item["messages"]
                if msg.get("role") == "user"
            ]

        # 处理每个初始问题
        for initial_question in initial_questions:
            conversation = []
            # 添加用户问题
            conversation.append({"role": "user", "content": initial_question})

            # ===== 阶段2: 生成初始回答 =====
            answer, think = await self._generate_initial_answer(
                text, initial_question, language, model_configs, lang_key
            )

            # ===== 阶段3: 优化思考过程 (改写) =====
            max_think = await self._rewrite_thinking(think, language, model_configs, lang_key)

            # 添加助手回答
            conversation.append({
                "role": "assistant",
                "think": max_think,
                "answer": answer
            })

            # ===== 阶段4: 生成多轮对话 =====
            if "multi_turn" in distill_config.get("answer_mode", "single"): # 假设有此配置
                num_rounds = self._generate_rounds(round_probabilities)
                num_rounds = min(num_rounds, max_rounds)

                prv_questions = [initial_question]
                prv_answers = [f'<think>{max_think}</think>{answer}']

                for round_num in range(1, num_rounds + 1):
                    # 生成追问
                    followup_question = await self._generate_followup_question(
                        text, prv_questions, prv_answers, language, model_configs, lang_key
                    )
                    # 生成追问回答
                    followup_answer, followup_think = await self._generate_followup_answer(
                        text, followup_question, prv_questions, prv_answers, language, model_configs, lang_key
                    )
                    # 优化追问的思考过程
                    max_followup_think = await self._rewrite_thinking(followup_think, language, model_configs, lang_key)

                    # 添加到对话
                    conversation.append({"role": "user", "content": followup_question})
                    conversation.append({
                        "role": "assistant",
                        "think": max_followup_think,
                        "answer": followup_answer
                    })

                    # 更新历史
                    prv_questions.append(followup_question)
                    prv_answers.append(f'<think>{max_followup_think}</think>{followup_answer}')

            # ===== 阶段5: 应用回答过滤器 (如果启用) =====
            if "answer_filter" in enabled_filters:
                is_valid, reason = await self._filter_answer(conversation, model_configs)
                if not is_valid:
                    log_msg = f"问答对被过滤 - 理由: {reason}"
                    return # 跳过保存

            # ===== 阶段6: 应用质量打分 (如果启用) =====
            score = 10 # 默认满分
            if "qa_score" in enabled_filters:
                score, reason = await self._score_qa_pair(text, conversation, model_configs)
                if score <= 5: # 低分过滤
                    log_msg = f"问答对分数过低 ({score}) 被过滤 - 理由: {reason}"
                    return # 跳过保存

            # ===== 阶段7: 保存最终结果 =====
            final_data_item = data_item.copy()
            final_data_item["messages"] = conversation
            if "qa_score" in enabled_filters:
                final_data_item["score"] = score

            final_output_path = os.path.join(output_dir, "final_output", f"sft_data_{item_id}.jsonl")
            success = FileService.write_jsonl_file(final_output_path, [final_data_item])
            if not success:
                raise RuntimeError(f"无法保存结果到 {final_output_path}")

    # --- 以下是核心的异步生成和过滤函数 ---

    async def _generate_initial_questions(self, text: str, language: str, model_configs: List[Dict], lang_key: str) -> List[str]:
        """生成初始问题"""
        text_token = len(self.tokenizer.encode(text, add_special_tokens=False))
        ques_num = max(1, int(text_token / self.QUES_NUM_PER))

        prompt_key = f"question_prompt_{lang_key}"
        prompt_template = self.prompts_loader.get_prompt(prompt_key)
        prompt = prompt_template.replace("{text}", text).replace("{ques_num}", str(ques_num))

        client, model_name = self._get_client_and_model(model_configs, "thinking")
        result = await self._call_llm_with_retry(client, model_name, prompt)
        extracted = self._extract_think_answer(result)
        questions = [q.strip() for q in extracted["answer"].strip().split("\n") if q.strip()]
        return questions if questions else ["请总结上述文本的主要内容。"] # 保底问题

    async def _generate_initial_answer(self, text: str, question: str, language: str, model_configs: List[Dict], lang_key: str) -> Tuple[str, str]:
        """生成初始回答"""
        prompt_key = f"answer_prompt_{lang_key}"
        prompt_template = self.prompts_loader.get_prompt(prompt_key)
        prompt = prompt_template.replace("{text}", text)

        client, model_name = self._get_client_and_model(model_configs, "thinking")
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
        result = await self._call_llm_with_retry(client, model_name, messages)
        extracted = self._extract_think_answer(result)
        return extracted["answer"].strip(), extracted["think"].strip()

    async def _rewrite_thinking(self, think: str, language: str, model_configs: List[Dict], lang_key: str) -> str:
        """改写思考过程"""
        prompt_key = f"rewrite_prompt_{lang_key}"
        prompt_template = self.prompts_loader.get_prompt(prompt_key)
        prompt = f"{prompt_template}\n---\n{think}"

        client, model_name = self._get_client_and_model(model_configs, "mix")
        result = await self._call_llm_with_retry(client, model_name, prompt)
        extracted = self._extract_think_answer(result)
        return extracted['answer'].strip()

    async def _generate_followup_question(self, text: str, prv_questions: List[str], prv_answers: List[str], language: str, model_configs: List[Dict], lang_key: str) -> str:
        """生成追问"""
        history = "\n".join([f"Q: {q}\nA: {a}" for q, a in zip(prv_questions, prv_answers)])
        prompt_key = f"followup_prompt_{lang_key}"
        prompt_template = self.prompts_loader.get_prompt(prompt_key)
        prompt = prompt_template.replace("{text}", text).replace("{prev_round}", history)

        client, model_name = self._get_client_and_model(model_configs, "thinking")
        result = await self._call_llm_with_retry(client, model_name, prompt)
        extracted = self._extract_think_answer(result)
        return extracted['answer'].strip()

    async def _generate_followup_answer(self, text: str, question: str, prv_questions: List[str], prv_answers: List[str], language: str, model_configs: List[Dict], lang_key: str) -> Tuple[str, str]:
        """生成追问回答"""
        prompt_key = f"answer_prompt_{lang_key}"
        prompt_template = self.prompts_loader.get_prompt(prompt_key)
        prompt = prompt_template.replace("{text}", text)

        client, model_name = self._get_client_and_model(model_configs, "thinking")
        messages = [{"role": "system", "content": prompt}]
        for q, a in zip(prv_questions, prv_answers):
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": question})

        result = await self._call_llm_with_retry(client, model_name, messages)
        extracted = self._extract_think_answer(result)
        return extracted["answer"].strip(), extracted["think"].strip()

    async def _filter_question(self, question: str, model_configs: List[Dict]) -> Tuple[bool, str]:
        """过滤问题"""
        prompt_template = self.prompts_loader.get_prompt("question_filter_prompt")
        prompt = prompt_template.replace("{question}", question)

        client, model_name = self._get_client_and_model(model_configs, "thinking")
        result = await self._call_llm_with_retry(client, model_name, prompt)
        extracted = self._extract_think_answer(result)
        try:
            filter_result = json_repair.loads(extracted['answer'])
            return not filter_result.get("is_filter", False), filter_result.get("reason", "无")
        except:
            return True, "解析异常，默认通过"

    async def _filter_answer(self, conversation: List[Dict], model_configs: List[Dict]) -> Tuple[bool, str]:
        """过滤回答"""
        qa_pair_str = self._convert_msg_to_str(conversation)
        prompt_template = self.prompts_loader.get_prompt("answer_filter_prompt")
        prompt = prompt_template.replace("{qa_pair}", qa_pair_str)

        client, model_name = self._get_client_and_model(model_configs, "thinking")
        result = await self._call_llm_with_retry(client, model_name, prompt)
        extracted = self._extract_think_answer(result)
        try:
            filter_result = json_repair.loads(extracted['answer'])
            return not filter_result.get("is_filter", False), filter_result.get("reason", "无")
        except:
            return True, "解析异常，默认通过"

    async def _score_qa_pair(self, ref_text: str, conversation: List[Dict], model_configs: List[Dict]) -> Tuple[int, str]:
        """为问答对打分"""
        qa_pair_str = self._convert_msg_to_str(conversation)
        prompt_template = self.prompts_loader.get_prompt("qa_score_prompt")
        prompt = prompt_template.replace("{qa_pair}", qa_pair_str).replace("{ref_paper}", ref_text)

        client, model_name = self._get_client_and_model(model_configs, "thinking")
        result = await self._call_llm_with_retry(client, model_name, prompt)
        extracted = self._extract_think_answer(result)
        try:
            score_result = json_repair.loads(extracted['answer'])
            return score_result.get("score", 10), score_result.get("reason", "无")
        except:
            return 10, "解析异常，默认满分"

    def _generate_rounds(self, weights: List[float]) -> int:
        """根据概率生成多轮对话的轮数"""
        possible_rounds = [0, 1, 2, 3, 4, 5, 6]
        return random.choices(possible_rounds, weights=weights, k=1)[0]

    def _convert_msg_to_str(self, messages: List[Dict]) -> str:
        """将消息列表转换为字符串"""
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

    def _extract_think_answer(self, text: str) -> Dict[str, str]:
        """从模型输出中提取思考和答案"""
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

    async def _call_llm_with_retry(self, client: AsyncOpenAI, model_name: str, messages_or_prompt, max_retries=3, base_delay=1.0) -> str:
        """带重试的LLM调用"""
        if isinstance(messages_or_prompt, str):
            messages = [{"role": "user", "content": messages_or_prompt}]
        else:
            messages = messages_or_prompt

        for attempt in range(max_retries + 1):
            try:
                completion = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    timeout=1200,
                )
                return completion.choices[0].message.content
            except Exception as e:
                if attempt == max_retries:
                    raise e
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)

    def _get_client_and_model(self, model_configs: List[Dict], required_type: str) -> Tuple[AsyncOpenAI, str]:
        """根据模型类型获取一个客户端和模型名"""
        # 筛选出符合类型的模型
        candidates = [cfg for cfg in model_configs if cfg.get("model_type") == required_type]
        if not candidates:
            # 如果没有指定类型的，降级使用thinking
            candidates = [cfg for cfg in model_configs if cfg.get("model_type") == "thinking"]
            if not candidates:
                # 如果还没有，就随便选一个
                candidates = model_configs

        if not candidates:
            raise ValueError("没有可用的大模型配置")

        # 随机选择一个
        config = random.choice(candidates)
        client = AsyncOpenAI(api_key=config["api_key"], base_url=config["api_path"])
        return client, config["model_name"]

class SimplePromptsLoader:
    """一个简单的提示词加载器，用于适配蒸馏服务"""
    def __init__(self, prompts_dict: Dict):
        self.prompts = prompts_dict

    def get_prompt(self, key: str) -> str:
        return self.prompts.get(key, f"# {key} - 未找到")
