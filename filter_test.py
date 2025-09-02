if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    file_dir = '/mnt/sizjwb25ctet/00-CPT-data/0830/data_process_test/ori_data'
    log_dir = '/mnt/sizjwb25ctet/00-CPT-data/0830/data_process_test/ori_data/log'
    os.makedirs(log_dir, exist_ok=True)
    file_list, file_name = tree(file_dir)
    file_process_list = []
    data_process_list = []
    for i, filepath in enumerate(file_list):
        if os.path.basename(filepath).endswith(".jsonl") and not os.path.basename(filepath).endswith("_F.jsonl"):
            file_process_list.append(filepath)
            data_process_list += read_jsonl(filepath)
     
    print(f"找到 {len(data_process_list)} 个待处理文档，开始并行处理...")
    
    # 准备任务参数
    task_params = [(data_item, file_dir, log_dir) for data_item in data_process_list]
    
    results = []
    
    # 使用简单的进程创建和join，而不是进程池
    processes = []
    result_queue = mp.Queue()
    
    # 分批处理，避免创建太多进程
    batch_size = 10
    for i in range(0, len(task_params), batch_size):
        batch_params = task_params[i:i+batch_size]
        
        # 为每个批次创建一个进程
        p = mp.Process(
            target=process_batch,
            args=(batch_params, result_queue, fasttext_model_dir)
        )
        processes.append(p)
        p.start()
        
        # 等待当前批次完成
        p.join()
        
        # 获取结果
        while not result_queue.empty():
            results.extend(result_queue.get())
        
        print(f"已完成批次 {i//batch_size + 1}/{(len(task_params)+batch_size-1)//batch_size}")
    
    # 保存结果
    filter_pkl = os.path.join(file_dir, "zh_en_filter_result.pkl")
    with open(filter_pkl, 'wb') as f:
        pickle.dump(results, f)
     
    output_file = os.path.join(file_dir, "filter_F.jsonl")
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for result_item in results:
            jsonl_file.write(json.dumps(result_item["data_json"], ensure_ascii=False) + '\n')
     
    print(f"\n处理完成! 保存结果到:")
    print(f"- 过滤结果: {filter_pkl}")
    print(f"- 最终JSONL: {output_file}")
    print(f"共处理 {len(results)} 个文档")

def process_batch(batch_params, result_queue, model_path):
    """处理一个批次的函数"""
    try:
        # 在当前进程加载模型
        model = fasttext.load_model(model_path)
        
        batch_results = []
        for params in batch_params:
            # 创建一个简单的返回值
            result = {
                "false_list": [], 
                "latex_false_list": [], 
                "data_json": params[0]
            }
            batch_results.append(result)
        
        # 将结果放入队列
        result_queue.put(batch_results)
    except Exception as e:
        print(f"处理批次时发生错误: {e}")
        # 返回空结果
        result_queue.put([])
    finally:
        # 确保模型被正确释放
        if 'model' in locals():
            del model
        import gc
        gc.collect()
