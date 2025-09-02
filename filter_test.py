if __name__ == "__main__":
    # 设置多进程启动方法为spawn
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
    
    # 计算合理的进程数
    cpu_count = mp.cpu_count()
    memory_available = psutil.virtual_memory().available
    model_size = 100 * 1024 * 1024  # 100MB
    max_workers = min(cpu_count, int(memory_available / (model_size * 1.5)))
    
    if max_workers < 1:
        max_workers = 1
    
    print(f"使用 {max_workers} 个进程进行并行处理 (CPU核心数: {cpu_count}, 可用内存: {memory_available / (1024**3):.2f}GB)")
    
    # 准备任务参数
    task_params = [(data_item, file_dir, log_dir) for data_item in data_process_list]
    
    results = []
    
    # 使用进程池，但添加超时和错误处理机制
    with mp.Pool(
        processes=max_workers,
        initializer=init_process,
        initargs=(fasttext_model_dir,),
        maxtasksperchild=10  # 每个子进程最多处理10个任务后重启，避免资源泄漏
    ) as pool:
        try:
            # 使用imap_unordered，它比imap更高效
            with tqdm(total=len(task_params), desc="文档处理进度", unit="docs") as pbar:
                # 设置超时时间为1小时
                timeout = 3600
                start_time = time.time()
                
                # 分批处理，避免一次性提交太多任务
                batch_size = 50
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
