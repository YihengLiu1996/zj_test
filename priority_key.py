def prioritize_keys(data, priority_keys):
    """
    将字典中指定的 key 放在最前面，其余保持原顺序
    :param data: 原始字典
    :param priority_keys: 你想优先显示的 key 列表（按你想要的顺序）
    :return: 重排序后的字典
    """
    # 提取优先 key（按你给的顺序 + 安全过滤）
    prioritized = {k: data[k] for k in priority_keys if k in data}
    # 提取剩余 key（保持原始顺序）
    remaining = {k: v for k, v in data.items() if k not in priority_keys}
    # 合并：优先的在前，其余在后
    return {**prioritized, **remaining}

# 使用示例
data = {"x": 1, "y": 2, "z": 3, "a": 4, "b": 5}
result = prioritize_keys(data, ["z", "a"])
print(json.dumps(result, indent=2))
