#!/usr/bin/env python3
"""
从损坏的JSON文件中恢复所有有效的实验结果
"""
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

INPUT_FILE = 'results/ijcai_experiments/results_main_comparison.json'
OUTPUT_FILE = 'results/ijcai_experiments/results_recovered.jsonl'
OUTPUT_JSON = 'results/ijcai_experiments/results_recovered.json'

METHODS = ["Random", "High-Degree", "PageRank", "FINDER", "Semantic-FINDER"]
BUDGETS = [762, 1524, 3812]


def find_json_object_boundaries(content: str, start_pos: int) -> Optional[Tuple[int, int]]:
    """找到从start_pos开始的完整JSON对象的边界"""
    depth = 0
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    
    obj_start = None
    for i in range(start_pos, len(content)):
        char = content[i]
        
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if in_string:
            continue
            
        if char == '{':
            if obj_start is None:
                obj_start = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and obj_start is not None:
                return (obj_start, i + 1)
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            
    return None


def extract_seed_objects(content: str) -> List[Dict]:
    """提取所有包含seed字段的JSON对象"""
    results = []
    pattern = r'"seed"\s*:\s*(\d+)'
    
    for match in re.finditer(pattern, content):
        seed_pos = match.start()
        # 向前查找对象开始
        obj_start = content.rfind('{', 0, seed_pos)
        if obj_start == -1:
            continue
            
        # 尝试找到完整的对象
        boundaries = find_json_object_boundaries(content, obj_start)
        if boundaries is None:
            continue
            
        obj_start, obj_end = boundaries
        obj_str = content[obj_start:obj_end]
        
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict) and 'seed' in obj:
                results.append((obj_start, obj))
        except json.JSONDecodeError:
            continue
    
    return results


def extract_method_budget_context(content: str, seed_obj_pos: int) -> Optional[Tuple[str, int]]:
    """从seed对象的位置推断它属于哪个method和budget"""
    # 向前查找最近的method和budget
    before_seed = content[:seed_obj_pos]
    
    # 查找最近的method（查找 "method": { 模式）
    method_found = None
    method_pos = -1
    for method in METHODS:
        # 查找 "method": { 或 "method":\n{
        pattern = rf'"{re.escape(method)}"\s*:\s*\{{'
        # 从后向前查找最近的匹配
        matches = list(re.finditer(pattern, before_seed))
        if matches:
            last_match = matches[-1]
            if last_match.end() > method_pos:
                method_pos = last_match.end()
                method_found = method
    
    if method_found is None:
        return None
    
    # 在method块内查找最近的budget
    # 从method位置开始，查找所有budget，选择最接近seed_obj_pos的那个
    method_block_start = content.rfind(f'"{method_found}"', 0, seed_obj_pos)
    if method_block_start == -1:
        return None
    
    # 查找method块的结束位置（下一个method或results的结束）
    method_block_end = seed_obj_pos + 1000000  # 限制搜索范围
    for other_method in METHODS:
        if other_method != method_found:
            other_pos = content.find(f'"{other_method}"', method_block_start)
            if other_pos != -1 and other_pos < method_block_end:
                method_block_end = other_pos
    
    method_block = content[method_block_start:method_block_end]
    seed_pos_in_block = seed_obj_pos - method_block_start
    
    budget_found = None
    budget_pos = -1
    for budget in BUDGETS:
        budget_key = f'"budget_{budget}"'
        pattern = rf'{re.escape(budget_key)}\s*:\s*\['
        matches = list(re.finditer(pattern, method_block))
        for match in matches:
            actual_pos = method_block_start + match.end()
            if actual_pos < seed_obj_pos and actual_pos > budget_pos:
                budget_pos = actual_pos
                budget_found = budget
    
    if budget_found is None:
        return None
    
    return (method_found, budget_found)


def extract_results_from_array(content: str, array_start: int, array_end: int) -> List[Dict]:
    """从数组内容中提取所有结果对象"""
    results = []
    array_content = content[array_start:array_end]
    
    # 查找所有seed对象
    seed_pattern = r'"seed"\s*:\s*(\d+)'
    for match in re.finditer(seed_pattern, array_content):
        seed_pos_in_array = match.start()
        # 在原始内容中的位置
        seed_pos = array_start + seed_pos_in_array
        
        # 向前查找对象开始
        obj_start_in_array = array_content.rfind('{', 0, seed_pos_in_array)
        if obj_start_in_array == -1:
            continue
        
        obj_start = array_start + obj_start_in_array
        
        # 找到完整的对象
        boundaries = find_json_object_boundaries(content, obj_start)
        if boundaries is None:
            continue
        
        obj_start_full, obj_end_full = boundaries
        # 确保对象在数组范围内
        if obj_end_full > array_end:
            continue
        
        obj_str = content[obj_start_full:obj_end_full]
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict) and 'seed' in obj:
                results.append(obj)
        except json.JSONDecodeError:
            continue
    
    return results


def recover_all_results() -> Dict:
    """恢复所有结果"""
    print(f"读取文件: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 文件不存在 {INPUT_FILE}")
        return {}
    
    content = Path(INPUT_FILE).read_text(encoding='utf-8')
    print(f"文件大小: {len(content)} 字符")
    
    # 首先尝试标准JSON解析
    print("\n尝试标准JSON解析...")
    try:
        data = json.loads(content)
        print("✓ 标准JSON解析成功！")
        return data
    except json.JSONDecodeError as e:
        print(f"✗ 标准JSON解析失败: {e}")
        print("开始恢复模式...")
    
    # 恢复模式：先提取所有seed对象，再推断method和budget
    print("\n提取所有seed对象...")
    seed_objects = extract_seed_objects(content)
    print(f"找到 {len(seed_objects)} 个seed对象")
    
    # 为每个对象推断method和budget
    print("\n推断method和budget...")
    recovered = defaultdict(lambda: defaultdict(list))
    
    for obj_pos, obj in seed_objects:
        context = extract_method_budget_context(content, obj_pos)
        if context is None:
            continue
        
        method, budget = context
        seed = obj.get('seed')
        
        # 检查是否已存在（去重）
        existing = recovered[method][budget]
        if not any(r.get('seed') == seed for r in existing):
            recovered[method][budget].append(obj)
    
    # 统计
    for method in METHODS:
        for budget in BUDGETS:
            count = len(recovered[method][budget])
            if count > 0:
                print(f"  {method} budget_{budget}: {count} 条结果")
    
    # 转换为标准格式
    results = {}
    total_count = 0
    for method in METHODS:
        results[method] = {}
        for budget in BUDGETS:
            budget_key = f'budget_{budget}'
            results[method][budget_key] = recovered[method][budget]
            count = len(recovered[method][budget])
            total_count += count
            print(f"  {method} budget_{budget}: {count} 条结果")
    
    print(f"\n总共恢复 {total_count} 条结果")
    
    # 构建完整的数据结构
    data = {
        "experiment_id": "exp1_main_comparison",
        "config": {
            "dataset": "ego-Twitter",
            "methods": METHODS,
            "budgets": BUDGETS,
            "n_seeds": 50,
            "n_nodes": 76245
        },
        "results": results
    }
    
    return data


def save_as_jsonl(data: Dict, output_file: str):
    """保存为JSONL格式（每行一个结果）"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for method in METHODS:
            for budget in BUDGETS:
                budget_key = f'budget_{budget}'
                for result in data['results'][method][budget_key]:
                    record = {
                        'method': method,
                        'budget': budget,
                        **result
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✓ JSONL格式已保存到: {output_file}")


def save_as_json(data: Dict, output_file: str):
    """保存为标准JSON格式"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON格式已保存到: {output_file}")


def main():
    print("=" * 60)
    print("实验结果恢复工具")
    print("=" * 60)
    
    # 恢复结果
    data = recover_all_results()
    
    if not data or not data.get('results'):
        print("\n✗ 未能恢复任何结果")
        return
    
    # 保存结果
    print("\n保存恢复的结果...")
    save_as_jsonl(data, OUTPUT_FILE)
    save_as_json(data, OUTPUT_JSON)
    
    # 统计信息
    print("\n" + "=" * 60)
    print("恢复统计:")
    print("=" * 60)
    total = 0
    for method in METHODS:
        for budget in BUDGETS:
            budget_key = f'budget_{budget}'
            count = len(data['results'][method][budget_key])
            total += count
            print(f"{method:20s} budget_{budget:4d}: {count:3d} 条")
    print(f"{'总计':20s} {'':8s}: {total:3d} 条")
    print("=" * 60)


if __name__ == '__main__':
    main()
