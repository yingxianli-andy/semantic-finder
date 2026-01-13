"""
从日志中恢复实验进度，保存到结果文件
用于在旧代码运行时提取已完成的结果，让新代码可以继续运行
"""
import json
import re
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.data_loader import load_graph
from src.agent.reward import compute_polarization_variance


def parse_log_for_completed_seeds(log_file: str):
    """
    从日志中解析已完成的seed信息
    
    注意：这个方法只能提取部分信息，因为旧代码没有保存完整结果
    只能提取进度信息，无法恢复完整的polarization_history
    """
    print(f"正在解析日志文件: {log_file}")
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 查找所有进度条记录
    # 格式: Random (K=762):  22%|██▏       | 11/50
    pattern = r'(\w+(?:-\w+)?)\s+\(K=(\d+)\):.*?(\d+)%\|.*?(\d+)/(\d+)'
    matches = re.findall(pattern, content)
    
    completed = {}
    for method, budget, pct, curr, total in matches:
        key = f"{method}_K={budget}"
        if key not in completed:
            completed[key] = 0
        completed[key] = max(completed[key], int(curr))
    
    print(f"从日志中提取的进度:")
    for key, count in completed.items():
        print(f"  {key}: {count}个seed")
    
    return completed


def create_partial_result_file(output_file: str, completed_info: dict, dataset_name: str):
    """
    创建部分结果文件，标记已完成的seed
    
    注意：由于旧代码没有保存完整结果，这里只能创建占位符
    新代码会识别这些占位符，跳过已完成的seed
    """
    # 解析completed_info
    results = {}
    for key, count in completed_info.items():
        # key格式: "Random_K=762"
        parts = key.split('_K=')
        if len(parts) != 2:
            continue
        method = parts[0]
        budget = int(parts[1])
        budget_key = f"budget_{budget}"
        
        if method not in results:
            results[method] = {}
        if budget_key not in results[method]:
            results[method][budget_key] = []
        
        # 为已完成的seed创建占位符结果
        # 新代码会检查len(results[method][budget_key])来跳过已完成的
        for seed in range(count):
            # 创建占位符结果（标记为已完成，但数据不完整）
            placeholder = {
                "seed": seed,
                "polarization_history": [],  # 空列表表示数据不完整
                "final_score": 0.0,
                "selected_nodes": [],
                "step_rewards": [],
                "_placeholder": True  # 标记为占位符
            }
            results[method][budget_key].append(placeholder)
    
    # 加载图获取节点数
    graph = load_graph(dataset_name, opinion_init="bimodal")
    n_nodes = graph.number_of_nodes()
    
    # 计算budgets
    budgets = [
        max(1, int(n_nodes * 0.01)),
        max(1, int(n_nodes * 0.02)),
        max(1, int(n_nodes * 0.05)),
    ]
    
    output_data = {
        "experiment_id": "exp1_main_comparison",
        "config": {
            "dataset": dataset_name,
            "methods": ["Random", "High-Degree", "PageRank", "FINDER", "Semantic-FINDER"],
            "budgets": budgets,
            "n_seeds": 50,
            "n_nodes": n_nodes,
        },
        "results": results,
    }
    
    # 保存文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n已创建结果文件: {output_file}")
    print(f"包含 {len(results)} 个方法的进度信息")
    print("\n注意：这些是占位符结果，新代码会跳过这些seed，重新运行以获取完整数据")


if __name__ == "__main__":
    log_file = "results/ijcai_experiments/exp1.log"
    output_file = "results/ijcai_experiments/results_main_comparison.json"
    dataset_name = "ego-Twitter"
    
    print("=" * 60)
    print("从日志恢复实验进度")
    print("=" * 60)
    
    # 解析日志
    completed_info = parse_log_for_completed_seeds(log_file)
    
    if not completed_info:
        print("\n未找到已完成的进度信息")
        sys.exit(1)
    
    # 创建结果文件
    create_partial_result_file(output_file, completed_info, dataset_name)
    
    print("\n" + "=" * 60)
    print("恢复完成！")
    print("=" * 60)
    print("\n现在可以使用新代码继续运行实验：")
    print("  python experiments/run_ijcai_experiments.py --experiment 1 --dataset ego-Twitter --device cuda --n_seeds 50")
    print("\n新代码会自动跳过已完成的seed，继续未完成的部分。")
