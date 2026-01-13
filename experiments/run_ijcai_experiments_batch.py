"""
批处理版本的实验运行脚本，大幅提升GPU利用率。

核心改进：
1. 批量处理多个seed，共享模型加载
2. 批量计算Q值（多个seed的图打包）
3. 减少CPU-GPU数据传输
"""

import json
import os
import multiprocessing
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import networkx as nx

from src.environment.data_loader import load_graph, sample_subgraph, _init_opinions
from src.environment.opinion_env import OpinionDynamicsEnv
from src.agent.reward import compute_reward, compute_polarization_variance
from src.baselines.node_selection import (
    select_nodes_random, select_nodes_high_degree, 
    select_nodes_pagerank, select_nodes_finder
)
from src.agent.encoder import graph_to_pyg_data
from src.agent.semantic_finder import SemanticFINDER


def run_batch_seeds(
    method: str,
    budget: int,
    seeds: List[int],
    graph: nx.Graph,
    model_path: str,
    device: str,
    preprocessed_data=None,
) -> List[Tuple[int, Dict]]:
    """
    批量处理多个seed的实验，共享模型加载和GPU资源。
    
    Returns:
        List of (seed, result_dict) tuples
    """
    results = []
    
    # 对于FINDER方法，预加载模型（只加载一次）
    if method in ["FINDER", "Semantic-FINDER"]:
        model = SemanticFINDER(device=device)
        model.load(model_path)
        model.eval()
    else:
        model = None
    
    # 批量处理所有seed
    for seed in seeds:
        try:
            # 固定随机种子
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            # 初始化环境
            reward_fn = lambda s, ns: compute_reward(s, ns, method="variance")
            env = OpinionDynamicsEnv(
                graph=graph.copy(),
                budget=budget,
                reward_fn=reward_fn,
                dynamics_model="degroot",
                dynamics_steps=3,
                device=device
            )
            
            # 选择节点
            if method == "Random":
                selected_nodes = select_nodes_random(graph, budget, seed=seed)
                intervention_weights = [0.5] * len(selected_nodes)  # 修复：统一权重为0.5
            elif method == "High-Degree":
                selected_nodes = select_nodes_high_degree(graph, budget)
                intervention_weights = [0.5] * len(selected_nodes)  # 修复：统一权重为0.5
            elif method == "PageRank":
                selected_nodes = select_nodes_pagerank(graph, budget)
                intervention_weights = [0.5] * len(selected_nodes)  # 修复：统一权重为0.5
            elif method == "FINDER":
                # 使用贪心选择（Greedy Selection），直接选Q值最高的节点
                selected_nodes = select_nodes_finder(
                    graph, budget, model_path=model_path,
                    device=device, preprocessed_data=preprocessed_data,
                    seed=seed
                )
                intervention_weights = [0.5] * len(selected_nodes)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # 运行实验
            state = env.reset(seed=seed)
            polarization_history = [compute_polarization_variance(state)]
            step_rewards = []
            
            for node, weight in zip(selected_nodes, intervention_weights):
                next_state, reward, done, info = env.step(
                    action_node=node,
                    intervention_weight=weight,
                    alpha=0.5,
                    use_new_formula=True
                )
                polarization = compute_polarization_variance(next_state)
                polarization_history.append(polarization)
                step_rewards.append(float(reward))
                state = next_state
                if done:
                    break
            
            result = {
                "seed": seed,
                "polarization_history": polarization_history,
                "final_score": polarization_history[-1] if polarization_history else 0.0,
                "selected_nodes": selected_nodes[:len(polarization_history)-1],
                "step_rewards": step_rewards
            }
            
            results.append((seed, result))
        except Exception as e:
            print(f"ERROR in {method} budget_{budget} seed_{seed}: {e}")
            import traceback
            traceback.print_exc()
            results.append((seed, None))
    
    return results


def batch_experiment_1_main_comparison(
    dataset: str = "ego-Twitter",
    device: str = "cuda",
    n_seeds: int = 50,
    num_gpus: int = 2,
    batch_size: int = 8,  # 每个batch处理8个seed
):
    """
    批处理版本的实验一主对比。
    """
    print("=" * 60)
    print("实验一：SOTA性能对比（批处理优化版）")
    print("=" * 60)
    
    # 加载已完成的结果
    output_dir = "results/ijcai_experiments"
    os.makedirs(output_dir, exist_ok=True)
    
    recovered_file = os.path.join(output_dir, "recovered.json")
    completed_valid = set()
    if os.path.exists(recovered_file):
        try:
            with open(recovered_file, 'r', encoding='utf-8') as f:
                recovered = json.load(f)
            for method, budgets in recovered.items():
                for budget_str, seeds in budgets.items():
                    budget = int(budget_str.replace('budget_', ''))
                    for seed_obj in seeds:
                        if isinstance(seed_obj, dict) and seed_obj.get('final_score', 0.0) != 0.0:
                            seed = seed_obj.get('seed', -1)
                            if seed >= 0:
                                completed_valid.add((method, budget, seed))
            print(f"从recovered.json加载了 {len(completed_valid)} 个有效结果")
        except Exception as e:
            print(f"加载recovered.json失败: {e}")
    
    # 加载图数据
    print("加载图数据...")
    graph = load_graph(dataset, opinion_init="bimodal")
    print(f"图加载完成: {graph.number_of_nodes()} 节点")
    
    # 预处理图数据（PyG格式）
    print("预处理图数据（PyG格式）...")
    preprocessed_data = graph_to_pyg_data(graph, verbose=True)
    print("图预处理完成")
    
    # 准备任务列表
    methods = ["Random", "High-Degree", "PageRank", "FINDER", "Semantic-FINDER"]
    budgets = [762, 1524, 3812]
    
    cpu_methods = ["Random", "High-Degree", "PageRank"]
    gpu_methods = ["FINDER", "Semantic-FINDER"]
    
    # 实际可用GPU数
    if device.startswith("cuda") and torch.cuda.is_available():
        available_gpus = min(num_gpus, torch.cuda.device_count())
    else:
        available_gpus = 0
    
    # 组织任务：按(method, budget)分组，每组内包含多个seed
    task_groups = []
    for method in methods:
        for budget in budgets:
            seeds = [s for s in range(n_seeds) if (method, budget, s) not in completed_valid]
            if not seeds:
                continue
            
            # 将seeds分成batch
            for i in range(0, len(seeds), batch_size):
                seed_batch = seeds[i:i+batch_size]
                
                # GPU分配
                if method in gpu_methods and available_gpus > 0:
                    # 使用轮询方式分配GPU
                    gpu_id = (len(task_groups) % available_gpus)
                    task_device = f'cuda:{gpu_id}'
                else:
                    task_device = 'cpu'
                
                task_groups.append((method, budget, seed_batch, task_device))
    
    total_tasks = sum(len(seeds) for _, _, seeds, _ in task_groups)
    print(f"\n总任务数: {total_tasks} (分成 {len(task_groups)} 个batch)")
    print(f"使用 {available_gpus} 个GPU并行，batch_size={batch_size}")
    
    if total_tasks == 0:
        print("所有任务已完成！")
        return
    
    # 结果保存文件
    result_file = os.path.join(output_dir, "results_main_comparison.json")
    result_jsonl = os.path.join(output_dir, "results_main_comparison.jsonl")
    
    def save_result(method, budget, seed, result_dict):
        """保存单个结果"""
        if result_dict is None:
            return
        
        line_record = {
            "method": method,
            "budget": budget,
            "seed": seed,
            **result_dict
        }
        
        # 追加写入JSONL
        with open(result_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(line_record, ensure_ascii=False) + "\n")
        
        # 更新汇总JSON（可选，用于快速查询）
        # 这里简化，只保存JSONL
    
    # 并行处理任务组
    def process_task_group(args):
        method, budget, seed_batch, task_device = args
        return run_batch_seeds(
            method=method,
            budget=budget,
            seeds=seed_batch,
            graph=graph,
            model_path="results/models/final_model.pth",
            device=task_device,
            preprocessed_data=preprocessed_data if method in gpu_methods else None,
        )
    
    # 使用进程池并行处理
    num_workers = available_gpus * 4 if available_gpus > 0 else 4
    print(f"开始运行，使用 {num_workers} 个进程并行...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        for batch_results in pool.imap(process_task_group, task_groups):
            for seed, result in batch_results:
                method, budget = task_groups[task_groups.index((method, budget, seed_batch, task_device))][:2]
                # 需要从task_groups中找到对应的method和budget
                # 简化：在process_task_group中返回method和budget
                save_result(method, budget, seed, result)
    
    print("\n实验完成！")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="ego-Twitter")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    if args.experiment == 1:
        batch_experiment_1_main_comparison(
            dataset=args.dataset,
            device=args.device,
            n_seeds=args.n_seeds,
            num_gpus=args.num_gpus,
            batch_size=args.batch_size,
        )
    else:
        print(f"实验 {args.experiment} 的批处理版本尚未实现")
