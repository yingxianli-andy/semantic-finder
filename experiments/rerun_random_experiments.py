#!/usr/bin/env python3
"""重跑Random方法的budget=762和1524的100个实验"""
import argparse
import json
import os
import sys
import multiprocessing
import threading
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import networkx as nx
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.data_loader import load_graph
from src.environment.opinion_env import OpinionDynamicsEnv
from src.agent.reward import compute_reward, compute_polarization_variance
from src.baselines.node_selection import select_nodes_random
import numpy as np

# 设置multiprocessing为spawn模式
multiprocessing.set_start_method('spawn', force=True)

def run_single_seed(args_tuple: Tuple) -> Tuple[str, int, int, Dict]:
    """运行单个seed的实验"""
    method, budget, seed, config_dict = args_tuple
    device = config_dict.get('device', 'cpu')
    graph = config_dict.get('graph')
    
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

        # Random方法选择节点
        selected_nodes = select_nodes_random(graph, budget, seed=seed)
        intervention_weights = [0.5] * len(selected_nodes)  # 修复：统一权重为0.5，确保公平对比

        # 运行环境
        state = env.reset()
        # 计算初始极化度
        initial_opinions = np.array([float(state.nodes[i].get("opinion", 0.0)) for i in range(state.number_of_nodes())])
        initial_polarization = compute_polarization_variance(opinions=initial_opinions)
        polarization_history = [initial_polarization]
        step_rewards = []

        for step, node in enumerate(selected_nodes):
            intervention_weight = intervention_weights[step]
            state, reward, done, info = env.step(node, intervention_weight)
            # 计算当前状态的极化度
            current_opinions = np.array([float(state.nodes[i].get("opinion", 0.0)) for i in range(state.number_of_nodes())])
            current_polarization = compute_polarization_variance(opinions=current_opinions)
            polarization_history.append(current_polarization)
            step_rewards.append(reward)

        final_score = polarization_history[-1] if polarization_history else 0.0

        result = {
            "final_score": float(final_score),
            "polarization_history": [float(x) for x in polarization_history],
            "selected_nodes": selected_nodes,
            "step_rewards": [float(x) for x in step_rewards]
        }

        return (method, budget, seed, result)
    except Exception as e:
        print(f"错误: {method} budget={budget} seed={seed} - {e}")
        return (method, budget, seed, None)

def main():
    parser = argparse.ArgumentParser(description='重跑Random方法的budget=762和1524实验')
    parser.add_argument('--dataset', type=str, default='ego-Twitter', help='数据集名称')
    parser.add_argument('--output-dir', type=str, default='results/ijcai_experiments', help='输出目录')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cpu/cuda)')
    parser.add_argument('--n-seeds', type=int, default=50, help='每个预算的seed数量')
    parser.add_argument('--n-procs', type=int, default=32, help='并行进程数')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("重跑Random方法的budget=762和1524实验")
    print("=" * 70)
    print()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    result_jsonl = os.path.join(args.output_dir, "results_main_comparison.jsonl")
    
    # 加载图
    print("加载图数据...")
    graph = load_graph(args.dataset, opinion_init="bimodal")
    n_nodes = graph.number_of_nodes()
    print(f"图加载完成: {n_nodes} 节点")
    print()
    
    # 构建任务列表（只跑Random方法的budget=762和1524）
    budgets = [762, 1524]
    tasks = []
    
    for budget in budgets:
        for seed in range(args.n_seeds):
            config_dict = {
                'device': args.device,
                'graph': graph
            }
            task = ("Random", budget, seed, config_dict)
            tasks.append(task)
    
    total_tasks = len(tasks)
    print(f"总任务数: {total_tasks} (Random方法, budget={budgets}, {args.n_seeds}个seed)")
    print()
    
    # 结果保存函数
    results_lock = threading.Lock()
    finished = 0
    
    def save_result_jsonl(method, budget, seed, result):
        if result is None:
            return
        line_record = {
            "method": method,
            "budget": budget,
            "seed": seed,
            "final_score": result.get("final_score", 0.0),
            "polarization_history": result.get("polarization_history", []),
            "selected_nodes": result.get("selected_nodes", []),
            "step_rewards": result.get("step_rewards", [])
        }
        with open(result_jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(line_record, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
    
    def save_callback(args_tuple):
        nonlocal finished
        method, budget, seed, result = args_tuple
        results_lock.acquire()
        try:
            save_result_jsonl(method, budget, seed, result)
            finished += 1
            print(f"✓ [{finished}/{total_tasks}] {method} budget_{budget} seed_{seed} 完成")
        finally:
            results_lock.release()
    
    # 运行任务
    print(f"开始运行，使用 {args.n_procs} 个进程并行...")
    start_time = time.time()
    
    with multiprocessing.Pool(processes=args.n_procs) as pool:
        for result in pool.imap_unordered(run_single_seed, tasks):
            save_callback(result)
    
    elapsed = time.time() - start_time
    print(f"\n实验完成！总耗时: {elapsed/60:.2f} 分钟")
    print(f"结果已保存到: {result_jsonl}")

if __name__ == '__main__':
    main()
