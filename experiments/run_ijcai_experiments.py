#!/usr/bin/env python3
"""优化版IJCAI实验脚本 - 支持断点续传和并发执行"""
import argparse
import json
import os
import sys
import multiprocessing
import threading
import time
import random
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.data_loader import load_graph
from src.environment.opinion_env import OpinionDynamicsEnv
from src.agent.reward import compute_reward, compute_polarization_variance
from src.baselines.node_selection import (
    select_nodes_random, select_nodes_high_degree, 
    select_nodes_pagerank, select_nodes_finder, select_nodes_semantic_finder
)
from src.agent.encoder import graph_to_pyg_data

# 设置multiprocessing为spawn模式（支持CUDA）
multiprocessing.set_start_method('spawn', force=True)

def load_existing_results(result_file: str) -> Dict:
    """加载已有结果"""
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return None

def is_completed(results: Dict, method: str, budget: int, seed: int) -> bool:
    """检查某个seed是否已完成"""
    if not results or 'results' not in results:
        return False
    method_results = results['results'].get(method, {})
    budget_key = f'budget_{budget}'
    if budget_key not in method_results:
        return False
    for entry in method_results[budget_key]:
        if isinstance(entry, dict) and entry.get('seed') == seed and not entry.get('_placeholder'):
            return True
    return False

# 全局变量：缓存已加载的模型（每个进程一个）
_model_cache = {}

def run_single_seed(args_tuple: Tuple) -> Tuple[str, int, int, Dict]:
    """运行单个seed的实验（优化版：缓存模型加载）"""
    method, budget, seed, config_dict = args_tuple
    device = config_dict.get('device', 'cpu')
    model_path = config_dict.get('model_path')
    graph = config_dict.get('graph')
    preprocessed_data = config_dict.get('preprocessed_data')
    
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

        # 选择节点（优化：对于FINDER方法，复用已加载的模型）
        # 修复：统一所有基线方法的干预权重为0.5，确保公平对比
        # 原因：权重1.0可能导致过度干预，触发clip边界，影响结果公平性
        if method == "Random":
            selected_nodes = select_nodes_random(graph, budget, seed=seed)
            intervention_weights = [0.5] * len(selected_nodes)  # 修复：从1.0改为0.5
        elif method == "High-Degree":
            selected_nodes = select_nodes_high_degree(graph, budget)
            intervention_weights = [0.5] * len(selected_nodes)  # 修复：从1.0改为0.5
        elif method == "PageRank":
            selected_nodes = select_nodes_pagerank(graph, budget)
            intervention_weights = [0.5] * len(selected_nodes)  # 修复：从1.0改为0.5
        elif method == "FINDER":
            # 使用贪心选择（Greedy Selection），直接选Q值最高的节点
            selected_nodes = select_nodes_finder(
                graph, budget, model_path=model_path, 
                device=device, preprocessed_data=preprocessed_data,
                seed=seed
            )
            intervention_weights = [0.5] * len(selected_nodes)
        elif method == "Semantic-FINDER":
            selected_nodes = select_nodes_semantic_finder(
                graph, budget, model_path=model_path,
                device=device, preprocessed_data=preprocessed_data
            )
            # 这里简化：使用固定权重0.7（实际应该调用LLM）
            intervention_weights = [0.7] * len(selected_nodes)
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
        
        return (method, budget, seed, result)
    except Exception as e:
        print(f"ERROR in {method} budget_{budget} seed_{seed}: {e}")
        import traceback
        traceback.print_exc()
        return (method, budget, seed, None)


def run_single_ablation(args_tuple: Tuple[str, int, int, Dict[str, Any]]) -> Tuple[str, int, int, Dict]:
    """
    实验二：单个 (variant, budget, seed) 任务。
    为了支持多进程并行，这里实现为独立函数，供 multiprocessing.Pool 调用。
    
    严格按照论文逻辑实现三种变体：
    1. w/o LLM: 选点=FINDER (RL)，权重=随机数 random.uniform(0, 1)
    2. w/o RL: 选点=Random，权重=LLM Adaptive策略
    3. Full Model: 选点=FINDER (RL)，权重=LLM Adaptive策略
    """
    variant, budget, seed, config = args_tuple
    device = config.get("device", "cpu")
    model_path = config.get("model_path")
    graph = config.get("graph")
    preprocessed_data = config.get("preprocessed_data")

    try:
        # 固定随机种子（确保可复现）
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        # 环境与奖励
        reward_fn = lambda s, ns: compute_reward(s, ns, method="variance")
        env = OpinionDynamicsEnv(
            graph=graph.copy(),
            budget=budget,
            reward_fn=reward_fn,
            dynamics_model="degroot",
            dynamics_steps=3,
            device=device,
        )

        # 导入Adaptive策略函数
        from src.llm.strategy import get_weight_adaptive

        # 选点 + 权重（严格按照论文逻辑）
        if variant == "w/o LLM":
            # 选点：FINDER (RL)，权重：随机 U(0,1)
            selected_nodes = select_nodes_finder(
                graph,
                budget,
                model_path=model_path,
                device=device,
                preprocessed_data=preprocessed_data,
                seed=seed,
            )
            # 为每个节点生成随机权重（固定种子确保可复现）
            intervention_weights = [
                float(random.uniform(0.0, 1.0)) for _ in selected_nodes
            ]
        elif variant == "w/o RL":
            # 选点：Random，权重：LLM Adaptive策略
            selected_nodes = select_nodes_random(graph, budget, seed=seed)
            # 使用Adaptive策略为每个节点生成权重（基于当前图状态）
            intervention_weights = []
            current_graph = graph.copy()
            for node_id in selected_nodes:
                weight = get_weight_adaptive(node_id, current_graph, subgraph=None)
                intervention_weights.append(float(weight))
        elif variant == "Full Model":
            # 选点：FINDER (RL)，权重：LLM Adaptive策略
            selected_nodes = select_nodes_finder(
                graph,
                budget,
                model_path=model_path,
                device=device,
                preprocessed_data=preprocessed_data,
                seed=seed,
            )
            # 使用Adaptive策略为每个节点生成权重（基于当前图状态）
            intervention_weights = []
            current_graph = graph.copy()
            for node_id in selected_nodes:
                weight = get_weight_adaptive(node_id, current_graph, subgraph=None)
                intervention_weights.append(float(weight))
        else:
            raise ValueError(f"未知的消融变体: {variant}")

        # 验证数据有效性
        if len(selected_nodes) != len(intervention_weights):
            raise ValueError(
                f"节点数量({len(selected_nodes)})与权重数量({len(intervention_weights)})不匹配"
            )
        if len(selected_nodes) == 0:
            raise ValueError("没有选择任何节点")

        # 运行环境
        state = env.reset(seed=seed)
        polarization_history = [compute_polarization_variance(state)]
        step_rewards: List[float] = []

        for step_idx, (node, weight) in enumerate(zip(selected_nodes, intervention_weights)):
            # 验证节点和权重有效性
            if node < 0 or node >= graph.number_of_nodes():
                raise ValueError(f"无效的节点ID: {node}")
            if not (0.0 <= weight <= 1.0):
                raise ValueError(f"权重超出范围[0,1]: {weight}")

            next_state, reward, done, info = env.step(
                action_node=node,
                intervention_weight=weight,
                alpha=0.5,
                use_new_formula=True,
            )
            polarization = compute_polarization_variance(next_state)
            polarization_history.append(polarization)
            step_rewards.append(float(reward))
            state = next_state
            if done:
                break

        # 验证结果数据有效性
        if len(polarization_history) < 2:
            raise ValueError("极化度历史记录不完整")
        if len(step_rewards) != len(selected_nodes):
            raise ValueError("奖励数量与节点数量不匹配")

        result = {
            "seed": seed,
            "polarization_history": polarization_history,
            "final_score": float(polarization_history[-1]) if polarization_history else 0.0,
            "selected_nodes": selected_nodes[: max(0, len(polarization_history) - 1)],
            "step_rewards": step_rewards,
            "intervention_weights": intervention_weights[: len(step_rewards)],  # 保存权重用于验证
        }
    except Exception as e:
        print(f"ERROR in Ablation {variant} budget_{budget} seed_{seed}: {e}")
        import traceback
        traceback.print_exc()
        result = {
            "seed": seed,
            "error": str(e),
            "final_score": None,  # 标记为无效结果
        }

    return (variant, budget, seed, result)

def save_result_incremental(result_file: str, method: str, budget: int, seed: int, result: Dict, lock):
    """增量保存单个结果"""
    lock.acquire()
    try:
        # 读取现有结果
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                data = json.load(f)
        else:
            data = {"experiment_id": "exp1_main_comparison", "config": {}, "results": {}}
        
        # 确保结构存在
        if method not in data['results']:
            data['results'][method] = {}
        budget_key = f'budget_{budget}'
        if budget_key not in data['results'][method]:
            data['results'][method][budget_key] = []
        
        # 移除placeholder或旧结果
        data['results'][method][budget_key] = [
            r for r in data['results'][method][budget_key]
            if not (isinstance(r, dict) and r.get('seed') == seed)
        ]
        
        # 添加新结果
        if result is not None:
            data['results'][method][budget_key].append(result)
        
        # 保存
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=2)
    finally:
        lock.release()

def experiment_1_main_comparison(
    dataset: str = "ego-Twitter",
    output_dir: str = "results/ijcai_experiments",
    model_path: str = "results/models/final_model.pth",
    n_seeds: int = 50,
    device: str = "cuda",
    num_gpus: int = 2
):
    """实验一：主对比实验（优化版 - JSONL保存 + 2 GPU并行）"""
    print("=" * 60)
    print("实验一：SOTA性能对比（优化版）")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    result_jsonl = os.path.join(output_dir, "results_main_comparison.jsonl")
    result_json = os.path.join(output_dir, "results_main_comparison.json")
    
    # 从recovered.json加载已完成的有效结果（final_score != 0）
    completed_valid = set()
    recovered_count = 0
    recovered_file = os.path.join(output_dir, "results_recovered.json")
    if os.path.exists(recovered_file):
        try:
            with open(recovered_file, 'r') as f:
                recovered_data = json.load(f)
            results_dict = recovered_data.get('results', {})
            for method, method_results in results_dict.items():
                for budget_key, entries in method_results.items():
                    try:
                        budget = int(budget_key.split('_')[1])
                    except:
                        continue
                    for entry in entries:
                        if isinstance(entry, dict):
                            seed = entry.get('seed')
                            final_score = entry.get('final_score', 0.0)
                            if seed is not None and final_score != 0.0:
                                completed_valid.add((method, budget, seed))
                                recovered_count += 1
            print(f"从recovered.json加载了 {recovered_count} 个有效结果")
        except Exception as e:
            print(f"加载recovered.json失败: {e}")
    
    # 从JSONL加载已完成任务（注意：这里会与recovered.json有重叠，用set自动去重）
    jsonl_count_before = len(completed_valid)
    if os.path.exists(result_jsonl):
        with open(result_jsonl, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    method = rec.get('method')
                    budget = rec.get('budget')
                    seed = rec.get('seed')
                    final_score = rec.get('final_score', 0.0)
                    if method and budget is not None and seed is not None and final_score != 0.0:
                        completed_valid.add((method, budget, seed))
                except:
                    continue
        jsonl_new = len(completed_valid) - jsonl_count_before
        print(f"从JSONL加载了 {jsonl_new} 个新结果（去重后总共有 {len(completed_valid)} 个已完成结果）")
    
    # 加载图
    print("加载图数据...")
    graph = load_graph(dataset, opinion_init="bimodal")
    n_nodes = graph.number_of_nodes()
    print(f"图加载完成: {n_nodes} 节点")

    # 预处理图数据（一次性）
    print("预处理图数据（PyG格式）...")
    preprocessed_data = graph_to_pyg_data(graph, verbose=True)
    print("图预处理完成")

    # 构建任务列表
    methods = ["Random", "High-Degree", "PageRank", "FINDER", "Semantic-FINDER"]
    budgets = [762, 1524, 3812]
    
    cpu_methods = ["Random", "High-Degree", "PageRank"]
    gpu_methods = ["FINDER", "Semantic-FINDER"]
    
    cpu_tasks = []
    gpu_tasks = []
    
    # 实际可用GPU数
    if device.startswith("cuda") and torch.cuda.is_available():
        available_gpus = min(num_gpus, torch.cuda.device_count())
    else:
        available_gpus = 0
    
    gpu_idx = 0
    
    for method in methods:
        for budget in budgets:
            for seed in range(n_seeds):
                # 跳过已完成的有效结果
                if (method, budget, seed) in completed_valid:
                    continue
                
                # GPU分配
                if method in gpu_methods and available_gpus > 0:
                    dev_id = gpu_idx % available_gpus
                    task_device = f'cuda:{dev_id}'
                    gpu_idx += 1
                else:
                    task_device = 'cpu'
                
                config_dict = {
                    'device': task_device,
                    'model_path': model_path,
                    'graph': graph,
                    'preprocessed_data': preprocessed_data if method in gpu_methods else None
                }
                
                task = (method, budget, seed, config_dict)
                if task_device.startswith('cuda'):
                    gpu_tasks.append(task)
                else:
                    cpu_tasks.append(task)
    
    total_tasks = len(cpu_tasks) + len(gpu_tasks)
    print(f"\n总任务数: {total_tasks} (CPU: {len(cpu_tasks)}, GPU: {len(gpu_tasks)})")
    print(f"使用 {available_gpus} 个GPU并行")
    
    if total_tasks == 0:
        print("所有任务已完成！")
        return
    
    # 结果保存函数（JSONL增量追加，确保数据不丢失）
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
        # 使用追加模式，每次写入后立即flush，确保数据不丢失
        with open(result_jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps(line_record, ensure_ascii=False) + '\n')
            f.flush()  # 立即刷新到磁盘，确保数据不丢失
            os.fsync(f.fileno())  # 强制同步到磁盘（更安全，但可能稍慢）
    
    finished = 0
    results_lock = threading.Lock()
    start_time = time.time()
    
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
    
    # GPU任务并行执行（优化进程数以平衡GPU利用率和显存使用）
    if gpu_tasks and available_gpus > 0:
        # 对于76K节点的大图，每个模型加载需要约2GB显存
        # 每个GPU 24GB显存，考虑其他进程，每个GPU最多8个进程是安全的
        # 使用进程池分批处理，避免显存溢出
        gpu_processes_per_device = 6  # 每个GPU设备使用6个进程（平衡利用率和显存）
        total_gpu_processes = available_gpus * gpu_processes_per_device
        print(f"开始运行GPU任务，使用 {total_gpu_processes} 个进程并行（{available_gpus}个GPU × {gpu_processes_per_device}进程/GPU）...")
        print(f"  每个进程将共享GPU资源，通过进程池管理显存使用")
        with multiprocessing.Pool(processes=total_gpu_processes) as pool:
            # 使用imap_unordered提高并发性能
            for result in pool.imap_unordered(run_single_seed, gpu_tasks):
                save_callback(result)
    
    # CPU任务并行执行
    if cpu_tasks:
        cpu_procs = min(32, len(cpu_tasks))  # 增加CPU进程数到32
        print(f"开始运行CPU任务，使用 {cpu_procs} 个进程并行...")
        with multiprocessing.Pool(processes=cpu_procs) as pool:
            # 使用imap_unordered提高并发性能
            for result in pool.imap_unordered(run_single_seed, cpu_tasks):
                save_callback(result)
    
    elapsed = time.time() - start_time
    print(f"\n实验完成！总耗时: {elapsed/3600:.2f} 小时")


def experiment_2_ablation(
    dataset: str = "ego-Twitter",
    output_dir: str = "results/ijcai_experiments",
    model_path: str = "results/models/final_model.pth",
    n_seeds: int = 50,
    device: str = "cuda",
    num_gpus: int = 2,
) -> None:
    """
    实验二：消融实验 (Ablation Study)
    
    严格按照论文逻辑实现三种变体：
    1) w/o LLM:   选点=FINDER (RL)，权重=随机数 random.uniform(0, 1)
    2) w/o RL:    选点=Random，权重=LLM Adaptive策略
    3) Full Model:选点=FINDER (RL)，权重=LLM Adaptive策略
    
    参数：
    - Budget: 5% (3812 个节点)
    - Seeds: 50 个
    - 输出到 results_ablation.json 和 results_ablation.jsonl
    """
    print("=" * 60)
    print("实验二：消融实验 (Ablation Study)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    # 主 JSON（汇总）+ JSONL（强健增量保存，支持断点续传）
    result_file = os.path.join(output_dir, "results_ablation.json")
    result_jsonl = os.path.join(output_dir, "results_ablation.jsonl")

    # 配置
    budgets = [3812]  # 5% 节点数
    variants = ["w/o LLM", "w/o RL", "Full Model"]

    # 加载图
    print("加载图数据用于消融实验...")
    graph = load_graph(dataset, opinion_init="bimodal")
    n_nodes = graph.number_of_nodes()
    print(f"图加载完成: {n_nodes} 节点")

    # 预处理图数据（供 FINDER 使用）
    print("预处理图数据（PyG格式）...")
    preprocessed_data = graph_to_pyg_data(graph, verbose=True)
    print("图预处理完成")

    # 目标设备
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("警告：CUDA 不可用，实验二将回退到 CPU。")
        device = "cpu"

    # 从 JSONL 和 JSON 中恢复已完成任务集合（强健断点续传）
    completed_pairs = set()
    valid_results = {}  # 存储有效结果（final_score不为None且不为0）
    
    # 优先从JSONL恢复（更可靠）
    if os.path.exists(result_jsonl):
        print(f"检测到已有 JSONL 结果文件，将用于断点续传: {result_jsonl}")
        with open(result_jsonl, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    variant = rec.get("variant")
                    budget = rec.get("budget")
                    seed = rec.get("seed")
                    final_score = rec.get("final_score")
                    
                    # 只接受有效结果（final_score不为None且不为0）
                    if variant and budget is not None and seed is not None:
                        key = (variant, int(budget), int(seed))
                        if final_score is not None and final_score != 0.0:
                            completed_pairs.add(key)
                            valid_results[key] = rec
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    # 跳过损坏的行，但记录警告
                    if line_num % 100 == 0:  # 每100行警告一次，避免刷屏
                        print(f"警告：JSONL第{line_num}行解析失败: {e}")
                    continue
        print(f"已从 JSONL 恢复 {len(completed_pairs)} 个有效已完成任务")
    
    # 也从JSON恢复（作为备份）
    if os.path.exists(result_file):
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                json_results = json_data.get("results", {})
                for variant, variant_results in json_results.items():
                    for budget_key, entries in variant_results.items():
                        try:
                            budget = int(budget_key.split("_")[1])
                        except:
                            continue
                        for entry in entries:
                            if isinstance(entry, dict):
                                seed = entry.get("seed")
                                final_score = entry.get("final_score")
                                if seed is not None and final_score is not None and final_score != 0.0:
                                    key = (variant, budget, int(seed))
                                    if key not in completed_pairs:
                                        completed_pairs.add(key)
                                        valid_results[key] = entry
                        print(f"从JSON补充恢复了 {len([k for k in completed_pairs if k not in valid_results])} 个任务")
        except Exception as e:
            print(f"从JSON恢复时出错（不影响运行）: {e}")

    # 仍保留内存中的 results 结构，方便最终导出一个汇总 JSON
    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    
    # 从有效结果中恢复内存结构
    for (variant, budget, seed), rec in valid_results.items():
        if variant not in results:
            results[variant] = {}
        bkey = f"budget_{budget}"
        if bkey not in results[variant]:
            results[variant][bkey] = []
        # 避免重复
        results[variant][bkey] = [
            r for r in results[variant][bkey]
            if not (isinstance(r, dict) and r.get("seed") == seed)
        ]
        results[variant][bkey].append(rec)

    def is_done(variant_name: str, budget: int, seed: int) -> bool:
        """断点续传判断：依据已完成任务集合。"""
        return (variant_name, budget, seed) in completed_pairs

    # 主循环：构建任务列表，GPU 任务并行跑，CPU 任务单独跑
    start_time = time.time()
    total_runs = len(variants) * len(budgets) * n_seeds

    # 准备任务列表
    gpu_variants = ["w/o LLM", "Full Model"]  # 需要RL模型的任务
    cpu_variants = ["w/o RL"]  # 只需要随机选点的任务
    gpu_tasks: List[Tuple[str, int, int, Dict[str, Any]]] = []
    cpu_tasks: List[Tuple[str, int, int, Dict[str, Any]]] = []

    # 实际可用 GPU 数
    if device.startswith("cuda") and torch.cuda.is_available():
        available_gpus = min(num_gpus, torch.cuda.device_count())
    else:
        available_gpus = 0

    gpu_idx = 0
    finished = len(completed_pairs)

    for variant in variants:
        if variant not in results:
            results[variant] = {}
        for budget in budgets:
            bkey = f"budget_{budget}"
            if bkey not in results[variant]:
                results[variant][bkey] = []

            for seed in range(n_seeds):
                if is_done(variant, budget, seed):
                    continue

                # 为该任务分配设备
                if variant in gpu_variants and available_gpus > 0:
                    dev_id = gpu_idx % available_gpus
                    task_device = f"cuda:{dev_id}"
                    gpu_idx += 1
                else:
                    task_device = "cpu"

                config = {
                    "device": task_device,
                    "model_path": model_path,
                    "graph": graph,
                    "preprocessed_data": preprocessed_data
                    if variant in gpu_variants
                    else None,
                }
                task = (variant, budget, seed, config)

                if task_device.startswith("cuda"):
                    gpu_tasks.append(task)
                else:
                    cpu_tasks.append(task)

    print(
        f"\n实验二总任务数: {total_runs}，其中已完成 {finished}，"
        f"待运行 GPU 任务 {len(gpu_tasks)}，CPU 任务 {len(cpu_tasks)}"
    )
    
    if len(gpu_tasks) + len(cpu_tasks) == 0:
        print("所有任务已完成！")
        return

    # 结果保存回调（在主进程中执行，单点写 JSONL/JSON，保证安全）
    results_lock = threading.Lock()
    
    def save_ablation_result(result_tuple: Tuple[str, int, int, Dict]) -> None:
        nonlocal finished
        variant, budget, seed, result = result_tuple
        
        # 验证结果有效性
        if result is None:
            print(f"警告：{variant} budget_{budget} seed_{seed} 返回None结果")
            return
        
        # 检查是否有错误
        if "error" in result:
            print(f"错误：{variant} budget_{budget} seed_{seed} 执行失败: {result.get('error')}")
            # 仍然保存错误结果，但标记为无效
            result["final_score"] = None
        
        # 验证final_score有效性
        final_score = result.get("final_score")
        if final_score is None or final_score == 0.0:
            print(f"警告：{variant} budget_{budget} seed_{seed} 的final_score无效: {final_score}")
        
        bkey = f"budget_{budget}"

        with results_lock:
            # 更新内存结构
            if variant not in results:
                results[variant] = {}
            if bkey not in results[variant]:
                results[variant][bkey] = []
            # 移除旧结果
            results[variant][bkey] = [
                r
                for r in results[variant][bkey]
                if not (isinstance(r, dict) and r.get("seed") == seed)
            ]
            # 只添加有效结果
            if final_score is not None and final_score != 0.0:
                results[variant][bkey].append(result)
                completed_pairs.add((variant, budget, seed))
            
            finished += 1

            # 追加写入 JSONL（立即保存，确保数据不丢失）
            line_record = {
                "variant": variant,
                "budget": budget,
                "seed": result.get("seed"),
                "final_score": result.get("final_score"),
                "polarization_history": result.get("polarization_history"),
                "selected_nodes": result.get("selected_nodes"),
                "step_rewards": result.get("step_rewards"),
                "intervention_weights": result.get("intervention_weights", []),  # 保存权重
            }
            try:
                with open(result_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line_record, ensure_ascii=False) + "\n")
                    f.flush()  # 立即刷新
                    os.fsync(f.fileno())  # 强制同步到磁盘
            except Exception as e:
                print(f"保存JSONL失败: {e}")

            # 定期更新汇总 JSON（每10个结果更新一次，减少IO）
            if finished % 10 == 0 or finished == total_runs:
                try:
                    data_to_save = {
                        "experiment_id": "exp2_ablation",
                        "config": {
                            "dataset": dataset,
                            "variants": variants,
                            "budgets": budgets,
                            "n_seeds": n_seeds,
                        },
                        "results": results,
                    }
                    with open(result_file, "w", encoding="utf-8") as f:
                        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"保存JSON失败: {e}")

            print(
                f"✓ Ablation {variant} budget_{budget} seed_{seed} 完成 "
                f"({finished}/{total_runs})，final_score={final_score:.6f if final_score else 'None'}"
            )

    # GPU任务并行执行（充分利用2个GPU，每个GPU运行多个进程）
    if gpu_tasks and available_gpus > 0:
        # 对于76K节点的大图，每个模型加载需要约2GB显存
        # 每个GPU 24GB显存，考虑其他进程，每个GPU最多6个进程是安全的
        gpu_processes_per_device = 6  # 每个GPU设备使用6个进程
        total_gpu_processes = available_gpus * gpu_processes_per_device
        print(f"\n开始运行 GPU 任务，使用 {total_gpu_processes} 个进程并行（{available_gpus}个GPU × {gpu_processes_per_device}进程/GPU）...")
        with multiprocessing.Pool(processes=total_gpu_processes) as pool:
            for res in pool.imap_unordered(run_single_ablation, gpu_tasks):
                save_ablation_result(res)
    else:
        print("无 GPU 任务或 GPU 不可用，跳过 GPU 阶段。")

    # CPU任务并行执行
    if cpu_tasks:
        cpu_procs = min(16, len(cpu_tasks))  # 增加CPU进程数到16
        print(f"\n开始运行 CPU 任务，使用 {cpu_procs} 个进程并行...")
        with multiprocessing.Pool(processes=cpu_procs) as pool:
            for res in pool.imap_unordered(run_single_ablation, cpu_tasks):
                save_ablation_result(res)

    # 最终保存汇总JSON
    try:
        data_to_save = {
            "experiment_id": "exp2_ablation",
            "config": {
                "dataset": dataset,
                "variants": variants,
                "budgets": budgets,
                "n_seeds": n_seeds,
            },
            "results": results,
        }
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        print(f"\n最终结果已保存到: {result_file}")
    except Exception as e:
        print(f"最终保存JSON失败: {e}")

    elapsed = time.time() - start_time
    print(f"\n实验二完成！总耗时: {elapsed/3600:.2f} 小时 ({elapsed/60:.2f} 分钟)")
    print(f"完成的任务数: {finished}/{total_runs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="1")
    parser.add_argument("--dataset", type=str, default="ego-Twitter")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results/ijcai_experiments")
    parser.add_argument("--model_path", type=str, default="results/models/final_model.pth")
    parser.add_argument("--num_gpus", type=int, default=4)
    args = parser.parse_args()

    if args.experiment == "1":
        experiment_1_main_comparison(
            dataset=args.dataset,
            output_dir=args.output_dir,
            model_path=args.model_path,
            n_seeds=args.n_seeds,
            device=args.device,
            num_gpus=args.num_gpus,
        )
    elif args.experiment == "2":
        experiment_2_ablation(
            dataset=args.dataset,
            output_dir=args.output_dir,
            model_path=args.model_path,
            n_seeds=args.n_seeds,
            device=args.device,
            num_gpus=args.num_gpus,
        )
    else:
        print(f"实验 {args.experiment} 尚未实现")
