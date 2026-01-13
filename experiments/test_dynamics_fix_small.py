#!/usr/bin/env python3
"""
小规模快速测试：验证动力学公式修复效果（使用子图采样）
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.data_loader import load_graph, sample_subgraph
from src.environment.opinion_env import OpinionDynamicsEnv
from src.agent.reward import compute_polarization_variance
from src.agent.semantic_finder import SemanticFINDER
from src.baselines.node_selection import (
    select_nodes_random,
    select_nodes_finder,
    select_nodes_semantic_finder,
)

def test_method(method_name, graph, budget, model_path, device="cpu", seed=0):
    """测试单个方法"""
    print(f"\n测试方法: {method_name} (seed={seed})")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 初始化环境
    reward_fn = lambda s, ns: compute_polarization_variance(s) - compute_polarization_variance(ns)
    env = OpinionDynamicsEnv(
        graph=graph.copy(),
        budget=budget,
        reward_fn=reward_fn,
        dynamics_model="degroot",
        dynamics_steps=3,
        device=device,
    )
    
    # 选择节点
    try:
        if method_name == "Random":
            selected_nodes = select_nodes_random(graph, budget, seed=seed)
            intervention_weights = [0.5] * len(selected_nodes)
        elif method_name == "FINDER":
            selected_nodes = select_nodes_finder(
                graph, budget, model_path=model_path, device=device, seed=seed
            )
            intervention_weights = [0.5] * len(selected_nodes)
        elif method_name == "Semantic-FINDER":
            selected_nodes = select_nodes_semantic_finder(
                graph, budget, model_path=model_path, device=device
            )
            intervention_weights = [0.7] * len(selected_nodes)
        else:
            raise ValueError(f"Unknown method: {method_name}")
    except Exception as e:
        print(f"  ❌ 节点选择失败: {e}")
        return None
    
    # 运行实验
    state = env.reset(seed=seed)
    initial_polarization = compute_polarization_variance(state)
    polarization_history = [initial_polarization]
    
    step_count = 0
    for node, weight in zip(selected_nodes, intervention_weights):
        try:
            next_state, reward, done, info = env.step(
                action_node=node,
                intervention_weight=weight,
                alpha=0.5,
                use_new_formula=True
            )
            polarization = compute_polarization_variance(next_state)
            polarization_history.append(polarization)
            state = next_state
            step_count += 1
            if done or step_count >= budget:
                break
        except Exception as e:
            print(f"  ❌ Step {step_count} 失败: {e}")
            break
    
    final_polarization = polarization_history[-1]
    
    # 检查是否有0值
    has_zero = any(abs(p) < 1e-10 for p in polarization_history)
    
    result = {
        "method": method_name,
        "seed": seed,
        "initial_polarization": float(initial_polarization),
        "final_polarization": float(final_polarization),
        "polarization_reduction": float((initial_polarization - final_polarization) / initial_polarization * 100) if initial_polarization > 0 else 0.0,
        "has_zero": has_zero,
        "n_steps": step_count,
    }
    
    print(f"  初始: {initial_polarization:.6f} → 最终: {final_polarization:.6f} (下降 {result['polarization_reduction']:.2f}%)")
    print(f"  是否有0值: {has_zero}, 步数: {step_count}")
    
    return result

def main():
    print("="*60)
    print("动力学公式修复验证测试（小规模）")
    print("="*60)
    
    # 配置 - 使用较小的子图
    dataset = "ego-Twitter"
    max_nodes = 5000  # 使用5000节点的子图
    budget = min(3812, 100)  # 预算限制为100步（快速测试）
    n_seeds = 5
    device = "cpu"
    model_path = "results/models/final_model.pth"
    
    methods = ["Random", "FINDER", "Semantic-FINDER"]
    
    # 加载图并采样
    print(f"\n加载数据集: {dataset}")
    try:
        original_graph = load_graph(dataset, opinion_init="bimodal")
        print(f"原始图: {original_graph.number_of_nodes()} 节点")
        
        if original_graph.number_of_nodes() > max_nodes:
            print(f"采样子图 (max_nodes={max_nodes})...")
            graph = sample_subgraph(original_graph, max_nodes=max_nodes, method="random")
            print(f"子图: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        else:
            graph = original_graph
    except Exception as e:
        print(f"❌ 加载图失败: {e}")
        return
    
    # 运行测试
    all_results = []
    for method in methods:
        print(f"\n{'='*60}")
        print(f"测试方法: {method}")
        print(f"{'='*60}")
        method_results = []
        for seed in range(n_seeds):
            try:
                result = test_method(method, graph, budget, model_path, device, seed)
                if result:
                    method_results.append(result)
                    all_results.append(result)
            except Exception as e:
                print(f"❌ 错误: {method} seed={seed} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 计算平均值
        if method_results:
            avg_initial = np.mean([r["initial_polarization"] for r in method_results])
            avg_final = np.mean([r["final_polarization"] for r in method_results])
            avg_reduction = np.mean([r["polarization_reduction"] for r in method_results])
            has_zeros = any(r["has_zero"] for r in method_results)
            
            print(f"\n{method} 平均结果 (n={len(method_results)}):")
            print(f"  初始极化度: {avg_initial:.6f}")
            print(f"  最终极化度: {avg_final:.6f}")
            print(f"  平均下降: {avg_reduction:.2f}%")
            print(f"  是否有0值: {has_zeros}")
    
    # 保存结果
    if all_results:
        output_file = "test_dynamics_fix_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n结果已保存到: {output_file}")
    
    # 检查目标
    print("\n" + "="*60)
    print("验证检查")
    print("="*60)
    
    # 1. 程序是否报错？
    expected_count = len(methods) * n_seeds
    actual_count = len(all_results)
    print(f"{'✅' if actual_count == expected_count else '❌'} 1. 程序是否报错？", 
          f"否 ({actual_count}/{expected_count} 成功)" if actual_count == expected_count else f"是 ({actual_count}/{expected_count} 成功)")
    
    # 2. 是否有polarization为0的情况？
    has_zeros = any(r["has_zero"] for r in all_results) if all_results else False
    print(f"{'❌' if has_zeros else '✅'} 2. 是否有polarization为0的情况？", "是" if has_zeros else "否")
    
    # 3. Random的效果是否已经变差？
    random_finals = [r["final_polarization"] for r in all_results if r["method"] == "Random"]
    finder_finals = [r["final_polarization"] for r in all_results if r["method"] == "FINDER"]
    if random_finals and finder_finals:
        random_avg = np.mean(random_finals)
        finder_avg = np.mean(finder_finals)
        random_worse = random_avg > finder_avg
        print(f"{'✅' if random_worse else '❌'} 3. Random的效果是否已经变差？", 
              f"是 (Random={random_avg:.6f} > FINDER={finder_avg:.6f})" if random_worse 
              else f"否 (Random={random_avg:.6f} <= FINDER={finder_avg:.6f})")
    
    # 4. Semantic-FINDER是否比FINDER好？
    semantic_finals = [r["final_polarization"] for r in all_results if r["method"] == "Semantic-FINDER"]
    if finder_finals and semantic_finals:
        semantic_avg = np.mean(semantic_finals)
        finder_avg = np.mean(finder_finals)
        semantic_better = semantic_avg < finder_avg
        print(f"{'✅' if semantic_better else '❌'} 4. Semantic-FINDER是否比FINDER好？",
              f"是 (Semantic={semantic_avg:.6f} < FINDER={finder_avg:.6f})" if semantic_better
              else f"否 (Semantic={semantic_avg:.6f} >= FINDER={finder_avg:.6f})")

if __name__ == "__main__":
    main()


