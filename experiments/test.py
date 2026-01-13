"""
测试脚本

在真实数据集上测试训练好的 Semantic-FINDER 模型。
"""
import os
import sys
import argparse
import numpy as np
import torch
import yaml
from tqdm import tqdm
import json
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.semantic_finder import SemanticFINDER
from src.agent.reward import compute_reward, compute_polarization_variance
from src.environment.opinion_env import OpinionDynamicsEnv
from src.environment.data_loader import load_graph


def load_config(config_path: str = "config/config.yaml"):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        config: 配置字典
    """
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    else:
        print(f"警告: 配置文件 {config_path} 不存在，使用默认参数")
        return {}


def test(
    model_path: str,
    dataset_name: str,
    budget: int = 10,
    n_runs: int = 10,
    reward_method: str = "variance",
    device: str = "cpu",
    use_llm: bool = True,
    output_dir: str = "results/test",
    dynamics_model: str = "degroot",
    dynamics_steps: int = 3,
    fj_alpha: float = 0.5,
    max_nodes: int = None,
    sample_method: str = "random",
):
    """
    测试训练好的模型
    
    Args:
        model_path: 模型检查点路径
        dataset_name: 数据集名称（SNAP 数据集）
        budget: 干预预算
        n_runs: 运行次数（取平均）
        reward_method: 奖励计算方法
        device: 设备（"cpu" 或 "cuda"）
        use_llm: 是否使用 LLM 生成干预权重
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    agent = SemanticFINDER(device=device)
    agent.load(model_path)
    agent.eval()  # 设置为评估模式
    
    # 加载数据集（支持合成图用于快速测试）
    print(f"加载数据集: {dataset_name}")
    if dataset_name == "synthetic":
        from src.environment.data_loader import generate_synthetic_graph
        graph = generate_synthetic_graph(n_nodes=1000, graph_type="ba")
        print(f"生成合成图: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
    else:
        graph = load_graph(dataset_name, opinion_init="bimodal")
        print(f"原始图: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        
        # 如果指定了max_nodes，进行子图采样
        if max_nodes is not None and graph.number_of_nodes() > max_nodes:
            from src.environment.data_loader import sample_subgraph
            print(f"采样子图 (max_nodes={max_nodes}, method={sample_method})...")
            graph = sample_subgraph(graph, max_nodes=max_nodes, method=sample_method)
        
        print(f"最终图: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
    
    # 奖励函数
    def reward_fn(state, next_state):
        return compute_reward(state, next_state, method=reward_method)
    
    # 初始化 LLM Controller（如果使用）
    llm_controller = None
    if use_llm:
        try:
            from src.llm.controller import LLMController
            print("正在初始化 LLM Controller（使用Mock模式以避免网络问题）...")
            # 使用Mock模式，避免连接HuggingFace
            llm_controller = LLMController(use_mock=True)
            print("LLM Controller 已初始化（Mock模式）")
        except (ImportError, Exception) as e:
            print(f"警告: LLM Controller 初始化失败 ({e})，使用随机权重")
            use_llm = False
            llm_controller = None
    
    # 运行多次测试（取平均）
    all_results = []
    
    print(f"\n开始测试（运行 {n_runs} 次）...")
    print("-" * 50)
    
    # 记录总开始时间
    total_start_time = time.time()
    
    for run in range(n_runs):
        run_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"运行 {run+1}/{n_runs} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        # 复制图（每次运行使用相同的初始状态）
        test_graph = graph.copy()
        
        # 初始化环境
        print(f"[{time.time()-run_start_time:.2f}s] 初始化环境...")
        env = OpinionDynamicsEnv(
            graph=test_graph,
            budget=budget,
            reward_fn=reward_fn,
            dynamics_model=dynamics_model,
            dynamics_steps=dynamics_steps,
            fj_alpha=fj_alpha,
            device=device,
        )
        
        # 重置环境
        state = env.reset()
        
        # 记录初始极化度
        initial_polarization = compute_polarization_variance(state)
        print(f"[{time.time()-run_start_time:.2f}s] 初始极化度: {initial_polarization:.6f}")
        
        # 运行测试
        done = False
        total_reward = 0.0
        selected_nodes = []
        step_rewards = []
        step_polarizations = [initial_polarization]
        
        step_count = 0
        while not done and step_count < budget:
            step_start_time = time.time()
            print(f"\n  Step {step_count+1}/{budget}:")
            # 构建 mask
            n_nodes = state.number_of_nodes()
            mask = [False] * n_nodes
            for node_idx in selected_nodes:
                if 0 <= node_idx < n_nodes:
                    mask[node_idx] = True
            
            # Agent 选择动作（测试时不探索）
            print(f"    [{time.time()-step_start_time:.2f}s] 计算Q值选择节点...", end="", flush=True)
            sys.stdout.flush()  # 确保立即输出
            action_start = time.time()
            action = agent.select_action(
                graph=state,
                mask=mask,
                epsilon=0.0,  # 测试时不探索
                training=False
            )
            action_time = time.time() - action_start
            print(f" 完成 (耗时 {action_time:.2f}s)", flush=True)
            sys.stdout.flush()
            
            if action == -1:
                print("    警告: 无法选择节点，提前结束")
                break
            
            selected_nodes.append(action)
            print(f"    选择节点: {action}")
            
            # 获取干预权重
            if use_llm and llm_controller is not None:
                print(f"    [{time.time()-step_start_time:.2f}s] LLM生成干预权重...", end="", flush=True)
                sys.stdout.flush()
                weight_start = time.time()
                intervention_weight = llm_controller.get_intervention_weight(action, state)
                weight_time = time.time() - weight_start
                print(f" 完成 (耗时 {weight_time:.2f}s, 权重={intervention_weight:.3f})", flush=True)
                sys.stdout.flush()
            else:
                # 如果没有 LLM，使用随机权重
                intervention_weight = np.random.uniform(0.3, 0.7)
                print(f"    使用随机权重: {intervention_weight:.3f}", flush=True)
                sys.stdout.flush()
            
            # 环境步进
            print(f"    [{time.time()-step_start_time:.2f}s] 环境步进（观点动力学）...", end="", flush=True)
            sys.stdout.flush()
            env_start = time.time()
            next_state, reward, done, info = env.step(action, intervention_weight)
            env_time = time.time() - env_start
            print(f" 完成 (耗时 {env_time:.2f}s)", flush=True)
            sys.stdout.flush()
            
            # 记录统计
            step_rewards.append(reward)
            current_polarization = compute_polarization_variance(next_state)
            step_polarizations.append(current_polarization)
            
            print(f"    奖励: {reward:.6f}")
            print(f"    当前极化度: {current_polarization:.6f} (下降 {initial_polarization-current_polarization:.6f})")
            print(f"    累计奖励: {total_reward+reward:.6f}")
            print(f"    总耗时: {time.time()-step_start_time:.2f}s")
            
            state = next_state
            total_reward += reward
            step_count += 1
        
        # 记录最终极化度
        final_polarization = compute_polarization_variance(state)
        
        # 计算极化度下降百分比
        polarization_reduction = (initial_polarization - final_polarization) / initial_polarization * 100
        
        # 运行总结
        run_time = time.time() - run_start_time
        print(f"\n  运行 {run+1} 完成:")
        print(f"    初始极化度: {initial_polarization:.6f}")
        print(f"    最终极化度: {final_polarization:.6f}")
        print(f"    极化度下降: {polarization_reduction:.2f}%")
        print(f"    总奖励: {total_reward:.6f}")
        print(f"    选择的节点: {selected_nodes}")
        print(f"    运行时间: {run_time:.2f}s ({run_time/60:.2f} 分钟)")
        
        # GPU使用情况
        if device == "cuda" and torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
            print(f"    GPU内存使用: {gpu_mem:.2f} GB")
        
        # 保存结果
        result = {
            'run': run + 1,
            'initial_polarization': float(initial_polarization),
            'final_polarization': float(final_polarization),
            'polarization_reduction': float(polarization_reduction),
            'total_reward': float(total_reward),
            'selected_nodes': selected_nodes,
            'step_rewards': [float(r) for r in step_rewards],
            'step_polarizations': [float(p) for p in step_polarizations],
            'run_time': float(run_time)
        }
        all_results.append(result)
        
        # 显示进度
        elapsed_total = time.time() - total_start_time
        avg_time_per_run = elapsed_total / (run + 1)
        remaining_runs = n_runs - (run + 1)
        estimated_remaining = avg_time_per_run * remaining_runs
        print(f"\n  总进度: {run+1}/{n_runs} ({100*(run+1)/n_runs:.1f}%)")
        print(f"  已用时间: {elapsed_total/60:.2f} 分钟")
        print(f"  预计剩余: {estimated_remaining/60:.2f} 分钟")
    
    # 计算平均结果
    avg_initial_polarization = np.mean([r['initial_polarization'] for r in all_results])
    avg_final_polarization = np.mean([r['final_polarization'] for r in all_results])
    avg_polarization_reduction = np.mean([r['polarization_reduction'] for r in all_results])
    avg_total_reward = np.mean([r['total_reward'] for r in all_results])
    
    # 打印结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    print(f"数据集: {dataset_name}")
    print(f"运行次数: {n_runs}")
    print(f"干预预算: {budget}")
    print(f"奖励方法: {reward_method}")
    print(f"使用 LLM: {use_llm}")
    print("-" * 50)
    print(f"初始极化度: {avg_initial_polarization:.6f}")
    print(f"最终极化度: {avg_final_polarization:.6f}")
    print(f"极化度下降: {avg_polarization_reduction:.2f}%")
    print(f"平均总奖励: {avg_total_reward:.4f}")
    print("=" * 50)
    
    # 保存结果到 JSON
    summary = {
        'dataset': dataset_name,
        'n_runs': n_runs,
        'budget': budget,
        'reward_method': reward_method,
        'use_llm': use_llm,
        'avg_initial_polarization': float(avg_initial_polarization),
        'avg_final_polarization': float(avg_final_polarization),
        'avg_polarization_reduction': float(avg_polarization_reduction),
        'avg_total_reward': float(avg_total_reward),
        'all_results': all_results
    }
    
    output_file = os.path.join(output_dir, f"test_results_{dataset_name}.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n详细结果已保存: {output_file}")
    
    return summary


def main():
    # 先加载配置文件
    config = load_config()
    
    # 从配置文件中提取默认值
    def get_config_value(path, default):
        """从嵌套字典中获取配置值"""
        keys = path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    parser = argparse.ArgumentParser(description="测试 Semantic-FINDER 模型")
    
    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("--budget", type=int, 
                       default=get_config_value("environment.budget", 10), 
                       help="干预预算")
    parser.add_argument("--n_runs", type=int, 
                       default=get_config_value("testing.n_runs", 10), 
                       help="运行次数")
    parser.add_argument("--reward_method", type=str, 
                       default=get_config_value("reward.method", "variance"),
                       choices=["variance", "weighted_disagreement", "echo_chamber"],
                       help="奖励计算方法")
    parser.add_argument("--device", type=str, 
                       default=get_config_value("training.device", "cpu"), 
                       help="设备 (cpu/cuda)")
    parser.add_argument("--use_llm", action="store_true", 
                       help="是否使用 LLM（默认从配置文件读取）")
    parser.add_argument("--output_dir", type=str, 
                       default=get_config_value("testing.output_dir", "results/test"), 
                       help="输出目录")
    parser.add_argument("--dynamics_model", type=str,
                       default=get_config_value("environment.dynamics_model", "degroot"),
                       choices=["degroot", "friedkin_johnsen"],
                       help="观点动力学模型")
    parser.add_argument("--dynamics_steps", type=int,
                       default=get_config_value("environment.dynamics_steps", 3),
                       help="观点演化步数")
    parser.add_argument("--fj_alpha", type=float,
                       default=get_config_value("environment.fj_alpha", 0.5),
                       help="Friedkin-Johnsen 模型的固执度参数")
    parser.add_argument("--max_nodes", type=int, default=None,
                       help="最大节点数（如果数据集太大，会进行子图采样）")
    parser.add_argument("--sample_method", type=str, default="random",
                       choices=["random", "largest_component", "degree"],
                       help="子图采样方法（random/largest_component/degree）")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 如果没有指定 --use_llm，从配置文件读取
    if not args.use_llm:
        args.use_llm = get_config_value("llm.use_llm_in_testing", True)
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        args.device = "cpu"
    
    # 开始测试（过滤掉config参数）
    test_args = vars(args).copy()
    test_args.pop('config', None)  # 移除config参数
    test(**test_args)


if __name__ == "__main__":
    main()



