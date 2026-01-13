"""
训练脚本（带 TensorBoard 支持）

这个脚本扩展了原始训练脚本，添加了 TensorBoard 日志记录功能，
方便监控训练过程中的关键指标。
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.agent.semantic_finder import SemanticFINDER
from src.agent.reward import compute_reward
from src.agent.replay_buffer import ReplayBuffer
from src.environment.opinion_env import OpinionDynamicsEnv
from src.environment.data_loader import generate_synthetic_graph
from src.utils.graph_utils import get_graph_statistics
from src.utils.metrics import compute_polarization_variance


def train(
    num_episodes: int = 1000,
    budget: int = 10,
    n_nodes: int = 50,
    graph_type: str = "BA",
    reward_method: str = "variance",
    learning_rate: float = 0.001,
    batch_size: int = 32,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    replay_buffer_size: int = 10000,
    save_frequency: int = 100,
    checkpoint_dir: str = "results/models",
    log_dir: str = "results/logs",
    device: str = "cpu",
    use_llm: bool = False,
    dynamics_model: str = "degroot",
    dynamics_steps: int = 3,
    fj_alpha: float = 0.5,
):
    """
    训练 Semantic-FINDER Agent（带 TensorBoard 日志）
    
    Args:
        num_episodes: 训练轮数
        budget: 每个 episode 的干预预算
        n_nodes: 合成图的节点数
        graph_type: 图类型（"BA" 或 "ER"）
        reward_method: 奖励计算方法（"variance", "weighted_disagreement", "echo_chamber"）
        learning_rate: 学习率
        batch_size: 批次大小
        gamma: 折扣因子
        epsilon_start: 初始探索率
        epsilon_end: 最终探索率
        epsilon_decay: 探索率衰减率
        replay_buffer_size: 经验回放缓冲区大小
        save_frequency: 保存模型的频率（每多少轮保存一次）
        checkpoint_dir: 模型保存目录
        log_dir: TensorBoard 日志目录
        device: 设备（"cpu" 或 "cuda"）
        use_llm: 是否使用 LLM 生成干预权重
        dynamics_model: 观点动力学模型（"degroot" 或 "friedkin_johnsen"）
        dynamics_steps: 每次 step 中动力学演化的步数
        fj_alpha: Friedkin-Johnsen 模型的固执度参数
    """
    # 创建保存目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # 初始化 Agent
    agent = SemanticFINDER(
        input_dim=2,  # 度数 + 观点值
        hidden_dim=128,
        output_dim=128,
        num_layers=2,
        encoder_type="GraphSAGE",
        learning_rate=learning_rate,
        gamma=gamma,
        device=device
    )
    
    # 初始化经验回放缓冲区
    replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
    
    # 初始化 LLM Controller（如果使用）
    llm_controller = None
    if use_llm:
        try:
            from src.llm.controller import LLMController
            llm_controller = LLMController()
            print("LLM Controller 已初始化")
        except ImportError:
            print("警告: 无法导入 LLM Controller，将使用随机权重")
            use_llm = False
    
    # 奖励函数
    def reward_fn(state, next_state):
        return compute_reward(state, next_state, method=reward_method)
    
    # 训练统计
    episode_rewards = []
    episode_losses = []
    episode_polarizations = []
    epsilon = epsilon_start
    
    print(f"开始训练...")
    print(f"设备: {device}")
    print(f"图类型: {graph_type}, 节点数: {n_nodes}")
    print(f"预算: {budget}, 奖励方法: {reward_method}")
    print(f"动力学模型: {dynamics_model}, 演化步数: {dynamics_steps}")
    if dynamics_model == "friedkin_johnsen":
        print(f"FJ Alpha: {fj_alpha}")
    print(f"总轮数: {num_episodes}, 探索率: {epsilon_start}->{epsilon_end}")
    print("-" * 50)
    
    # 主训练循环
    for episode in tqdm(range(num_episodes), desc="训练进度"):
        # 生成新的合成图
        graph = generate_synthetic_graph(n_nodes=n_nodes, graph_type=graph_type)
        
        # 初始化环境
        env = OpinionDynamicsEnv(
            graph=graph,
            budget=budget,
            reward_fn=reward_fn,
            dynamics_model=dynamics_model,
            dynamics_steps=dynamics_steps,
            fj_alpha=fj_alpha,
        )
        
        # 重置环境
        state = env.reset()
        done = False
        episode_reward = 0.0
        selected_nodes = []
        
        # 记录初始极化度
        initial_polarization = compute_polarization_variance(state)
        
        # Episode 内的步骤循环
        step_count = 0
        while not done and step_count < budget * 2:  # 防止无限循环
            # 构建 mask（已选节点不能重复选）
            n_nodes = state.number_of_nodes()
            mask = [False] * n_nodes
            for node_idx in selected_nodes:
                if 0 <= node_idx < n_nodes:
                    mask[node_idx] = True
            
            # Agent 选择动作
            action = agent.select_action(
                graph=state,
                mask=mask,
                epsilon=epsilon,
                training=True
            )
            
            if action == -1:  # 无效动作
                break
                
            selected_nodes.append(action)
            
            # 获取干预权重
            if use_llm and llm_controller is not None:
                intervention_weight = llm_controller.get_intervention_weight(action, state)
            else:
                # 训练时使用随机权重或简单启发式
                intervention_weight = np.random.uniform(0.3, 0.7)
            
            # 环境步进
            next_state, reward, done, info = env.step(action, intervention_weight)
            
            # 存储经验
            replay_buffer.push(state, action, reward, next_state, done)
            
            # 更新 Agent（如果缓冲区有足够经验）
            if len(replay_buffer) > batch_size:
                loss = agent.update(replay_buffer, batch_size=batch_size)
                episode_losses.append(loss)
                
                # 记录到 TensorBoard
                writer.add_scalar('train/loss', loss, episode * budget + step_count)
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # 记录到 TensorBoard
            writer.add_scalar('train/step_reward', reward, episode * budget + step_count)
            writer.add_scalar('train/epsilon', epsilon, episode * budget + step_count)
        
        # 计算最终极化度
        final_polarization = compute_polarization_variance(state)
        polarization_reduction = (initial_polarization - final_polarization) / initial_polarization if initial_polarization > 0 else 0.0
        
        # 记录统计信息
        episode_rewards.append(episode_reward)
        episode_polarizations.append(final_polarization)
        
        # 记录到 TensorBoard
        writer.add_scalar('train/episode_reward', episode_reward, episode)
        writer.add_scalar('train/episode_length', step_count, episode)
        writer.add_scalar('train/initial_polarization', initial_polarization, episode)
        writer.add_scalar('train/final_polarization', final_polarization, episode)
        writer.add_scalar('train/polarization_reduction', polarization_reduction, episode)
        
        # 衰减探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 定期保存检查点
        if (episode + 1) % save_frequency == 0 or episode == num_episodes - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode+1}.pth")
            agent.save(checkpoint_path)
            
            # 打印统计信息
            avg_reward = np.mean(episode_rewards[-save_frequency:])
            avg_loss = np.mean(episode_losses[-save_frequency:]) if episode_losses else 0.0
            avg_polarization = np.mean(episode_polarizations[-save_frequency:])
            
            print(f"\nEpisode {episode+1}/{num_episodes}")
            print(f"  平均奖励: {avg_reward:.4f}")
            print(f"  平均损失: {avg_loss:.4f}" if episode_losses else "  平均损失: N/A")
            print(f"  最终极化度: {final_polarization:.4f} (初始: {initial_polarization:.4f}, 降低: {polarization_reduction*100:.1f}%)")
            print(f"  探索率: {epsilon:.4f}")
            print(f"  检查点已保存: {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    agent.save(final_model_path)
    print(f"\n训练完成！最终模型已保存: {final_model_path}")
    
    # 关闭 TensorBoard writer
    writer.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'episode_polarizations': episode_polarizations,
        'final_epsilon': epsilon
    }


def main():
    parser = argparse.ArgumentParser(description="训练 Semantic-FINDER（带 TensorBoard 支持）")
    
    # 训练参数
    parser.add_argument("--num_episodes", type=int, default=1000, help="训练轮数")
    parser.add_argument("--budget", type=int, default=10, help="干预预算")
    parser.add_argument("--n_nodes", type=int, default=50, help="合成图节点数")
    parser.add_argument("--graph_type", type=str, default="BA", help="图类型（BA 或 ER）")
    parser.add_argument("--reward_method", type=str, default="variance",
                        choices=["variance", "weighted_disagreement", "echo_chamber"],
                        help="奖励计算方法")
    
    # 模型参数
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    
    # 探索参数
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="初始探索率")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="最终探索率")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="探索率衰减")
    
    # 其他参数
    parser.add_argument("--replay_buffer_size", type=int, default=10000, help="经验回放缓冲区大小")
    parser.add_argument("--save_frequency", type=int, default=100, help="保存频率")
    parser.add_argument("--checkpoint_dir", type=str, default="results/models", help="检查点目录")
    parser.add_argument("--log_dir", type=str, default="results/logs", help="TensorBoard 日志目录")
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu/cuda)")
    parser.add_argument("--use_llm", action="store_true", help="是否使用 LLM 生成干预权重")
    
    # 动力学模型参数
    parser.add_argument("--dynamics_model", type=str, default="degroot",
                        choices=["degroot", "friedkin_johnsen"],
                        help="观点动力学模型")
    parser.add_argument("--dynamics_steps", type=int, default=3, help="观点演化步数")
    parser.add_argument("--fj_alpha", type=float, default=0.5,
                        help="Friedkin-Johnsen 模型的固执度参数")
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        args.device = "cpu"
    
    # 开始训练
    train(**vars(args))


if __name__ == "__main__":
    main()