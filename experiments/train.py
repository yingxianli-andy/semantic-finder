"""
训练脚本

实现 Semantic-FINDER 的主训练循环。
"""
import os
import sys
import argparse
import numpy as np
import torch
import yaml
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.semantic_finder import SemanticFINDER
from src.agent.reward import compute_reward
from src.agent.replay_buffer import ReplayBuffer
from src.environment.opinion_env import OpinionDynamicsEnv
from src.environment.data_loader import generate_synthetic_graph


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


def train(
    num_episodes: int = 10000,
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
    device: str = "cpu",
    use_llm: bool = False,
    dynamics_model: str = "degroot",
    dynamics_steps: int = 3,
    fj_alpha: float = 0.5,
):
    """
    训练 Semantic-FINDER Agent
    
    Args:
        num_episodes: 训练轮数
        budget: 每个 episode 的干预预算（最多选择 K 个节点）
        n_nodes: 合成图的节点数
        graph_type: 图类型（"BA" 表示 Barabasi-Albert）
        reward_method: 奖励计算方法
        learning_rate: 学习率
        batch_size: 批次大小
        gamma: 折扣因子
        epsilon_start: 初始探索率
        epsilon_end: 最终探索率
        epsilon_decay: 探索率衰减
        replay_buffer_size: 经验回放缓冲区大小
        save_frequency: 保存频率（每 N 个 episode）
        checkpoint_dir: 检查点保存目录
        device: 设备（"cpu" 或 "cuda"）
        use_llm: 训练时是否使用 LLM（建议 False，使用随机权重加速训练）
    """
    # 创建保存目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
    
    # 可选：初始化 LLM Controller（训练阶段通常不用）
    llm_controller = None
    if use_llm:
        try:
            from src.llm.controller import LLMController

            llm_controller = LLMController()
            print("训练阶段启用 LLM Controller")
        except Exception as exc:  # noqa: BLE001
            print(f"警告: LLM Controller 初始化失败，改用随机权重。原因: {exc}")
            llm_controller = None
            use_llm = False

    # 奖励函数
    def reward_fn(state, next_state):
        return compute_reward(state, next_state, method=reward_method)
    
    # 训练统计
    episode_rewards = []
    episode_losses = []
    epsilon = epsilon_start
    
    print(f"开始训练...")
    print(f"设备: {device}")
    print(f"图类型: {graph_type}, 节点数: {n_nodes}")
    print(f"预算: {budget}, 奖励方法: {reward_method}")
    print(f"总轮数: {num_episodes}")
    print("-" * 50)
    
    # 主训练循环
    for episode in tqdm(range(num_episodes), desc="Training"):
        # 生成新的合成图（每个 episode 使用不同的图）
        graph = generate_synthetic_graph(n_nodes=n_nodes, graph_type=graph_type)
        
        # 初始化环境
        env = OpinionDynamicsEnv(
            graph=graph,
            budget=budget,
            reward_fn=reward_fn,
            dynamics_model=dynamics_model,
            dynamics_steps=dynamics_steps,
            fj_alpha=fj_alpha,
            device=device,
        )
        
        # 重置环境
        state = env.reset()
        done = False
        episode_reward = 0.0
        selected_nodes = []  # 记录已选节点（用于 masking）
        
        # Episode 内的步骤循环
        step_count = 0
        while not done and step_count < budget:
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
            
            # 记录已选节点
            selected_nodes.append(action)
            
            # 获取干预权重
            # 训练时：使用随机权重或简单启发式（加速训练）
            # 测试时：使用 LLM（由上交爷提供）
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
            if replay_buffer.is_ready(batch_size):
                loss = agent.update(replay_buffer, batch_size=batch_size)
                episode_losses.append(loss)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            step_count += 1
        
        # 记录统计信息
        episode_rewards.append(episode_reward)
        
        # 衰减探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 定期保存检查点
        if (episode + 1) % save_frequency == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode+1}.pth")
            agent.save(checkpoint_path)
            
            # 打印统计信息
            avg_reward = np.mean(episode_rewards[-save_frequency:])
            avg_loss = np.mean(episode_losses[-save_frequency:]) if episode_losses else 0.0
            print(f"\nEpisode {episode+1}/{num_episodes}")
            print(f"  平均奖励: {avg_reward:.4f}")
            print(f"  平均损失: {avg_loss:.4f}")
            print(f"  探索率: {epsilon:.4f}")
            print(f"  检查点已保存: {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    agent.save(final_model_path)
    print(f"\n训练完成！最终模型已保存: {final_model_path}")
    
    # 返回训练统计
    return {
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'final_epsilon': epsilon
    }


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
    
    parser = argparse.ArgumentParser(description="训练 Semantic-FINDER")
    
    # 训练参数（从配置文件读取默认值）
    parser.add_argument("--num_episodes", type=int, 
                       default=get_config_value("training.num_episodes", 10000), 
                       help="训练轮数")
    parser.add_argument("--budget", type=int, 
                       default=get_config_value("environment.budget", 10), 
                       help="干预预算")
    parser.add_argument("--n_nodes", type=int, 
                       default=get_config_value("data.synthetic.n_nodes", 50), 
                       help="合成图节点数")
    parser.add_argument("--graph_type", type=str, 
                       default=get_config_value("data.synthetic.graph_type", "BA"), 
                       help="图类型")
    parser.add_argument("--reward_method", type=str, 
                       default=get_config_value("reward.method", "variance"),
                       choices=["variance", "weighted_disagreement", "echo_chamber"],
                       help="奖励计算方法")
    
    # 模型参数
    parser.add_argument("--learning_rate", type=float, 
                       default=get_config_value("agent.learning_rate", 0.001), 
                       help="学习率")
    parser.add_argument("--batch_size", type=int, 
                       default=get_config_value("agent.batch_size", 32), 
                       help="批次大小")
    parser.add_argument("--gamma", type=float, 
                       default=get_config_value("agent.gamma", 0.99), 
                       help="折扣因子")
    
    # 探索参数
    parser.add_argument("--epsilon_start", type=float, 
                       default=get_config_value("exploration.epsilon_start", 1.0), 
                       help="初始探索率")
    parser.add_argument("--epsilon_end", type=float, 
                       default=get_config_value("exploration.epsilon_end", 0.01), 
                       help="最终探索率")
    parser.add_argument("--epsilon_decay", type=float, 
                       default=get_config_value("exploration.epsilon_decay", 0.995), 
                       help="探索率衰减")
    
    # 其他参数
    parser.add_argument("--replay_buffer_size", type=int, 
                       default=get_config_value("agent.replay_buffer_size", 10000), 
                       help="经验回放缓冲区大小")
    parser.add_argument("--save_frequency", type=int, 
                       default=get_config_value("training.save_frequency", 100), 
                       help="保存频率")
    parser.add_argument("--checkpoint_dir", type=str, 
                       default=get_config_value("training.checkpoint_dir", "results/models"), 
                       help="检查点目录")
    parser.add_argument("--device", type=str, 
                       default=get_config_value("training.device", "cpu"), 
                       help="设备 (cpu/cuda)")
    parser.add_argument("--use_llm", action="store_true", 
                       help="训练时是否使用 LLM")
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
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU")
        args.device = "cpu"
    
    # 开始训练（过滤掉config参数，因为train函数不接受它）
    train_args = vars(args).copy()
    train_args.pop('config', None)  # 移除config参数
    train(**train_args)


if __name__ == "__main__":
    main()



