# Semantic-FINDER IJCAI 实验开发任务文档

**致哈工爷：**  
本文档详细列出了需要为 IJCAI 投稿实验编写的所有代码和需要执行的所有实验。请按照文档逐项完成。

---

## 一、公式理解确认

### 1.1 核心公式（手册要求）

实验执行手册中的观点更新公式：

$$x_i(t+1) = \alpha \cdot x_i(t) + (1-\alpha) \cdot \left( \sum_{j \in N(i)} w_{ij} x_j(t) + \text{Intervention}_i \right)$$

**参数说明**：
- $x_i(t)$: 节点 $i$ 在时刻 $t$ 的观点值（范围：-1 到 1，或 0 到 1）
- $\alpha$: **顽固度（Stubbornness）参数**，范围 0-1
  - $\alpha$ 接近 0：耳根子软，容易受邻居影响
  - $\alpha$ 接近 1：极其固执，很难改变观点
- $w_{ij}$: 节点 $i$ 对邻居 $j$ 的权重（通常归一化为 $1/deg(i)$）
- $\text{Intervention}_i$: 干预项，当节点被选中时，由 LLM 决定权重 $\omega$

**与当前实现的差异**：
- 当前 DeGroot 模型：$x(t+1) = A_{norm} \cdot x(t)$（没有 $\alpha$ 和 Intervention）
- 当前 FJ 模型：$x(t+1) = \alpha \cdot A_{norm} \cdot x(t) + (1-\alpha) \cdot x(0)$（没有 Intervention）
- **需要修改**：在动力学更新时支持 Intervention 项的添加

### 1.2 干预项（Intervention）的实现

当节点 $i$ 被选中干预时：
- 当前观点：$x_i(t)$
- 干预权重：$\omega$（由 LLM 提供，范围 0.0-1.0）
- 干预后的观点：$x_i(t) = x_i(t) \cdot (1 - \omega)$ （向 0 拉近）

**注意**：Intervention 应该在动力学更新**之前**或**之中**应用。建议在动力学更新的公式中直接加入干预项。

---

## 二、需要修改的现有代码

### 2.1 修改 `src/environment/dynamics.py`

**任务**：在 `update_opinions` 函数中支持新的公式，包含：
1. $\alpha$ 顽固度参数（用于 DeGroot 变体）
2. Intervention 项的支持

**需要添加的参数**：
```python
def update_opinions(
    graph: nx.Graph,
    model: DynamicsModelType = "degroot",
    steps: int = 3,
    alpha: float = 0.5,  # 顽固度参数（现在用于 DeGroot 变体）
    device: str = "cpu",
    intervention_dict: Optional[Dict[int, float]] = None,  # 新增：{节点ID: 干预权重}
) -> nx.Graph:
```

**修改后的公式实现**（对于 DeGroot 变体）：
```python
# 如果使用新公式（带 alpha 和 Intervention）
if intervention_dict is not None:
    intervention_vector = np.zeros(n, dtype=np.float32)
    for node_id, weight in intervention_dict.items():
        intervention_vector[node_id] = weight
    
    for _ in range(steps):
        # x(t+1) = alpha * x(t) + (1-alpha) * (A_norm * x(t) + Intervention)
        neighbor_influence = a_norm @ x
        total_influence = neighbor_influence + intervention_vector
        x = alpha * x + (1.0 - alpha) * total_influence
else:
    # 原有逻辑（向后兼容）
    ...
```

**要求**：
- 保持向后兼容（`intervention_dict=None` 时使用原有逻辑）
- 支持 GPU 加速（修改 `_update_opinions_torch` 函数）

### 2.2 修改 `src/environment/opinion_env.py`

**任务**：在 `step` 方法中支持新的动力学公式

**需要修改的地方**：
```python
def step(
    self,
    action_node: int,
    intervention_weight: float,
    alpha: Optional[float] = None,  # 新增：允许动态设置顽固度
) -> Tuple[nx.Graph, float, bool, Dict]:
    """
    Args:
        action_node: 被选中的节点索引
        intervention_weight: 干预权重（由 LLM 提供）
        alpha: 顽固度参数（如果提供，覆盖初始化时的值）
    """
    # 1. 应用干预（修改节点观点）
    # 2. 构建 intervention_dict = {action_node: intervention_weight}
    # 3. 调用 update_opinions，传入 intervention_dict 和 alpha
```

**注意**：如果使用新公式，步骤1（应用干预）可能需要在动力学公式内部完成，而不是提前修改观点值。

---

## 三、需要新编写的代码

### 3.1 基线方法实现模块

**文件**：`src/baselines/node_selection.py`

**需要实现的类/函数**：

#### 3.1.1 Random 选择器
```python
def select_nodes_random(graph: nx.Graph, k: int, seed: Optional[int] = None) -> List[int]:
    """
    随机选择 K 个节点
    
    Args:
        graph: 图对象
        k: 需要选择的节点数量
        seed: 随机种子
    
    Returns:
        selected_nodes: 选中的节点索引列表
    """
    pass
```

#### 3.1.2 High-Degree 选择器
```python
def select_nodes_high_degree(graph: nx.Graph, k: int) -> List[int]:
    """
    选择度数最大的 K 个节点（大V策略）
    
    Args:
        graph: 图对象
        k: 需要选择的节点数量
    
    Returns:
        selected_nodes: 选中的节点索引列表（按度数从大到小排序）
    """
    pass
```

#### 3.1.3 PageRank 选择器
```python
def select_nodes_pagerank(graph: nx.Graph, k: int, alpha: float = 0.85) -> List[int]:
    """
    选择 PageRank 值最高的 K 个节点
    
    Args:
        graph: 图对象
        k: 需要选择的节点数量
        alpha: PageRank 阻尼系数（默认0.85）
    
    Returns:
        selected_nodes: 选中的节点索引列表（按 PageRank 值从大到小排序）
    """
    # 使用 networkx.pagerank 函数
    pass
```

#### 3.1.4 Original FINDER 选择器
```python
def select_nodes_finder(
    graph: nx.Graph, 
    k: int, 
    model_path: str = "results/models/final_model.pth",
    device: str = "cpu"
) -> List[int]:
    """
    使用预训练的 RL 模型（Original FINDER，无语义）选择节点
    
    Args:
        graph: 图对象
        k: 需要选择的节点数量
        model_path: 模型文件路径
        device: 设备（cpu 或 cuda）
    
    Returns:
        selected_nodes: 选中的节点索引列表
    """
    # 加载模型，调用 model.select_action，循环 K 次
    pass
```

#### 3.1.5 Semantic-FINDER 选择器（完整版）
```python
def select_nodes_semantic_finder(
    graph: nx.Graph,
    k: int,
    model_path: str = "results/models/final_model.pth",
    llm_controller,  # LLMController 实例
    device: str = "cpu"
) -> List[int]:
    """
    使用 Semantic-FINDER（RL 选点 + LLM 权重）选择节点
    
    注意：这个方法会在选点的同时获取 LLM 权重，但权重不在这里返回
    （权重在实验循环中使用）
    
    Args:
        graph: 图对象
        k: 需要选择的节点数量
        model_path: 模型文件路径
        llm_controller: LLMController 实例
        device: 设备（cpu 或 cuda）
    
    Returns:
        selected_nodes: 选中的节点索引列表
    """
    pass
```

### 3.2 LLM 权重生成模块（扩展）

**文件**：`src/llm/strategy.py`

**需要实现的函数**：

#### 3.2.1 Conservative 策略（温和派）
```python
def get_weight_conservative(node_id: int, graph: nx.Graph, subgraph: Optional[nx.Graph] = None) -> float:
    """
    温和派策略：总是给低权重，慢慢感化
    
    Args:
        node_id: 节点ID
        graph: 完整图对象
        subgraph: 节点周围的子图（可选）
    
    Returns:
        weight: 干预权重（0.3 到 0.4 之间）
    """
    import random
    return 0.3 + 0.1 * random.random()
```

#### 3.2.2 Aggressive 策略（激进派）
```python
def get_weight_aggressive(node_id: int, graph: nx.Graph, subgraph: Optional[nx.Graph] = None) -> float:
    """
    激进派策略：总是给高权重，强力洗脑
    
    Args:
        node_id: 节点ID
        graph: 完整图对象
        subgraph: 节点周围的子图（可选）
    
    Returns:
        weight: 干预权重（0.8 到 1.0 之间）
    """
    import random
    return 0.8 + 0.2 * random.random()
```

#### 3.2.3 Adaptive 策略（聪明派 - 我们的方法）
```python
def get_weight_adaptive(node_id: int, graph: nx.Graph, subgraph: Optional[nx.Graph] = None) -> float:
    """
    自适应策略：根据周围邻居的方差动态调整权重
    
    逻辑：
    - 如果邻居方差大（吵得凶）：给高权重（1.0），强力介入
    - 如果邻居方差小（很和平）：给低权重（0.2），轻轻推一下
    
    Args:
        node_id: 节点ID
        graph: 完整图对象
        subgraph: 节点周围的子图（如果为None，从完整图中提取）
    
    Returns:
        weight: 干预权重（0.2 到 1.0 之间）
    """
    # 1. 获取节点邻居
    # 2. 计算邻居观点值的方差
    # 3. 根据方差决定权重
    # var_high_threshold = 0.5  # 可调参数
    # var_low_threshold = 0.1
    pass
```

### 3.3 噪声注入模块

**文件**：`src/utils/noise.py`

```python
def add_gaussian_noise(value: float, noise_std: float, clip_range: Optional[Tuple[float, float]] = None) -> float:
    """
    在值上添加高斯噪声
    
    Args:
        value: 原始值
        noise_std: 噪声标准差
        clip_range: 裁剪范围（如 (0.0, 1.0)）
    
    Returns:
        noisy_value: 添加噪声后的值
    """
    import numpy as np
    noisy = value + np.random.normal(0, noise_std)
    if clip_range is not None:
        noisy = np.clip(noisy, clip_range[0], clip_range[1])
    return float(noisy)
```

### 3.4 主控实验脚本

**文件**：`experiments/run_ijcai_experiments.py`

**核心结构**：
```python
"""
IJCAI 实验主控脚本

运行所有4组实验，生成结果文件。
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm

# 导入所需模块
from src.environment.data_loader import load_graph
from src.environment.opinion_env import OpinionDynamicsEnv
from src.agent.reward import compute_reward
from src.baselines.node_selection import (
    select_nodes_random,
    select_nodes_high_degree,
    select_nodes_pagerank,
    select_nodes_finder,
    select_nodes_semantic_finder,
)
from src.llm.strategy import (
    get_weight_conservative,
    get_weight_aggressive,
    get_weight_adaptive,
)
from src.utils.noise import add_gaussian_noise


def run_single_experiment(
    graph: nx.Graph,
    method: str,
    budget: int,
    seed: int,
    config: Dict[str, Any],  # 包含 alpha, noise_std, strategy 等参数
) -> Dict[str, Any]:
    """
    运行单次实验（单个 seed）
    
    Returns:
        result: 包含 polarization_history, final_score, selected_nodes 等
    """
    # 固定随机种子
    np.random.seed(seed)
    random.seed(seed)
    
    # 初始化环境
    # 选择节点
    # 循环干预
    # 记录极化度历史
    pass


def experiment_1_main_comparison():
    """实验一：SOTA 性能对比"""
    pass


def experiment_2_ablation():
    """实验二：消融实验"""
    pass


def experiment_3_robustness():
    """实验三：鲁棒性测试"""
    pass


def experiment_4_strategies():
    """实验四：LLM 策略多样性"""
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, choices=["1", "2", "3", "4", "all"])
    parser.add_argument("--dataset", type=str, default="twitter_combined.txt")
    parser.add_argument("--output_dir", type=str, default="results/ijcai_experiments")
    args = parser.parse_args()
    
    # 运行指定实验
```

---

## 四、需要执行的实验详情

### 实验一：SOTA 性能对比 (Main Result)

**目的**：证明 Semantic-FINDER 效果最好，收敛最快。

**需要实现的方法对比**：
1. **Random**: 随机选 K 个节点，权重固定 1.0
2. **High-Degree**: 度数最大 K 个节点，权重固定 1.0
3. **PageRank**: PageRank 最高 K 个节点，权重固定 1.0
4. **Original FINDER**: RL 选点，权重固定 0.5（无语义）
5. **Semantic-FINDER (Ours)**: RL 选点 + LLM 动态权重

**参数组合**：
- **数据集**: `twitter_combined.txt`（完整数据集，76K节点）
- **Budget (K)**: [1%, 2%, 5%] 的节点数
  - 1% ≈ 762 个节点
  - 2% ≈ 1525 个节点
  - 5% ≈ 3812 个节点
- **随机种子 (Seeds)**: 50 个固定种子（0-49）
- **Metric**: 记录每一步的 Network Variance（方差）作为极化度指标

**输出文件**：`results/ijcai_experiments/results_main_comparison.json`

**数据结构**：
```json
{
  "experiment_id": "exp1_main_comparison",
  "config": {
    "dataset": "twitter_combined.txt",
    "methods": ["Random", "High-Degree", "PageRank", "FINDER", "Semantic-FINDER"],
    "budgets": [762, 1525, 3812],
    "n_seeds": 50
  },
  "results": {
    "Random": {
      "budget_762": [
        {"seed": 0, "polarization_history": [0.8, 0.7, ...], "final_score": 0.1},
        ...
      ],
      "budget_1525": [...],
      "budget_3812": [...]
    },
    ...
  }
}
```

### 实验二：消融实验 (Ablation Study)

**目的**：证明 "RL选点" 和 "LLM给权重" 缺一不可。

**需要实现的变体**：
1. **w/o LLM (去掉大模型)**:
   - 选点：使用 RL (FINDER)
   - 权重：固定为随机数 `random.uniform(0, 1)`

2. **w/o RL (去掉结构学习)**:
   - 选点：随机选择 (Random)
   - 权重：使用 LLM 生成的完美权重（使用 Adaptive 策略）

3. **Full Model (Ours)**:
   - 选点：RL
   - 权重：LLM

**参数**：
- **数据集**: `twitter_combined.txt`
- **Budget**: 5% (3812 个节点)
- **Seeds**: 50 个

**输出文件**：`results/ijcai_experiments/results_ablation.json`

### 实验三：鲁棒性测试 (Robustness)

**目的**：证明环境变化或模型不准时，系统依然稳定。

#### 3.1 变量 A：LLM 的"幻觉"噪声 (Semantic Noise)

**参数**：
- **噪声标准差 `noise_std`**: [0.0, 0.2, 0.5, 0.8]
- **数据集**: `twitter_combined.txt`
- **Budget**: 5% (3812 个节点)
- **Seeds**: 50 个
- **方法**: Semantic-FINDER (完整版)

**实现逻辑**：
```python
# 在获取 LLM 权重后，添加噪声
llm_weight = llm_controller.get_intervention_weight(node_id, graph)
actual_weight = add_gaussian_noise(llm_weight, noise_std, clip_range=(0.0, 1.0))
```

**输出文件**：`results/ijcai_experiments/results_robustness_noise.json`

#### 3.2 变量 B：人群顽固度 (User Stubbornness)

**参数**：
- **顽固度 `alpha`**: [0.1, 0.3, 0.5, 0.7, 0.9]
  - 0.1 = 耳根子软，听风就是雨
  - 0.9 = 极其固执，不论怎么干预都很难改变
- **数据集**: `twitter_combined.txt`
- **Budget**: 5% (3812 个节点)
- **Seeds**: 50 个
- **方法对比**: Random vs. Semantic-FINDER

**实现逻辑**：
- 在 `OpinionDynamicsEnv.step()` 中传入 `alpha` 参数
- 使用新的动力学公式（包含 Intervention）

**输出文件**：`results/ijcai_experiments/results_robustness_alpha.json`

### 实验四：LLM 策略多样性 (Strategy Analysis)

**目的**：展示不同 LLM Prompt 策略带来的社会治理效果差异。

**需要实现的策略**：
1. **Conservative (温和派)**: `weight = 0.3 + 0.1 * random()`
2. **Aggressive (激进派)**: `weight = 0.8 + 0.2 * random()`
3. **Adaptive (聪明派 - 我们的方法)**: 根据邻居方差动态调整

**参数**：
- **数据集**: `twitter_combined.txt`
- **Budget**: 5% (3812 个节点)
- **Seeds**: 50 个
- **选点方法**: 统一使用 RL (FINDER)

**输出文件**：`results/ijcai_experiments/results_strategies.json`

---

## 五、实现细节和注意事项

### 5.1 Seed 固定和可复现性

**关键要求**：
- 所有实验必须使用**固定的随机种子列表**（0-49）
- 对于同一个 seed，不同方法应该使用**相同的初始图状态**
- 在实验开始时，调用 `np.random.seed(seed)`, `random.seed(seed)`, `torch.manual_seed(seed)`

**实现示例**：
```python
def run_experiment_with_seed(seed: int):
    # 固定所有随机源
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    # 加载图并初始化观点（使用相同的 seed）
    graph = load_graph("data/raw/twitter_combined.txt", seed=seed)
    initialize_opinions(graph, seed=seed)
    
    # 运行实验...
```

### 5.2 极化度记录

**要求**：
- 每个 Step 都要记录当前的极化度值
- 使用 `compute_polarization_variance` 函数（或指定其他方法）
- 初始状态也要记录（Step 0）

**数据结构**：
```python
polarization_history = [initial_polarization]  # Step 0
for step in range(budget):
    # ... 执行干预 ...
    current_polarization = compute_polarization_variance(graph)
    polarization_history.append(current_polarization)
```

### 5.3 并行计算优化

**建议**：
- 使用 `multiprocessing` 或 `joblib` 并行运行不同的 Seed
- 每个进程运行一个 Seed，避免共享状态
- 注意：不同 Seed 之间完全独立，可以并行

**示例**：
```python
from multiprocessing import Pool
from functools import partial

def run_single_seed(seed, config):
    """运行单个 seed 的实验"""
    return run_single_experiment(..., seed=seed, ...)

def run_experiment_batch(config, n_seeds=50):
    seeds = list(range(n_seeds))
    with Pool(processes=8) as pool:  # 使用8个进程
        results = pool.map(partial(run_single_seed, config=config), seeds)
    return results
```

### 5.4 结果保存格式

**要求**：
- 每次实验都要保存**完整的详细数据**（不要只保存平均值）
- 每个 Seed 的结果都要单独保存（包含 `polarization_history`）
- 保存配置信息（参数组合）

**JSON 结构示例**：
```json
{
  "experiment_id": "exp1_baseline_01",
  "config": {
    "method": "Random",
    "budget": 762,
    "dataset": "twitter_combined.txt",
    "alpha": 0.5,
    "noise_std": 0.0,
    "strategy": null
  },
  "results": [
    {
      "seed": 0,
      "polarization_history": [0.6429, 0.6, 0.55, ...],
      "final_score": 0.1,
      "selected_nodes": [123, 456, ...],
      "step_rewards": [0.0429, 0.05, ...]
    },
    {
      "seed": 1,
      ...
    }
  ],
  "statistics": {
    "mean_final_score": 0.105,
    "std_final_score": 0.015,
    "mean_polarization_reduction": 0.84
  }
}
```

### 5.5 性能优化建议

**针对大规模图（76K节点）**：
1. 使用 GPU 加速观点动力学计算
2. 子图采样：对于某些基线方法（如 PageRank），可以考虑在子图上计算
3. 批量处理：如果可能，批量计算多个节点的特征

**监控指标**：
- 每个实验的运行时间
- 内存使用情况
- GPU 利用率（如果使用）

---

## 六、交付清单

### 6.1 代码文件

- [ ] `src/environment/dynamics.py` (修改：支持新公式)
- [ ] `src/environment/opinion_env.py` (修改：支持 alpha 参数)
- [ ] `src/baselines/node_selection.py` (新建：5种基线方法)
- [ ] `src/llm/strategy.py` (新建：3种LLM策略)
- [ ] `src/utils/noise.py` (新建：噪声注入)
- [ ] `experiments/run_ijcai_experiments.py` (新建：主控脚本)

### 6.2 结果文件

- [ ] `results/ijcai_experiments/results_main_comparison.json`
- [ ] `results/ijcai_experiments/results_ablation.json`
- [ ] `results/ijcai_experiments/results_robustness_noise.json`
- [ ] `results/ijcai_experiments/results_robustness_alpha.json`
- [ ] `results/ijcai_experiments/results_strategies.json`

### 6.3 文档和说明

- [ ] README 说明如何运行实验
- [ ] 记录每个实验的运行时间
- [ ] 说明遇到的任何问题或限制

---

## 七、测试和验证

### 7.1 单元测试

在实现每个模块后，编写简单的测试脚本验证功能：

```python
# tests/test_baselines.py
def test_random_selection():
    graph = nx.erdos_renyi_graph(100, 0.1)
    nodes = select_nodes_random(graph, k=10, seed=42)
    assert len(nodes) == 10
    assert len(set(nodes)) == 10  # 无重复

def test_high_degree_selection():
    # 创建一个明显的hub节点
    graph = nx.star_graph(50)  # 中心节点度数为50
    nodes = select_nodes_high_degree(graph, k=1)
    assert nodes[0] == 0  # 中心节点应该被选中

# ... 其他测试
```

### 7.2 小规模验证

在运行完整实验前，先用小规模数据验证：
- 使用小型合成图（100节点）
- 只跑 3-5 个 Seed
- 验证结果格式正确
- 验证不同方法确实产生不同结果

---

## 八、时间估计和优先级

### 优先级排序

1. **P0（必须完成）**：
   - 修改动力学模型支持新公式
   - 实现5种基线方法
   - 实现主控实验脚本
   - 完成实验一（SOTA对比）

2. **P1（重要）**：
   - 完成实验二（消融实验）
   - 完成实验三（鲁棒性测试）

3. **P2（加分项）**：
   - 完成实验四（策略多样性）
   - 性能优化和并行计算

### 时间估计

- **修改现有代码**：1-2 天
- **实现基线方法**：1-2 天
- **实现主控脚本**：2-3 天
- **运行实验一**：1-2 天（取决于并行度）
- **运行实验二、三、四**：2-3 天
- **调试和优化**：1-2 天

**总计**：约 8-14 个工作日

---

## 九、常见问题解答

### Q1: Budget 设置为节点数的百分比，但实际干预需要多少步？

**A**: Budget 就是干预步数（K）。如果 Budget=762，就是干预 762 步。每步选择一个节点，共选择 762 个节点。

### Q2: 干预权重是每步都重新获取，还是每个节点只获取一次？

**A**: 每步都重新获取。即使同一个节点在后续步骤中再次被选中（虽然不应该），也要重新获取权重。

### Q3: 如果选择的节点重复了怎么办？

**A**: 对于 RL 方法，应该使用 mask 避免重复选择。对于基线方法（Random, High-Degree 等），也要确保不重复。

### Q4: 新公式中的 alpha 和 FJ 模型中的 alpha 有什么区别？

**A**: 
- FJ 模型的 alpha：表示对初始观点的坚持程度
- 新公式中的 alpha：表示顽固度（对当前观点的坚持程度）

如果使用新公式，建议：
- DeGroot 变体：使用 alpha 作为顽固度
- FJ 变体：可以同时使用两个 alpha（一个用于顽固度，一个用于初始观点坚持），或者简化使用一个 alpha

**建议**：先实现 DeGroot 变体的新公式，验证效果后再考虑 FJ 变体。

### Q5: 完整数据集运行时间太长怎么办？

**A**: 
- 可以先在子图上测试（使用 `sample_subgraph` 函数）
- 对于基线方法（Random, High-Degree），可以先采样再选择
- 使用并行计算加速
- 如果确实太慢，可以先用较小的 Budget（如 1%）快速验证

---

## 十、联系和反馈

如果在实现过程中遇到问题，请：
1. 先查看现有代码和文档
2. 检查实验执行手册中的要求
3. 记录问题并反馈给团队

**祝开发顺利！**

---

**文档版本**: v1.0  
**创建日期**: 2026-01-10  
**最后更新**: 2026-01-10

