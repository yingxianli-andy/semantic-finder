这是一个为“代码能力强但缺乏实验设计背景”的队友量身定制的**详细实验执行手册**。

这份文档的特点是：**去学术化、强执行性、参数化**。我将实验设计转化为了具体的**输入参数（Args）**和**循环逻辑（Loops）**，他只需要照着写代码逻辑，跑出数据即可。

你可以直接把下面这个文档发给他。

---

# Semantic-FINDER 实验执行手册 (IJCAI 投稿版)

**致哈工爷：**
为了保证论文实验数据的丰满度和说服力，我们需要脱离“跑通就行”的模式，改为**“控制变量法”**的批量测试。目标是生成带误差带（Error Bar）的曲线图，证明我们的方法在各种极端的环境下都比别人的好。

**核心任务：** 编写一个统一的测试脚本（或主控入口），通过修改配置参数，跑完以下 **4 大组实验**，并将结果保存为 JSON/CSV。

---

## 0. 全局设置与基础定义

在开始之前，请确保模拟器（Simulator）支持以下参数的动态调整。

### 核心公式 (DeGroot Model 变体)
观点更新公式需支持参数化：
$$x_i(t+1) = \alpha \cdot x_i(t) + (1-\alpha) \cdot \left( \sum_{j \in N(i)} w_{ij} x_j(t) + \text{Intervention}_i \right)$$

*   **$x_i$**: 观点值 (0~1)。
*   **$\alpha$ (Stubbornness)**: 顽固度，**这是一个关键变量**。
*   **Intervention**: 如果节点被选中，我会给它施加一个由 LLM 决定的权重 $\omega$。

### 统一输出格式
所有实验必须保存**每一次 Run 的详细数据**（不要只保存平均值，画图需要方差）。
建议保存为 `.json`，结构如下：
```json
{
  "experiment_id": "exp1_baseline_01",
  "config": {"budget": 5, "method": "Ours", "alpha": 0.5, "noise": 0.0},
  "results": [
    {"seed": 1, "polarization_history": [0.8, 0.7, 0.5...], "final_score": 0.1},
    {"seed": 2, "polarization_history": [0.85, 0.75, 0.6...], "final_score": 0.12},
    ... // 跑 20-50 个 Seed
  ]
}
```

---

## 实验一：SOTA 性能对比 (Main Result)
**目的：** 证明我们的 Semantic-FINDER 效果最好，收敛最快。
**变量：** 不同的选点策略 (Methods)。

请实现一个循环，遍历以下策略，每种策略跑 **50 个随机种子 (Seeds)**。

1.  **Baseline 1: Random**
    *   逻辑：随机选 $K$ 个节点。
    *   干预权重：固定为 1.0 (强力洗脑)。
2.  **Baseline 2: High-Degree (大V策略)**
    *   逻辑：选度数（Degree）最大的 $K$ 个节点。
    *   干预权重：固定为 1.0。
3.  **Baseline 3: PageRank**
    *   逻辑：选 PageRank 值最高的 $K$ 个节点。
    *   干预权重：固定为 1.0。
4.  **Baseline 4: Original FINDER (无语义)**
    *   逻辑：用预训练好的 RL 模型选点。
    *   干预权重：固定为 0.5 (模拟没有 LLM 指导，只能给个折中值)。
5.  **Ours: Semantic-FINDER (完整版)**
    *   逻辑：RL 选点 + LLM 动态权重。
    *   干预权重：**调用 `get_llm_weight(node_features)`** (模拟 0.1~1.0 的动态值)。

**你需要跑的参数组合：**
*   **图数据：** `Twitter_Combined` (真实数据)
*   **Budget (K):** [1%, 2%, 5%] (测试三种预算下的表现)
*   **Metric:** 记录每一步的 Network Variance (方差) 或 Polarization Index。

---

## 实验二：消融实验 (Ablation Study)
**目的：** 证明“RL选点”和“LLM给权重”缺一不可。
**变量：** 开关特定模块。

请设置一个 flag 参数，跑以下 3 种变体：

1.  **w/o LLM (去掉大模型):**
    *   选点：使用 RL (FINDER)。
    *   权重：**固定为随机数** (`random.uniform(0, 1)`)。
    *   *解释：证明如果有好的节点，但瞎给权重，效果也不行。*
2.  **w/o RL (去掉结构学习):**
    *   选点：**随机选择 (Random)**。
    *   权重：使用 LLM 生成的完美权重。
    *   *解释：证明只靠嘴炮（LLM），找不到关键人（RL），也没用。*
3.  **Full Model (Ours):**
    *   选点：RL。
    *   权重：LLM。

---

## 实验三：鲁棒性测试 (Robustness - 核心加分项)
**目的：** 证明环境变化或模型不准时，系统依然稳定（这是为了打破“实验固化”）。
**变量：** 引入噪声和环境阻力。

**请在代码中加入以下两个干扰变量：**

### 3.1 变量 A：LLM 的“幻觉”噪声 (Semantic Noise)
模拟 LLM 有时候会判断错误。在 LLM 输出的权重上加高斯噪声。
*   公式：`actual_weight = llm_weight + Noise`
*   **Loop 参数 `noise_std`:** `[0.0, 0.2, 0.5, 0.8]`
*   **预期结果：** 随着噪声变大，效果会变差，但我们希望看到曲线下降得比较平缓（说明耐造）。

### 3.2 变量 B：人群顽固度 (User Stubbornness)
模拟这群人很难被说服。
*   修改 DeGroot 公式中的 $\alpha$。
*   **Loop 参数 `alpha`:** `[0.1, 0.3, 0.5, 0.7, 0.9]`
    *   0.1 = 耳根子软，听风就是雨。
    *   0.9 = 极其固执，不论怎么干预都很难改变。
*   **预期结果：** 展示在 $\alpha=0.9$ 这种极端情况下，我们的方法比 Baseline (如 Random) 强出更多。

---

## 实验四：LLM 策略多样性 (Strategy Analysis)
**目的：** 展示 LLM 不是复读机，不同的 Prompt 策略会带来不同的社会治理效果。
**变量：** 模拟三种不同的 LLM 性格（在代码里写成三种函数即可，不用真调 API，省钱省时间）。

请实现一个 `strategy` 参数：

1.  **Strategy: "Conservative" (温和派)**
    *   代码逻辑：`weight = 0.3 + 0.1 * random()` (总是给低权重，慢慢感化)。
2.  **Strategy: "Aggressive" (激进派)**
    *   代码逻辑：`weight = 0.8 + 0.2 * random()` (总是给高权重，强力洗脑)。
3.  **Strategy: "Adaptive" (我们的方法 - 聪明派)**
    *   代码逻辑：计算该节点邻居的方差 `var_neighbor`。
    *   `if var_neighbor > high: weight = 1.0` (周围吵得凶，就要强力介入)
    *   `if var_neighbor < low: weight = 0.2` (周围很和平，轻轻推一下就行)

**任务：** 跑这三种策略的对比图。证明 **Adaptive** 策略在消耗最小的情况下（如果考虑权重作为成本），效果最好。

---

## 总结：你需要交付的文件清单

1.  **`results_main_comparison.json`**: 包含 5 种 Method x 3 种 Budget 的数据。
2.  **`results_ablation.json`**: 包含 w/o LLM, w/o RL, Full Model 的数据。
3.  **`results_robustness_noise.json`**: 不同噪声水平下的数据。
4.  **`results_robustness_alpha.json`**: 不同顽固度 $\alpha$ 下的数据。
5.  **`results_strategies.json`**: 三种策略的对比数据。

**注意事项：**
*   **并行计算：** 这里的 Loop 很多，建议利用多核 CPU (`multiprocessing`) 并行跑不同的 Seed。
*   **Seed 固定：** 虽然我们要跑 50 个 Seed，但请把这 50 个数固定下来（比如 0-49），确保对比不同方法时，初始图的状态是一模一样的，这样才公平。