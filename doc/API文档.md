# API 接口文档

## 一、核心接口对接

### 1.1 安迪 ↔ 哈工爷

**安迪提供：**
- `src/agent/reward.py` 中的 `compute_reward(state, next_state)` 函数
- 输入：两个 NetworkX 图对象（当前状态和下一状态）
- 输出：奖励值（float）

**哈工爷提供：**
- `src/environment/opinion_env.py` 中的 `OpinionDynamicsEnv` 类
- 方法：
  - `reset()` → 返回初始状态
  - `step(action_node, intervention_weight)` → 返回 (next_state, reward, done, info)

**使用示例：**
```python
from src.environment.opinion_env import OpinionDynamicsEnv
from src.agent.reward import compute_reward

env = OpinionDynamicsEnv(graph, budget=K, reward_fn=compute_reward)
state = env.reset()
next_state, reward, done, info = env.step(action_node, intervention_weight)
```

---

### 1.2 安迪 ↔ 上交爷

**上交爷提供：**
- `src/llm/controller.py` 中的 `LLMController` 类
- 方法：`get_intervention_weight(node_id, graph)` → 返回 0-1 的权重值（float）

**使用示例：**
```python
from src.llm.controller import LLMController

llm_controller = LLMController()
intervention_weight = llm_controller.get_intervention_weight(node_id, graph)
```

---

### 1.3 数据格式约定

**图对象（NetworkX Graph）：**
- 每个节点必须有 `opinion` 属性（float，范围 -1 到 1）
- 节点索引从 0 开始

**状态表示：**
- 使用 NetworkX Graph 对象
- 节点特征通过图的节点属性访问

**动作：**
- `action_node`: 节点索引（int）

**干预权重：**
- `intervention_weight`: 0.0 到 1.0 之间的浮点数

---

## 二、文件命名规范

### 2.1 代码文件命名

- **安迪的文件**：`src/agent/*.py`
  - `semantic_finder.py` - 主类
  - `encoder.py` - 编码器
  - `decoder.py` - 解码器
  - `reward.py` - 奖励函数
  - `replay_buffer.py` - 经验回放

- **哈工爷的文件**：`src/environment/*.py`
  - `opinion_env.py` - 环境主类
  - `dynamics.py` - 观点动力学模型
  - `data_loader.py` - 数据加载器

- **上交爷的文件**：`src/llm/*.py` 和 `src/visualization/*.py`
  - `controller.py` - LLM 控制器
  - `prompt_template.py` - Prompt 模板
  - `gephi_exporter.py` - Gephi 导出工具
  - `color_mapper.py` - 颜色映射

### 2.2 命名约定

- **文件命名**：小写字母 + 下划线，如 `opinion_env.py`
- **类命名**：大驼峰，如 `OpinionDynamicsEnv`、`LLMController`
- **函数命名**：小写字母 + 下划线，如 `compute_reward`、`get_intervention_weight`
- **变量命名**：小写字母 + 下划线，如 `node_id`、`intervention_weight`

---

## 三、工作流程

### 3.1 训练流程

```
1. 哈工爷：生成/加载合成图 → env.reset()
2. 安迪：Agent.select_action(state) → 选择节点
3. 上交爷（可选）：LLM.get_intervention_weight() → 权重（训练时可随机）
4. 哈工爷：env.step(action, weight) → 下一状态 + 奖励
5. 安迪：Agent.update() → 更新模型
```

### 3.2 测试流程

```
1. 哈工爷：加载真实数据集 → env.reset()
2. 安迪：加载模型 → Agent.select_action(state)
3. 上交爷：LLM.get_intervention_weight(node, graph) → 权重
4. 哈工爷：env.step(action, weight) → 下一状态
5. 上交爷：导出可视化图
```

---

**文档维护者**: 安迪  
**最后更新**: 2024-01-07




