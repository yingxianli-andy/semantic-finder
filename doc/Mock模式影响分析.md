# Mock 模式影响分析

## 一、当前实现

**代码位置**: `src/llm/controller.py`

```python
def _call_llm(self, prompt: str) -> str:
    if self.use_mock:
        # Mock模式：返回一个基于规则的简单回复
        return "0.5"
```

**问题**: Mock 模式总是返回固定值 `0.5`，这意味着：
- 所有节点的干预权重都是 0.5
- 没有考虑节点特征（观点值、邻居情况等）
- 不够智能

---

## 二、影响分析

### 2.1 训练阶段的影响

**影响程度**: ⚠️ **轻微影响**

**原因**:
1. 根据文档，训练时建议**不使用 LLM**（`use_llm_in_training: false`）
2. 训练时使用**随机权重**加速训练
3. 训练脚本中的代码：
   ```python
   if use_llm:
       # 调用 LLM
   else:
       # 使用随机权重
       intervention_weight = np.random.uniform(0.3, 0.7)
   ```

**结论**: Mock 模式在训练阶段**影响很小**，因为训练时通常不使用 LLM。

---

### 2.2 测试阶段的影响

**影响程度**: ⚠️⚠️ **中等影响**

**原因**:
1. 测试时应该使用**真实的 LLM**来评估模型性能
2. 如果使用 Mock 模式：
   - 所有节点都得到相同的干预权重（0.5）
   - 无法体现 LLM 的语义理解能力
   - 测试结果**不够真实**
3. 但 Mock 模式可以用于：
   - 开发调试（没有 GPU 时）
   - 快速验证代码逻辑

**结论**: Mock 模式在测试阶段**有一定影响**，但主要用于开发环境。

---

### 2.3 实际使用场景

**场景 1: 有 GPU 和模型**
- ✅ 使用真实 LLM（`use_mock=False`）
- ✅ 获得真实的干预权重
- ✅ 测试结果可靠

**场景 2: 没有 GPU 或模型**
- ⚠️ 使用 Mock 模式（`use_mock=True`）
- ⚠️ 所有节点权重都是 0.5
- ⚠️ 测试结果不够真实，但可以验证代码逻辑

---

## 三、改进建议

### 3.1 改进 Mock 模式（推荐）

**当前问题**: 返回固定值 0.5，不够智能

**改进方案**: 基于节点特征返回不同的权重

```python
def _call_llm(self, prompt: str) -> str:
    if self.use_mock:
        # 改进的 Mock 模式：基于规则的启发式方法
        # 从 prompt 中提取信息（如果可能）
        # 或者使用节点特征计算权重
        
        # 简单启发式：观点越极端，需要的干预权重越大
        # 这里需要访问节点信息，但当前接口只有 prompt
        # 可以考虑在 Mock 模式下，直接使用节点特征计算
        
        return "0.5"  # 暂时保持原样
```

**更好的方案**: 修改 `get_intervention_weight` 方法，在 Mock 模式下直接使用节点特征：

```python
def get_intervention_weight(self, node_id: int, graph: nx.Graph) -> float:
    # 如果是 Mock 模式，直接基于规则计算
    if self.use_mock:
        return self._mock_intervention_weight(node_id, graph)
    
    # 否则使用 LLM
    # ... LLM 调用代码 ...

def _mock_intervention_weight(self, node_id: int, graph: nx.Graph) -> float:
    """
    Mock 模式下的启发式权重计算
    """
    if node_id not in graph:
        return 0.5
    
    node_opinion = abs(graph.nodes[node_id].get('opinion', 0.0))
    neighbors = list(graph.neighbors(node_id))
    
    # 规则1: 观点越极端，需要的干预权重越大
    base_weight = node_opinion * 0.7  # 0.0 到 0.7
    
    # 规则2: 如果邻居观点相反，需要更多干预
    if len(neighbors) > 0:
        neighbor_opinions = [graph.nodes[n].get('opinion', 0.0) for n in neighbors]
        node_opinion_val = graph.nodes[node_id].get('opinion', 0.0)
        opposing_count = sum(1 for op in neighbor_opinions if op * node_opinion_val < 0)
        opposing_ratio = opposing_count / len(neighbors)
        base_weight += opposing_ratio * 0.3  # 额外 0.0 到 0.3
    
    # 限制在 0.0-1.0 之间
    return max(0.0, min(1.0, base_weight))
```

---

### 3.2 添加警告信息

在 Mock 模式下添加明确的警告：

```python
def __init__(self, ..., use_mock: bool = False):
    if use_mock:
        logger.warning("⚠️ 使用 Mock 模式：所有节点的干预权重将基于启发式规则计算，"
                      "不是真实的 LLM 输出。测试结果可能不够准确。")
```

---

## 四、最终建议

### ✅ 短期方案（当前可用）

**现状**: Mock 模式返回固定值 0.5

**影响**:
- 训练阶段：✅ **无影响**（训练时不用 LLM）
- 测试阶段：⚠️ **有影响**（但主要用于开发调试）

**建议**: 
- 在测试脚本中添加警告，提醒用户 Mock 模式的结果不够真实
- 生产环境必须使用真实 LLM

### ✅ 长期方案（推荐改进）

**改进**: 实现基于规则的 Mock 模式

**好处**:
- 即使没有 LLM，也能获得相对合理的权重
- 更接近真实 LLM 的行为
- 便于开发和调试

**实现**: 按照 3.1 节的方案改进 `_mock_intervention_weight` 方法

---

## 五、总结

| 场景 | 影响程度 | 说明 |
|------|----------|------|
| **训练阶段** | ⚠️ 轻微 | 训练时通常不使用 LLM，影响很小 |
| **测试阶段（开发）** | ⚠️ 中等 | 可以验证代码逻辑，但结果不够真实 |
| **测试阶段（生产）** | ❌ 不可用 | 必须使用真实 LLM，否则结果不可信 |

**结论**: 
- ✅ **当前实现可以工作**，但 Mock 模式不够智能
- ✅ **建议改进** Mock 模式，使其基于节点特征计算权重
- ✅ **生产环境**必须使用真实 LLM

---

**文档维护者**: 安迪  
**日期**: 2024-01-07



