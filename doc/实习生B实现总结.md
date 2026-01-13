# 实习生B任务实现总结

## 已完成功能

### ✅ 任务一：LLM决策接口 (The Controller)

**文件位置：**
- `src/llm/controller.py` - LLM控制器主类
- `src/llm/prompt_template.py` - Prompt模板

**核心功能：**
1. **LLMController类**：封装Qwen模型推理接口
   - 支持HuggingFace transformers加载Qwen模型
   - 支持Mock模式（用于测试，不加载真实模型）
   - 自动提取子图信息（目标节点+邻居）
   - 解析LLM输出，提取沟通成本（0.0-1.0的浮点数）

2. **沟通成本评估**：
   - 输入：节点ID和图对象
   - 输出：沟通成本（0.0=低成本，1.0=高成本）
   - Prompt设计：考虑节点观点极端程度、邻居观点分歧、连接强度等因素

**使用方法：**
```python
from src.llm import LLMController

controller = LLMController(use_mock=True)  # 或 use_mock=False 使用真实模型
cost = controller.get_intervention_weight(node_id=0, graph=graph)
```

### ✅ 任务二：自动化Gephi流水线 (Visualization)

**文件位置：**
- `src/visualization/gephi_exporter.py` - Gephi导出工具
- `src/visualization/color_mapper.py` - 颜色映射工具

**核心功能：**
1. **颜色映射**：
   - 根据opinion值（-1到1）自动分配颜色
   - 颜色方案：-1.0→红色(#FF0000), 0.0→白色(#FFFFFF), 1.0→蓝色(#0000FF)
   - 中间值线性插值

2. **Gephi导出**：
   - 导出.gexf格式文件（NetworkX原生支持）
   - 自动为节点设置颜色属性（viz:color）
   - 支持批量导出多个图
   - 支持自定义节点大小和边权重

**使用方法：**
```python
from src.visualization import export_to_gexf

export_to_gexf(
    graph=graph,
    filepath="output/graph.gexf",
    color_by_opinion=True
)
```

### ✅ 任务三：语义增强 (Semantic Features)

**文件位置：**
- `src/llm/semantic_feature.py` - 语义特征提取器

**核心功能：**
1. **模拟推文生成**：
   - 使用LLM为节点生成模拟推文
   - 考虑节点的opinion值和度数
   - 生成符合节点观点的自然语言文本

2. **Embedding提取**：
   - 将生成的推文转换为Embedding向量
   - 使用sentence-transformers模型（默认：all-MiniLM-L6-v2）
   - 支持缓存机制，避免重复计算
   - 输出归一化的128维向量（可配置）

**使用方法：**
```python
from src.llm.semantic_feature import SemanticFeatureExtractor

extractor = SemanticFeatureExtractor(cache_dir="data/embeddings")
tweet = extractor.generate_tweet_for_node(node_id=0, graph=graph)
embedding = extractor.extract_semantic_embedding(node_id=0, graph=graph)
```

## 项目结构

```
src/
├── __init__.py
├── llm/
│   ├── __init__.py
│   ├── controller.py          # LLM控制器
│   ├── prompt_template.py     # Prompt模板
│   ├── semantic_feature.py    # 语义特征提取
│   └── README.md              # 使用说明
└── visualization/
    ├── __init__.py
    ├── gephi_exporter.py      # Gephi导出工具
    ├── color_mapper.py        # 颜色映射工具
    └── README.md              # 使用说明

examples/
└── llm_visualization_demo.py  # 功能演示脚本
```

## 关键特性

### 1. 沟通成本评估（而非persuasiveness score）

根据用户要求，LLM输出的浮点数**衡量沟通成本**：
- **0.0** = 沟通成本很低（容易沟通和说服）
- **1.0** = 沟通成本很高（难以沟通和说服）

Prompt设计考虑了：
- 节点观点的极端程度
- 邻居观点的对立程度
- 网络连接的强度
- 局部网络的极化程度

### 2. Mock模式支持

所有LLM相关功能都支持Mock模式：
- 不加载真实模型，适合测试和开发
- 使用启发式规则生成合理的结果
- 可以随时切换到真实模型

### 3. 缓存机制

语义特征提取支持缓存：
- 避免重复生成推文和计算Embedding
- 提高批量处理效率
- 可配置缓存目录

### 4. 错误处理

- 优雅降级：如果模型加载失败，自动切换到Mock模式
- 输出解析：如果LLM输出无法解析，返回默认值
- 日志记录：详细的日志信息，便于调试

## 依赖要求

### 必需依赖
- `networkx` - 图处理
- `numpy` - 数值计算

### 可选依赖（用于真实LLM推理）
- `transformers` - HuggingFace模型库
- `torch` - PyTorch（用于模型推理）
- `sentence-transformers` - 用于Embedding生成（可选，会自动fallback）

### 安装建议

```bash
# 基础依赖
pip install networkx numpy

# 如果要使用真实LLM模型
pip install transformers torch sentence-transformers
```

## 使用示例

运行演示脚本：
```bash
python examples/llm_visualization_demo.py
```

这将展示：
1. LLM控制器计算沟通成本
2. 语义特征提取（生成推文和Embedding）
3. Gephi可视化导出

## 与主流程的对接

### 与安迪（RL算法）的对接

```python
# 在训练/测试循环中
from src.llm import LLMController

llm_controller = LLMController(use_mock=False)  # 测试时使用真实模型
intervention_weight = llm_controller.get_intervention_weight(node_id, graph)
# intervention_weight 是沟通成本（0.0-1.0），可以直接用于环境step
```

### 与哈工爷（环境）的对接

```python
# 环境step方法接收intervention_weight
next_state, reward, done, info = env.step(action_node, intervention_weight)
```

### 可视化流水线

```python
# 在训练过程中定期导出图
from src.visualization import export_to_gexf

export_to_gexf(
    graph=current_graph,
    filepath=f"results/figures/step_{step}.gexf",
    color_by_opinion=True
)
```

## 注意事项

1. **GPU要求**：使用真实Qwen模型需要GPU（建议16GB+显存）
2. **模型下载**：首次使用会自动从HuggingFace下载模型（需要网络）
3. **Gephi布局**：导出的.gexf文件需要在Gephi中手动应用Force Atlas 2布局
4. **Mock模式**：默认使用Mock模式，适合快速测试，但结果可能不够准确

## 后续优化建议

1. **vLLM集成**：可以集成vLLM实现更高效的推理
2. **批量推理**：优化LLM批量处理多个节点的效率
3. **Prompt优化**：根据实际效果调整Prompt设计
4. **布局自动化**：研究Gephi Java API，实现布局自动化

---

**实现完成时间**：2024-01-XX  
**实现者**：实习生B（上交爷）
