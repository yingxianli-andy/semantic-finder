# LLM模块使用说明

本模块实现了实习生B的LLM相关功能，包括：
1. LLM控制器（计算沟通成本）
2. 语义特征提取（生成模拟推文和Embedding）

## 快速开始

### 1. LLM控制器

```python
from src.llm import LLMController
import networkx as nx

# 初始化控制器（Mock模式，用于测试）
controller = LLMController(use_mock=True)

# 或者使用真实Qwen模型（需要GPU）
# controller = LLMController(
#     model_name="Qwen/Qwen2-7B-Instruct",
#     device="cuda",
#     use_mock=False
# )

# 创建图并添加opinion属性
graph = nx.Graph()
graph.add_node(0, opinion=-0.9)
graph.add_node(1, opinion=0.8)
graph.add_edge(0, 1)

# 计算沟通成本
cost = controller.get_intervention_weight(node_id=0, graph=graph)
print(f"沟通成本: {cost:.3f}")  # 输出: 0.0-1.0之间的浮点数
```

### 2. 语义特征提取

```python
from src.llm.semantic_feature import SemanticFeatureExtractor

# 初始化提取器
extractor = SemanticFeatureExtractor(
    cache_dir="data/embeddings"  # 可选：缓存目录
)

# 生成模拟推文
tweet = extractor.generate_tweet_for_node(node_id=0, graph=graph)
print(f"生成的推文: {tweet}")

# 提取语义Embedding
embedding = extractor.extract_semantic_embedding(node_id=0, graph=graph)
print(f"Embedding维度: {embedding.shape}")  # 通常是 (128,)
```

## 接口说明

### LLMController

- `get_intervention_weight(node_id, graph) -> float`: 计算沟通成本（0.0-1.0）
- `get_intervention_strategy(subgraph_text) -> float`: 根据文本描述计算成本

### SemanticFeatureExtractor

- `generate_tweet_for_node(node_id, graph) -> str`: 生成模拟推文
- `extract_semantic_embedding(node_id, graph) -> np.ndarray`: 提取Embedding向量
- `batch_extract_embeddings(graph, node_ids) -> Dict[int, np.ndarray]`: 批量提取

## 注意事项

1. **Mock模式**：默认使用Mock模式，不加载真实模型，适合测试
2. **GPU要求**：使用真实Qwen模型需要GPU和足够的显存（至少16GB）
3. **模型下载**：首次使用会自动从HuggingFace下载模型
4. **缓存机制**：语义特征提取支持缓存，避免重复计算
