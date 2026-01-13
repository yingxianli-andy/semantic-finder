# 数据目录说明

## 目录结构

- `raw/` - 原始数据（SNAP 数据集）
  - 存放从 SNAP 下载的原始图数据文件
  - 支持格式：`.edgelist` 或 `.txt`
  - 使用 `scripts/download_datasets.py` 下载数据

- `processed/` - 处理后的数据
  - 存放预处理后的图数据
  - 可用于加速训练和测试

- `synthetic/` - 合成图数据
  - 存放训练时生成的合成图（可选）
  - 训练时会自动生成，无需手动准备

## 数据准备

### 训练阶段
- **不需要真实数据**
- 使用 `generate_synthetic_graph()` 自动生成合成图
- 数据会在内存中生成，不保存到磁盘

### 测试阶段（可选）
如果需要测试真实数据集：

1. 下载数据集：
```bash
python scripts/download_datasets.py --dataset ego-Twitter
```

2. 数据会自动保存到 `data/raw/` 目录

3. 使用 `load_graph()` 加载：
```python
from src.environment import load_graph
graph = load_graph("ego-Twitter", opinion_init="bimodal")
```

## 注意事项

- 训练时不需要任何数据文件
- 测试时如果需要真实数据，才需要下载
- 所有目录都是空的，这是正常的

