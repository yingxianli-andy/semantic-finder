# 可视化模块使用说明

本模块实现了Gephi可视化流水线，包括：
1. 颜色映射工具（根据opinion值映射颜色）
2. Gephi导出工具（导出.gexf格式）

## 快速开始

### 1. 颜色映射

```python
from src.visualization import map_opinion_to_color

# 将opinion值映射到颜色
color = map_opinion_to_color(opinion=-0.9)  # 返回 "#FF0000" (红色)
color = map_opinion_to_color(opinion=0.0)   # 返回 "#FFFFFF" (白色)
color = map_opinion_to_color(opinion=0.9)   # 返回 "#0000FF" (蓝色)
```

### 2. 导出到Gephi

```python
from src.visualization import export_to_gexf
import networkx as nx

# 创建图并添加opinion属性
graph = nx.Graph()
graph.add_node(0, opinion=-0.8)
graph.add_node(1, opinion=0.5)
graph.add_edge(0, 1)

# 导出为.gexf格式
export_to_gexf(
    graph=graph,
    filepath="output/graph.gexf",
    color_by_opinion=True  # 根据opinion值自动设置颜色
)
```

### 3. 批量导出

```python
from src.visualization.gephi_exporter import batch_export_gexf

# 准备多个图（例如：不同时间步的图）
graphs = {
    "step_0": graph_initial,
    "step_10": graph_step10,
    "step_20": graph_step20,
}

# 批量导出
batch_export_gexf(
    graphs=graphs,
    output_dir="results/figures",
    prefix="evolution",
    color_by_opinion=True
)
```

## 颜色方案

默认颜色方案（red_white_blue）：
- **-1.0** (极左) → `#FF0000` (深红)
- **0.0** (中立) → `#FFFFFF` (白色)
- **1.0** (极右) → `#0000FF` (深蓝)
- 中间值：线性插值

## Gephi使用建议

1. **导入文件**：在Gephi中打开导出的.gexf文件
2. **应用布局**：使用 **Force Atlas 2** 布局算法
3. **保存预设**：保存布局预设，确保所有图布局一致
4. **调整颜色**：颜色已自动设置，但可以在Gephi中进一步调整
5. **导出图片**：使用Gephi的导出功能生成高质量图片

## 接口说明

### 便捷函数

- `map_opinion_to_color(opinion) -> str`: 映射opinion到Hex颜色代码
- `export_to_gexf(graph, filepath, ...)`: 导出图到.gexf格式

### 类接口

- `ColorMapper`: 颜色映射器类，支持自定义颜色方案
- `GephiExporter`: Gephi导出器类，提供更多控制选项
