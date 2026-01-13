"""
Gephi导出工具：将NetworkX图导出为.gexf格式，支持颜色映射
"""

import logging
import networkx as nx
from typing import Optional, Dict, Any
import os

from .color_mapper import map_opinion_to_color, ColorMapper

logger = logging.getLogger(__name__)


class GephiExporter:
    """
    Gephi导出器：将图导出为.gexf格式，并自动应用颜色映射
    """
    
    def __init__(self, color_mapper: Optional[ColorMapper] = None):
        """
        初始化Gephi导出器
        
        Args:
            color_mapper: 颜色映射器（如果为None，使用默认的）
        """
        self.color_mapper = color_mapper or ColorMapper()
    
    def export_to_gexf(
        self,
        graph: nx.Graph,
        filepath: str,
        color_by_opinion: bool = True,
        node_size_attr: Optional[str] = None,
        edge_weight_attr: Optional[str] = None
    ) -> None:
        """
        导出图到.gexf格式
        
        Args:
            graph: NetworkX图对象
            filepath: 输出文件路径
            color_by_opinion: 是否根据opinion值设置颜色
            node_size_attr: 节点大小属性名（可选）
            edge_weight_attr: 边权重属性名（可选）
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # 创建图的副本，避免修改原图
        g = graph.copy()
        
        # 为节点添加可视化属性
        for node_id in g.nodes():
            node = g.nodes[node_id]
            
            # 设置颜色
            if color_by_opinion and 'opinion' in node:
                opinion = node['opinion']
                color = self.color_mapper.map(opinion)
                # Gephi使用viz:color属性
                node['viz'] = {'color': {'r': int(color[1:3], 16), 
                                        'g': int(color[3:5], 16), 
                                        'b': int(color[5:7], 16), 
                                        'a': 255}}
                # 同时保存为字符串属性（便于查看）
                node['color'] = color
            
            # 设置节点大小
            if node_size_attr and node_size_attr in node:
                size = node[node_size_attr]
                if 'viz' not in node:
                    node['viz'] = {}
                node['viz']['size'] = float(size)
            elif 'degree' in node:
                # 默认使用度数作为大小
                degree = node['degree']
                if 'viz' not in node:
                    node['viz'] = {}
                node['viz']['size'] = float(degree) * 2.0  # 缩放因子
        
        # 设置边权重
        if edge_weight_attr:
            for u, v in g.edges():
                if edge_weight_attr in g[u][v]:
                    g[u][v]['weight'] = float(g[u][v][edge_weight_attr])
        
        # 导出为.gexf格式
        try:
            nx.write_gexf(g, filepath)
            logger.info(f"Graph exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export graph to {filepath}: {e}")
            raise


def export_to_gexf(
    graph: nx.Graph,
    filepath: str,
    color_by_opinion: bool = True,
    node_size_attr: Optional[str] = None,
    edge_weight_attr: Optional[str] = None
) -> None:
    """
    便捷函数：导出图到.gexf格式
    
    Args:
        graph: NetworkX图对象
        filepath: 输出文件路径
        color_by_opinion: 是否根据opinion值设置颜色
        node_size_attr: 节点大小属性名（可选）
        edge_weight_attr: 边权重属性名（可选）
    """
    exporter = GephiExporter()
    exporter.export_to_gexf(
        graph, filepath, color_by_opinion, node_size_attr, edge_weight_attr
    )


def batch_export_gexf(
    graphs: Dict[str, nx.Graph],
    output_dir: str,
    prefix: str = "graph",
    color_by_opinion: bool = True
) -> None:
    """
    批量导出多个图到.gexf格式
    
    Args:
        graphs: 图字典，键为标识符，值为图对象
        output_dir: 输出目录
        prefix: 文件名前缀
        color_by_opinion: 是否根据opinion值设置颜色
    """
    os.makedirs(output_dir, exist_ok=True)
    
    exporter = GephiExporter()
    
    for key, graph in graphs.items():
        filepath = os.path.join(output_dir, f"{prefix}_{key}.gexf")
        exporter.export_to_gexf(graph, filepath, color_by_opinion=color_by_opinion)
    
    logger.info(f"Exported {len(graphs)} graphs to {output_dir}")
