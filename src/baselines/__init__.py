"""
基线方法模块：实现各种节点选择策略。

用于IJCAI实验对比。
"""

from .node_selection import (
    select_nodes_random,
    select_nodes_high_degree,
    select_nodes_pagerank,
    select_nodes_finder,
    select_nodes_semantic_finder,
)

__all__ = [
    "select_nodes_random",
    "select_nodes_high_degree",
    "select_nodes_pagerank",
    "select_nodes_finder",
    "select_nodes_semantic_finder",
]
