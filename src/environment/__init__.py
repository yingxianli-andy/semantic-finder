"""
Environment module for opinion dynamics simulation.

This package is implemented by 实习生A（哈工爷） and provides:
- 数据加载与合成图生成（data_loader）
- 观点动力学模型（dynamics）
- Gym 环境封装（opinion_env）

接口规范请参考 `doc/开发文档.md` 与 `doc/接口对接文档.md`。
"""

from .opinion_env import OpinionDynamicsEnv
from .dynamics import update_opinions
from .data_loader import load_graph, generate_synthetic_graph

__all__ = [
    'OpinionDynamicsEnv',
    'update_opinions',
    'load_graph',
    'generate_synthetic_graph',
]

