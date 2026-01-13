"""
颜色映射工具：根据节点的opinion值自动分配颜色代码
"""

import numpy as np
from typing import Tuple


def map_opinion_to_color(opinion: float) -> str:
    """
    将观点值映射到颜色代码（Hex Code）
    
    颜色方案：
    - -1.0（极左）→ #FF0000（深红）
    - 0.0（中立）→ #FFFFFF（白色）
    - 1.0（极右）→ #0000FF（深蓝）
    - 中间值线性插值
    
    Args:
        opinion: 观点值（-1.0 到 1.0）
    
    Returns:
        color: 颜色代码（Hex格式，如 "#FF0000"）
    """
    # 限制opinion值在[-1, 1]范围内
    opinion = max(-1.0, min(1.0, opinion))
    
    # 定义颜色端点
    # 红色（-1.0）
    red = (255, 0, 0)  # #FF0000
    # 白色（0.0）
    white = (255, 255, 255)  # #FFFFFF
    # 蓝色（1.0）
    blue = (0, 0, 255)  # #0000FF
    
    # 线性插值
    # 将 opinion 从 [-1, 1] 映射到 [0, 1]
    # -1.0 -> 0.0 (红色), 0.0 -> 0.5 (白色), 1.0 -> 1.0 (蓝色)
    t = (opinion + 1.0) / 2.0  # 将 [-1, 1] 映射到 [0, 1]
    
    if t < 0.5:
        # 从红色到白色 (t: 0.0 -> 0.5)
        t_normalized = t * 2.0  # 将 [0, 0.5] 映射到 [0, 1]
        r = int(red[0] * (1 - t_normalized) + white[0] * t_normalized)
        g = int(red[1] * (1 - t_normalized) + white[1] * t_normalized)
        b = int(red[2] * (1 - t_normalized) + white[2] * t_normalized)
    else:
        # 从白色到蓝色 (t: 0.5 -> 1.0)
        t_normalized = (t - 0.5) * 2.0  # 将 [0.5, 1.0] 映射到 [0, 1]
        r = int(white[0] * (1 - t_normalized) + blue[0] * t_normalized)
        g = int(white[1] * (1 - t_normalized) + blue[1] * t_normalized)
        b = int(white[2] * (1 - t_normalized) + blue[2] * t_normalized)
    
    # 转换为Hex格式
    color = f"#{r:02X}{g:02X}{b:02X}"
    return color


def map_opinion_to_rgb(opinion: float) -> Tuple[int, int, int]:
    """
    将观点值映射到RGB元组
    
    Args:
        opinion: 观点值（-1.0 到 1.0）
    
    Returns:
        rgb: RGB元组 (r, g, b)，每个值在0-255之间
    """
    # 限制opinion值在[-1, 1]范围内
    opinion = max(-1.0, min(1.0, opinion))
    
    # 定义颜色端点
    red = (255, 0, 0)
    white = (255, 255, 255)
    blue = (0, 0, 255)
    
    # 线性插值
    # 将 opinion 从 [-1, 1] 映射到 [0, 1]
    t = (opinion + 1.0) / 2.0  # 将 [-1, 1] 映射到 [0, 1]
    
    if t < 0.5:
        # 从红色到白色 (t: 0.0 -> 0.5)
        t_normalized = t * 2.0  # 将 [0, 0.5] 映射到 [0, 1]
        r = int(red[0] * (1 - t_normalized) + white[0] * t_normalized)
        g = int(red[1] * (1 - t_normalized) + white[1] * t_normalized)
        b = int(red[2] * (1 - t_normalized) + white[2] * t_normalized)
    else:
        # 从白色到蓝色 (t: 0.5 -> 1.0)
        t_normalized = (t - 0.5) * 2.0  # 将 [0.5, 1.0] 映射到 [0, 1]
        r = int(white[0] * (1 - t_normalized) + blue[0] * t_normalized)
        g = int(white[1] * (1 - t_normalized) + blue[1] * t_normalized)
        b = int(white[2] * (1 - t_normalized) + blue[2] * t_normalized)
    
    return (r, g, b)


class ColorMapper:
    """
    颜色映射器类：提供更灵活的颜色映射功能
    """
    
    def __init__(
        self,
        color_scheme: str = "red_white_blue",
        min_opinion: float = -1.0,
        max_opinion: float = 1.0
    ):
        """
        初始化颜色映射器
        
        Args:
            color_scheme: 颜色方案（目前只支持 "red_white_blue"）
            min_opinion: 最小观点值
            max_opinion: 最大观点值
        """
        self.color_scheme = color_scheme
        self.min_opinion = min_opinion
        self.max_opinion = max_opinion
        
        if color_scheme == "red_white_blue":
            self.negative_color = (255, 0, 0)  # 红色
            self.neutral_color = (255, 255, 255)  # 白色
            self.positive_color = (0, 0, 255)  # 蓝色
        else:
            raise ValueError(f"Unsupported color scheme: {color_scheme}")
    
    def map(self, opinion: float) -> str:
        """
        映射观点值到颜色代码
        
        Args:
            opinion: 观点值
        
        Returns:
            color: Hex颜色代码
        """
        # 归一化到[-1, 1]
        normalized = (opinion - self.min_opinion) / (self.max_opinion - self.min_opinion) * 2.0 - 1.0
        normalized = max(-1.0, min(1.0, normalized))
        
        # 线性插值
        # 将 normalized 从 [-1, 1] 映射到 [0, 1]
        t = (normalized + 1.0) / 2.0  # 将 [-1, 1] 映射到 [0, 1]
        
        if t < 0.5:
            # 从负向颜色到中性颜色 (t: 0.0 -> 0.5)
            t_normalized = t * 2.0  # 将 [0, 0.5] 映射到 [0, 1]
            r = int(self.negative_color[0] * (1 - t_normalized) + self.neutral_color[0] * t_normalized)
            g = int(self.negative_color[1] * (1 - t_normalized) + self.neutral_color[1] * t_normalized)
            b = int(self.negative_color[2] * (1 - t_normalized) + self.neutral_color[2] * t_normalized)
        else:
            # 从中性颜色到正向颜色 (t: 0.5 -> 1.0)
            t_normalized = (t - 0.5) * 2.0  # 将 [0.5, 1.0] 映射到 [0, 1]
            r = int(self.neutral_color[0] * (1 - t_normalized) + self.positive_color[0] * t_normalized)
            g = int(self.neutral_color[1] * (1 - t_normalized) + self.positive_color[1] * t_normalized)
            b = int(self.neutral_color[2] * (1 - t_normalized) + self.positive_color[2] * t_normalized)
        
        return f"#{r:02X}{g:02X}{b:02X}"
    
    def map_batch(self, opinions: np.ndarray) -> list:
        """
        批量映射观点值到颜色代码
        
        Args:
            opinions: 观点值数组
        
        Returns:
            colors: 颜色代码列表
        """
        return [self.map(op) for op in opinions]
