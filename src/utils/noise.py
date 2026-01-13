"""
噪声注入模块：用于鲁棒性测试。

用于IJCAI实验三（鲁棒性测试 - 语义噪声）。
"""

import numpy as np
from typing import Optional, Tuple


def add_gaussian_noise(
    value: float,
    noise_std: float,
    clip_range: Optional[Tuple[float, float]] = None,
) -> float:
    """
    在值上添加高斯噪声。

    Args:
        value: 原始值
        noise_std: 噪声标准差
        clip_range: 裁剪范围（如 (0.0, 1.0)）

    Returns:
        noisy_value: 添加噪声后的值
    """
    noisy = value + np.random.normal(0, noise_std)
    if clip_range is not None:
        noisy = np.clip(noisy, clip_range[0], clip_range[1])
    return float(noisy)
