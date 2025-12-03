"""数据生成工具函数，便于快速创建教学用数据集。"""

from __future__ import annotations

import numpy as np


def make_linear_regression(n_samples: int = 100, noise: float = 0.1, random_state: int | None = None):
    """生成一维线性回归数据 (y = 3x + 1 + 噪声)。"""
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-1, 1, size=(n_samples, 1))
    y = 3 * X[:, 0] + 1 + rng.normal(scale=noise, size=n_samples)
    return X, y


def make_binary_classification(n_samples: int = 200, random_state: int | None = None):
    """生成可线性分的二分类数据。"""
    rng = np.random.default_rng(random_state)
    X_pos = rng.normal(loc=1.0, scale=0.6, size=(n_samples // 2, 2))
    X_neg = rng.normal(loc=-1.0, scale=0.6, size=(n_samples // 2, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    return X, y
