"""线性回归的闭式解实现。

使用正规方程求解参数：
    theta = (X^T X)^{-1} X^T y
其中 X 为包含截距列的设计矩阵，y 为目标向量。
本示例对应 Module 02 的线性代数与 Module 05 的优化内容，展示最小二乘的代数求解。
"""

from __future__ import annotations

import numpy as np


def add_intercept(X: np.ndarray) -> np.ndarray:
    """在特征矩阵前添加截距列。"""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    intercept = np.ones((X.shape[0], 1))
    return np.hstack([intercept, X])


def normal_equation(X: np.ndarray, y: np.ndarray, l2_reg: float | None = None) -> np.ndarray:
    """使用正规方程求解线性回归参数。

    参数:
        X: 形状 (n_samples, n_features) 的特征矩阵，不含截距列。
        y: 形状 (n_samples,) 的目标向量。
        l2_reg: 可选的 L2 正则系数，默认为 None（无正则）。

    返回:
        theta: 形状 (n_features + 1,) 的参数向量，包含截距。"""

    X_design = add_intercept(X)
    xtx = X_design.T @ X_design
    if l2_reg is not None and l2_reg > 0:
        xtx = xtx + l2_reg * np.eye(xtx.shape[0])
    xty = X_design.T @ y
    theta = np.linalg.pinv(xtx) @ xty
    return theta


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """根据参数预测输出。"""
    X_design = add_intercept(X)
    return X_design @ theta


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X_demo = rng.normal(size=(100, 1))
    true_theta = np.array([1.5, -3.0])  # 截距与斜率
    noise = rng.normal(scale=0.2, size=100)
    y_demo = predict(X_demo, true_theta) + noise

    theta_hat = normal_equation(X_demo, y_demo)
    print("估计参数:", theta_hat)
    print("真实参数:", true_theta)
