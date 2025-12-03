"""使用梯度下降求解线性回归。

本实现对应 Module 05 的优化基础，演示批量梯度下降的参数更新：
    theta <- theta - lr * grad
支持可选的 L2 正则化，用于缓解过拟合并连接 MAP 视角。
"""

from __future__ import annotations

import numpy as np

from .linear_regression_closed_form import add_intercept, predict


def compute_gradients(X: np.ndarray, y: np.ndarray, theta: np.ndarray, l2_reg: float = 0.0) -> np.ndarray:
    """计算均方误差目标的梯度。"""
    preds = X @ theta
    error = preds - y
    grad = (2 / len(X)) * (X.T @ error)
    if l2_reg > 0:
        grad += 2 * l2_reg * theta
        grad[0] -= 2 * l2_reg * theta[0]  # 不对截距正则化
    return grad


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    num_iter: int = 1000,
    l2_reg: float = 0.0,
) -> tuple[np.ndarray, list[float]]:
    """批量梯度下降训练线性回归。

    参数:
        X: 形状 (n_samples, n_features) 的特征矩阵，不含截距。
        y: 形状 (n_samples,) 的目标向量。
        lr: 学习率。
        num_iter: 迭代次数。
        l2_reg: L2 正则系数。"""

    X_design = add_intercept(X)
    theta = np.zeros(X_design.shape[1])
    history: list[float] = []
    for _ in range(num_iter):
        grad = compute_gradients(X_design, y, theta, l2_reg=l2_reg)
        theta -= lr * grad
        mse = np.mean((X_design @ theta - y) ** 2)
        history.append(mse)
    return theta, history


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    X_demo = rng.normal(size=(120, 2))
    true_theta = np.array([0.5, 2.0, -1.0])
    noise = rng.normal(scale=0.1, size=120)
    y_demo = predict(X_demo, true_theta) + noise

    theta_hat, loss_history = gradient_descent(X_demo, y_demo, lr=0.05, num_iter=500)
    print("估计参数:", theta_hat)
    print("前几次损失:", loss_history[:5])
