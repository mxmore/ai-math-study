"""二分类逻辑回归的简易实现。

使用交叉熵损失与梯度下降训练，对应 Module 05/06 的优化与信息论内容。
"""

from __future__ import annotations

import numpy as np

from .linear_regression_closed_form import add_intercept


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid 激活函数。"""
    return 1 / (1 + np.exp(-z))


def compute_loss_and_gradients(X: np.ndarray, y: np.ndarray, theta: np.ndarray, l2_reg: float = 0.0) -> tuple[float, np.ndarray]:
    """计算交叉熵损失及其梯度。"""
    logits = X @ theta
    probs = sigmoid(logits)
    eps = 1e-8
    loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
    if l2_reg > 0:
        loss += l2_reg * np.sum(theta[1:] ** 2)
    grad = (X.T @ (probs - y)) / len(X)
    if l2_reg > 0:
        grad[1:] += 2 * l2_reg * theta[1:]
    return loss, grad


def fit(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    num_iter: int = 1000,
    l2_reg: float = 0.0,
) -> tuple[np.ndarray, list[float]]:
    """使用批量梯度下降训练逻辑回归。"""
    X_design = add_intercept(X)
    theta = np.zeros(X_design.shape[1])
    history: list[float] = []
    for _ in range(num_iter):
        loss, grad = compute_loss_and_gradients(X_design, y, theta, l2_reg=l2_reg)
        theta -= lr * grad
        history.append(loss)
    return theta, history


def predict_proba(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """返回正类概率。"""
    return sigmoid(add_intercept(X) @ theta)


def predict(X: np.ndarray, theta: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """根据阈值返回类别标签。"""
    return (predict_proba(X, theta) >= threshold).astype(int)


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    X_pos = rng.normal(loc=1.0, scale=0.8, size=(60, 2))
    X_neg = rng.normal(loc=-1.0, scale=0.8, size=(60, 2))
    X_demo = np.vstack([X_pos, X_neg])
    y_demo = np.hstack([np.ones(60), np.zeros(60)])

    theta_hat, losses = fit(X_demo, y_demo, lr=0.2, num_iter=300)
    acc = (predict(X_demo, theta_hat) == y_demo).mean()
    print(f"最终准确率: {acc:.3f}")
    print("前几次损失:", losses[:5])
