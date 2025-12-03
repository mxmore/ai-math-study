"""绘图工具函数，便于在 Notebook 中复用。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(losses: list[float], title: str = "损失曲线") -> None:
    """绘制损失随迭代的变化。"""
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("迭代")
    plt.ylabel("损失")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model_predict, title: str = "决策边界") -> None:
    """绘制 2D 数据的决策边界。

    参数:
        X: 形状 (n_samples, 2) 的二维特征。
        y: 二分类标签。
        model_predict: 接受 (n_samples, 2) 返回预测标签的可调用对象。
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model_predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, levels=50, cmap="coolwarm", alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
