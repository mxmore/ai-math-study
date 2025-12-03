# Module 00：环境与项目准备

## 模块简介
本模块帮助快速搭建 Python 数据科学环境，熟悉 Jupyter、Git/GitHub 的基本操作，为后续数学与机器学习练习提供稳定工具链。

## 学习目标
- 能使用 conda 或 venv 创建独立的 Python 环境。
- 能安装并管理 `requirements.txt` 中的依赖。
- 能启动并使用 Jupyter Notebook 进行实验。
- 能使用 Git 进行基本版本控制、推送到 GitHub。
- 能理解项目目录结构并快速定位文档、代码与练习。

## 核心概念
- 虚拟环境：隔离不同项目的依赖，避免冲突。
- 依赖管理：使用 `pip`/`conda` 安装与冻结版本。
- 版本控制：Git 的工作区、暂存区与提交历史。
- Notebook 工作流：Markdown 说明 + 代码单元的混合体验。

## 推荐学习顺序
1. 安装 Python（3.10+）与包管理工具（conda 或 venv）。
2. 克隆仓库并创建虚拟环境，安装 `requirements.txt`。
3. 熟悉 Git 基本命令：`status`、`add`、`commit`、`push`。
4. 启动 Jupyter，运行示例 Notebook，体验代码与可视化。

## 配套编程练习
- 在 Notebook 中打印 Python 与主要库版本，验证环境。
- 写一段示例代码：生成随机数组并绘制直方图。
- 使用 Git 完成一次提交并推送到远程（练习用）。

## 与机器学习的联系
- 稳定的环境能保证实验可复现性。
- Notebook 便于将数学推导、代码实验与可视化整合在一起。

## 自我检测问题
1. 为什么需要虚拟环境？如何在同一台机器上管理多个项目的依赖？
2. `pip freeze > requirements.txt` 的作用是什么？
3. 如何在 Jupyter 中插入 Markdown 与代码单元？
4. Git 的工作区、暂存区、历史提交各是什么？
5. 如何在 Notebook 中快速预览数据的前几行？
6. 如何配置 matplotlib 以显示中文或更高分辨率图像？
7. 如果依赖安装失败，如何根据报错信息定位并解决问题？
