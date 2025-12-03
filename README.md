# 机器学习数学自学项目

本仓库旨在帮助具有软件开发背景的学习者系统化复习机器学习所需的数学基础，并通过 Python 实践巩固概念。作者背景：40+ 岁、信息与计算科学本科、具备全栈与 Python 编程经验，目标是补齐数学短板并结合现代机器学习的直觉与实现。

## 学习路线概览
- **Module 00：环境与项目准备**（Python、Jupyter、GitHub、依赖）
- **Module 01：函数与微积分基础**
- **Module 02：线性代数与向量空间**
- **Module 03：概率论基础**
- **Module 04：数理统计与估计理论**
- **Module 05：优化基础与梯度方法**
- **Module 06：信息论与机器学习连接**

项目结构已按照模块化设计，每个模块包含文档、练习 Notebook 与配套源码骨架，便于分阶段深入与迭代完善。

## 环境准备
1. 安装 Python 3.10+，推荐使用 conda 或 venv 创建隔离环境。
2. 克隆本仓库后，安装依赖：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
3. 启动 Jupyter：
   ```bash
   jupyter notebook
   ```
   在浏览器中打开对应的 Notebook 文件开始练习。

## 迭代使用方法
本仓库支持与智能助手（如 GitHub Copilot/Codex Agent）协同改进：
- 当你说“继续完善 Module 02 的线性代数练习”时，助手应在对应 README 增补概念说明，并在 Notebook 中增加练习或参考答案。
- 当你粘贴自己的解答代码时，助手会进行代码审查，指出数学和实现问题，给出优化建议。

## 学习与使用建议
- 建议每周学习 3 次，每次 1–2 小时：先阅读 `docs` 对应模块，再完成 `notebooks` 中的练习，最后使用“自我检测问题”检查理解。
- 用 Git 提交记录学习进度，可在 README 或各模块文档中勾选完成项。
- 结合 `src` 目录的示例代码，在练习中直接调用工具函数或模型骨架，逐步补全实现。

## 目录结构
```
README.md                # 总体介绍与学习指南
requirements.txt         # Python 依赖
/docs
  module-XX-*/README.md  # 各模块学习指引
/notebooks               # 各模块配套练习（含 TODO）
/src
  linear_models          # 线性/逻辑回归示例
  utils                  # 数据与可视化工具
```

欢迎在 Issues 中记录问题或想法，也可以使用 Pull Request 逐步完善内容。
