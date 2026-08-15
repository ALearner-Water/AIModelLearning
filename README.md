# ai_learning

面向计算机本科阶段 AI 学习与科研准备的长期学习总仓库。

本仓库用于记录从 Python 基础到深度学习入门的代码、笔记、练习和阶段验收结果。它不是为了堆积文件或仓库数量，而是为了形成一条可以持续复习、运行和展示的学习轨迹。

## 仓库目标

- 补齐 Python 编程基础，能够独立阅读、修改和调试代码。
- 掌握 NumPy 与必要的线性代数，为张量计算建立直觉。
- 掌握 PyTorch 基础，能够独立完成数据处理、模型定义、训练和评估。
- 学习深度学习、Attention 和 Transformer 的核心概念与基础实现。
- 保留算法、数据结构和计算机基础练习。
- 为后续 MiniGPT、MiniRAG、论文复现和正式科研项目做准备。

## 当前阶段

当前处于 **Python 基础 → NumPy 准备阶段**。

现有主要内容：

| 文件或目录 | 用途 |
|---|---|
| `PYTHON_TO_NUMPY_STUDY_GUIDE.md` | Python 到 NumPy 的学习指南与每日练习 |
| `python_foundation_for_numpy.py` | 变量、列表、切片、循环、函数和列表推导式示例 |
| `python_numpy_readiness_challenge.py` | 进入 NumPy 前的 Python 通关练习 |
| `jupyter/` | Jupyter Notebook 练习 |

## 长期学习路线

1. **Python 基础**
   - 变量、数据类型、判断、循环
   - 字符串、列表、元组、集合、字典
   - 函数、文件读写、异常、模块和包
   - 类、对象、`self`、`__init__` 与基础类型注解

2. **NumPy 与线性代数**
   - `ndarray`、形状、索引、切片、广播
   - 向量化运算、矩阵乘法、统计操作
   - 向量、矩阵、线性变换、特征值和梯度的必要知识

3. **PyTorch**
   - Tensor、自动求导、Dataset 与 DataLoader
   - `nn.Module`、损失函数、优化器
   - 完整训练循环、验证、保存与加载模型

4. **Deep Learning**
   - 线性回归、逻辑回归、多层感知机
   - 过拟合、正则化、归一化与优化
   - CNN 基础及常见训练方法

5. **Attention 与 Transformer 准备**
   - Query、Key、Value 与 Scaled Dot-Product Attention
   - Multi-Head Attention、位置编码、残差连接和 LayerNorm
   - 在学习总仓库中完成小型练习后，拆分正式项目

6. **算法与计算机基础**
   - 数组、链表、栈、队列、哈希表、树和图
   - 排序、查找、递归、动态规划
   - 操作系统、计算机网络和数据库课程复习笔记

## 建议目录结构

目录按实际学习进度逐步创建，不为了看起来丰富而建立空目录。

```text
ai_learning/
├─ README.md
├─ python_basics/       # Python语法、数据容器、函数、文件与OOP练习
├─ numpy/               # NumPy练习与小实验
├─ linear_algebra/      # 面向AI的线性代数代码和笔记
├─ pytorch/             # Tensor、autograd、模型与训练循环
├─ deep_learning/       # 深度学习课程练习
├─ algorithms/          # 算法与数据结构练习
├─ cs_foundations/      # 操作系统、网络、数据库等复习材料
├─ notes/               # 阶段总结与错题记录
└─ jupyter/             # Notebook实验
```

## 使用方法

每次学习建议遵循下面的循环：

1. 观看少量课程并手动敲示例。
2. 关掉视频，独立重写核心代码。
3. 完成当天练习并运行验证。
4. 在对应目录补充简短说明或错题记录。
5. 完成一个明确的小目标后提交一次 Git 记录。

常用命令：

```powershell
git status
git add .
git commit -m "learn: complete Python list and dict exercises"
```

## 提交信息建议

- `learn:` 新知识练习，例如 `learn: practice Python dictionaries`
- `feat:` 新增可运行功能或小项目
- `fix:` 修复代码错误
- `docs:` 更新笔记或说明文档
- `refactor:` 调整代码结构但不改变功能

一次提交只对应一个清晰目标，避免把多个阶段的内容混在一次提交中。

## 当前阶段验收标准

开始 NumPy 前，应能在不照抄视频的情况下完成：

- 使用列表和字典保存并处理多条数据。
- 使用判断和循环完成筛选、统计与转换。
- 将代码拆成多个函数，并正确使用参数和返回值。
- 读取和写入文本或 JSON 文件。
- 使用 `try/except` 处理常见输入和文件错误。
- 创建并导入自己的模块。
- 独立完成一个学生成绩管理系统或同等规模项目。

## 何时拆分为独立仓库

本仓库只保存课程练习、知识实验和小型综合作业。满足以下条件时再建立独立仓库：

- 项目有明确目标和独立 README。
- 项目能够单独安装和运行。
- 项目有可展示结果、实验记录或测试。
- 项目后续会持续迭代，而不是一次性课堂练习。

计划中的独立仓库：

| 仓库 | 预计创建时机 |
|---|---|
| `transformer-from-scratch` | 开始完整手写并验证 Transformer 时 |
| `minigpt` | 开始构建可训练、可生成的 MiniGPT 时 |
| `minirag` | 开始构建完整检索增强生成系统时 |
| `paper-reproduction-xxx` | 确定第一篇正式复现论文后 |
| `research-project-xxx` | 获得正式科研课题后 |

未来可以按照真实项目增减仓库，但不要为了数量提前创建空仓库。

## 公开展示原则

`ai_learning` 可以公开，用于展示持续学习过程，但需要保持以下质量：

- 不上传密码、密钥、个人隐私和受限制的数据。
- 不上传虚拟环境、缓存、模型大文件和临时输出。
- 每个阶段保留可运行代码，并写清学习目标和运行方法。
- 定期整理重复或无意义的实验文件。

真正申请实验室或实习时，重点展示独立项目与论文复现仓库；本仓库用于证明基础积累和持续学习能力。

