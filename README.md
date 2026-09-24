# AIModelLearning

面向人工智能基础学习的长期代码仓库，记录 Python、NumPy 与 PyTorch 的可运行练习、阶段验收和小型实验。

## 学习路线入口

本仓库按 Python → NumPy → PyTorch Tensor → Autograd → 训练组件 → 完整训练工程逐级推进。当前关卡、阻断点和唯一下一步只记录在本机 `planning/LEARNING_INVENTORY.md`，本周任务只记录在 `planning/CURRENT_WEEK.md`，避免 README 与实际进度相互冲突。

`planning/` 含个人学习安排，不进入版本控制；公开 README 只保留稳定的仓库说明和运行方式。

## 目录与脚本角色

| 目录或脚本 | 角色 |
|---|---|
| `Python_basic/` | Python 基础、文件与异常、CLI、NumPy 前置 |
| `Numpy_basic/` | 数组、广播、矩阵运算、MSE、数值梯度与线性回归 |
| `Pytorch_basic/` | Tensor、Autograd、Dataset、DataLoader 与训练循环 |
| `*_lesson*.py`、`*_basics.py` | 学习示例，不单独代表阶段通关 |
| `*_challenge.py` | 阶段验收脚本 |
| `Pytorch_basic/torch_review/` | 延迟复测产出，结果以本地复测日历为准 |

## 运行环境

2026-09-05 已验收环境：

- Python 3.11.15
- NumPy 2.4.6
- PyTorch 2.12.1+cu126
- JupyterLab 4.6.3

现有环境名为 `dl-study`。GPU 可用不是必需条件，脚本应能在 CPU 上运行。

```powershell
conda activate dl-study
python --version
python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"
```

新环境可先安装仓库的最小依赖；CUDA 版本应根据机器和 PyTorch 官方安装方式另行选择。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 核心验收命令

从仓库根目录一键复验当前六个核心脚本：

```powershell
.\verify_core.ps1
```

也可以逐条运行：

```powershell
conda run -n dl-study python Python_basic/python_to_numpy/python_numpy_readiness_challenge.py
conda run -n dl-study python Numpy_basic/numpy_final_challenge.py
conda run -n dl-study python Pytorch_basic/pytorch_class/tensor/tensor_phase2_challenge.py
conda run -n dl-study python Pytorch_basic/pytorch_class/autograd/autograd.py
conda run -n dl-study python Pytorch_basic/pytorch_class/dataset_dataloader/dataset_basics.py
conda run -n dl-study python Pytorch_basic/pytorch_class/dataset_dataloader/dataloader_training_loop.py
```

2026-09-05 复验结果：以上六条命令均正常退出；Python→NumPy 显示“全部通过”，NumPy 为 `8/8`，PyTorch Tensor 第二阶段为 `6/6`，批训练脚本最终 loss 小于 `0.01`。

打开 Notebook：

```powershell
conda run -n dl-study jupyter notebook Python_basic/jupyter/Base_python.ipynb
```

## 学习闭环

每个新关卡依次执行：

1. 确认它在路线图中的位置与直接前置；
2. 讲清解决的问题、调用时机、输入和输出；
3. 只引入一个主要新概念，并完成最小手写练习；
4. 实际运行，记录输出和第一个阻断点；
5. 用户闭屏复述，通过即时验收；
6. 登记 D+3、D+10、D+30 延迟复测。

## 仓库边界

本仓库只保存适合公开的学习代码、技术笔记和虚构测试数据。以下内容不进入版本控制：

- 个人课表、学习日程和长期规划；
- 助手协作规则和本机专用配置；
- 简历、面试材料及生成文档；
- 数据集、模型权重、训练输出和临时渲染文件；
- 密钥、虚拟环境、缓存及本机绝对路径。

形成能够独立安装、运行和展示的完整项目后，再拆分为单独仓库。
