# AIModelLearning

面向人工智能基础学习的长期代码仓库，主要记录 Python、NumPy 与 PyTorch 的可运行练习、阶段验收和小型实验。

## 当前内容

| 目录 | 内容 |
|---|---|
| `Python_basic/` | Python 基础、文件处理、异常处理、命令行程序与 NumPy 前置练习 |
| `Numpy_basic/` | 数组操作、广播、矩阵运算、MSE、数值梯度和阶段挑战 |
| `Pytorch_basic/` | Tensor、Autograd、Dataset、DataLoader 与训练循环练习 |

当前学习阶段为 **PyTorch Dataset / DataLoader**。Tensor 与 Autograd 基础练习已经完成，正在学习数据集封装、批量加载和训练循环。

## 运行示例

在仓库根目录运行相应脚本：

```powershell
python Python_basic/python_to_numpy/python_numpy_readiness_challenge.py
python Numpy_basic/numpy_final_challenge.py
python Pytorch_basic/pytorch_tensor_phase2_challenge.py
python Pytorch_basic/pytorch_Autograd.py
python Pytorch_basic/dataset_basics.py
python Pytorch_basic/dataloader_training_loop.py
```

打开 Python 基础 Notebook：

```powershell
jupyter notebook Python_basic/jupyter/Base_python.ipynb
```

## 学习方式

每项练习尽量遵循以下闭环：

1. 理解知识点解决的问题；
2. 独立手写最小示例；
3. 实际运行并检查输出；
4. 完成阶段挑战或闭卷复写；
5. 用清晰的提交记录保存可验证进度。

## 仓库边界

本仓库只保存适合公开的学习代码、技术笔记和虚构测试数据。以下内容不进入版本控制：

- 个人课表、学习日程和长期规划；
- 助手协作规则和本机专用配置；
- 简历、面试材料及生成文档；
- 数据集、模型权重、训练输出和临时渲染文件；
- 密钥、虚拟环境、缓存及本机绝对路径。

形成能够独立安装、运行和展示的完整项目后，再拆分为单独仓库。
