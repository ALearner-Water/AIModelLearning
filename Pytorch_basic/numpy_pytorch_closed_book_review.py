"""NumPy + PyTorch Tensor 闭卷巩固。

规则：
1. 关闭之前的课堂示例和挑战文件。
2. 只根据函数说明完成 8 个 TODO。
3. 不查看旧答案，不使用 Autograd，不修改函数名和参数。
4. 每题先在纸上或注释中预测 shape/dtype/device，再写代码。
5. 完成后告诉我，由我运行独立测试进行验收。
"""

import numpy as np
import torch


def center_rows_numpy(matrix):
    """让 NumPy 矩阵的每一行减去该行自己的平均值。

    返回结果必须与 matrix 的 shape 相同。
    """
    # TODO
    return matrix-np.mean(matrix,axis=1)[:,np.newaxis]  #行数不方便广播，在后边加一个维度


def changed_independent_block(matrix):
    """取得 NumPy 矩阵第2行至末行、第2～3列的独立副本。

    把副本左上角改为 -1 后返回；传入的 matrix 不能被修改。
    """
    # TODO
    a=np.copy(matrix[1:,1:3])
    a[0][0]=-1
    return a


def tensor_view_and_transpose(tensor):
    """把含24个元素的 Tensor view 成 (2,3,4)，再交换 dim1 与 dim2。

    返回 (viewed, transposed)。
    """
    # TODO
    a=tensor.view(2,3,4)
    b=a.transpose(1,2)
    return a,b

def combine_tensor_list(tensors):
    """组合一组 shape 相同的二维 Tensor。

    返回：
    1. 沿 dim0 使用 cat 的结果；
    2. 沿 dim1 使用 stack 的结果。
    """
    # TODO
    return torch.cat(tensors,dim=0),torch.stack(tensors,dim=1)


def prepare_training_tensor(array):
    """把 NumPy 数组准备成模型输入 Tensor。

    要求：
    1. dtype 为 torch.float32；
    2. CUDA 可用时位于 cuda，否则位于 cpu。
    """
    # TODO
    a=torch.tensor(array,dtype=torch.float32)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a=a.to(device)
    return a


def torch_linear_layer(inputs, weights, bias):
    """使用 PyTorch 计算 inputs @ weights + bias。"""
    # TODO
    return inputs @ weights+bias


def mse_numpy(true_y, pred_y):
    """使用 NumPy 返回均方误差标量。"""
    # TODO
    return np.mean((pred_y-true_y)**2)


def one_weight_gradient_step(
    x, true_y, w, learning_rate=0.1, epsilon=0.001
):
    """对模型 pred_y = x*w 完成一次数值梯度下降。

    使用中心差分计算 w 的梯度，不使用手工求导公式。
    返回 (new_w, new_loss)。
    """
    # TODO
    def mse(pred_y,true_y):
        return np.mean((pred_y-true_y)**2)
    def loss (w):
        pred_y=w*x
        return mse(pred_y,true_y)
    loss_of_right=loss(w+epsilon)
    loss_of_left=loss(w-epsilon)
    gradient=(loss_of_right-loss_of_left)/(2*epsilon)
    w=w-gradient*learning_rate
    new_loss=loss(w)
    return w,new_loss
