"""NumPy 阶段闭卷综合测试。

覆盖内容：数组创建与 shape、切片与独立复制、axis 与广播、矩阵乘法、
MSE、数值梯度、梯度下降以及完整的一元线性回归训练。

规则：
1. 只修改标有 TODO 的 8 个函数，不要修改测试代码。
2. 不查看以前的答案；可以根据每题运行后的报错自行修改。
3. 除训练循环外，尽量使用 NumPy 运算，不逐个处理数组元素。
4. 数值梯度统一使用中心差分，不使用手工求导公式。

运行：
    conda run -n dl-study python Numpy_basic/numpy_final_challenge.py
"""

import numpy as np


def prepare_column(values):
    """把 values 转成浮点型 NumPy 数组，并整理为形状 (n, 1)。""" #n行1列
    a = np.array(values, dtype=float)[:,np.newaxis]     #改参数类型是dtype
    return a

def changed_block_copy(matrix):
    """取得第 2 行至末行、第 2～3 列，并制作独立副本。

    把副本左上角改成 -1 后返回副本；传入的 matrix 不能发生变化。
    """
    # TODO
    a=np.copy(matrix[1:,1:3])
    a[0][0]=-1
    return a


def center_each_column(matrix):
    """让每一列都减去该列自己的平均值。"""
    # TODO：正确选择 axis，并利用广播完成计算
    return matrix-np.mean(matrix,axis=0)


def linear_layer(inputs, weights, bias):
    """计算二维线性层 outputs = inputs @ weights + bias。"""
    # TODO
    return inputs @ weights +bias


def mse(true_y, pred_y):
    """返回真实值与预测值之间的均方误差。"""
    # TODO
    return np.mean((pred_y-true_y)**2)


def numerical_gradient_w(w, b, x, true_y, epsilon=0.001):
    """用中心差分计算当前 w 对 loss 的数值梯度。"""
    # 模型为 pred_y = x * w + b
    # TODO
    pred_y_right=x*(w+epsilon)+b
    pred_y_left =x * (w - epsilon) + b
    loss_of_right=mse(true_y,pred_y_right)
    loss_of_left=mse(true_y,pred_y_left)
    return (loss_of_right-loss_of_left)/(2*epsilon)


def gradient_step(w, b, x, true_y, learning_rate=0.05, epsilon=0.001):
    """对 w 和 b 同时完成一次梯度下降更新。

    必须先基于同一组旧的 w、b 分别计算两个数值梯度，再更新参数。
    返回 (new_w, new_b, new_loss)。
    """
    # TODO
    def loss(w,b):
        pred_y=w*x+b
        return mse(true_y,pred_y)
    right_of_w=loss(w+epsilon,b)
    left_of_w=loss(w-epsilon,b)
    right_of_b=loss(w,b+epsilon)
    left_of_b=loss(w,b-epsilon)
    gradient_of_w=(right_of_w-left_of_w)/(2*epsilon)
    gradient_of_b = (right_of_b - left_of_b) / (2 * epsilon)
    w=w-learning_rate*gradient_of_w
    b=b-learning_rate*gradient_of_b
    new_loss=loss(w,b)
    return w,b,new_loss


def train_linear_regression(
    x, true_y, steps=400, learning_rate=0.05, epsilon=0.001
):
    """从 w=0、b=0 开始，用数值梯度训练线性回归。

    每次参数更新后记录一次 loss。
    返回 (w, b, losses)，其中 losses 是包含每轮 loss 的一维 NumPy 数组。
    """
    w=0
    b=0
    def loss(w, b):
        pred_y = w * x + b
        return mse(true_y, pred_y)
    loss_of_list=[]
    for i in range(400):
        right_of_w = loss(w + epsilon, b)
        left_of_w = loss(w - epsilon, b)
        right_of_b = loss(w, b + epsilon)
        left_of_b = loss(w, b - epsilon)
        gradient_of_w = (right_of_w - left_of_w) / (2 * epsilon)
        gradient_of_b = (right_of_b - left_of_b) / (2 * epsilon)
        w = w - learning_rate * gradient_of_w
        b = b - learning_rate * gradient_of_b
        new_loss=loss(w,b)
        loss_of_list.append(new_loss)
    return w,b,np.array(loss_of_list)


def _test_prepare_column():
    result = prepare_column([1, 2, 3, 4])
    assert isinstance(result, np.ndarray), "结果必须是 ndarray"
    assert result.shape == (4, 1), "shape 应为 (4, 1)"
    assert np.issubdtype(result.dtype, np.floating), "数据类型应为浮点型"
    assert np.array_equal(result[:, 0], np.array([1.0, 2.0, 3.0, 4.0])), "数值不正确"


def _test_changed_block_copy():
    matrix = np.arange(1, 13).reshape(3, 4)
    original = matrix.copy()
    result = changed_block_copy(matrix)
    expected = np.array([[-1, 7], [10, 11]])
    assert np.array_equal(result, expected), "切片范围或修改位置不正确"
    assert np.array_equal(matrix, original), "不能修改传入的原矩阵"
    assert not np.shares_memory(result, matrix), "返回结果必须拥有独立数据"


def _test_center_each_column():
    matrix = np.array([[1.0, 10.0, 100.0], [3.0, 20.0, 200.0]])
    expected = np.array([[-1.0, -5.0, -50.0], [1.0, 5.0, 50.0]])
    result = center_each_column(matrix)
    assert np.allclose(result, expected), "没有让每列减去它自己的平均值"
    assert result.shape == matrix.shape, "结果 shape 不能改变"
    assert np.allclose(result.mean(axis=0), np.zeros(3)), "处理后每列平均值应为 0"


def _test_linear_layer():
    inputs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = np.array([[1.0, 10.0], [2.0, 20.0]])
    bias = np.array([0.5, -0.5])
    expected = inputs @ weights + bias
    result = linear_layer(inputs, weights, bias)
    assert np.allclose(result, expected), "线性层计算不正确"
    assert result.shape == (3, 2), "结果 shape 应为 (3, 2)"


def _test_mse():
    true_y = np.array([1.0, 3.0, 5.0])
    pred_y = np.array([2.0, 5.0, 3.0])
    result = mse(true_y, pred_y)
    assert np.isclose(result, 3.0), "MSE 计算不正确"
    assert np.ndim(result) == 0, "MSE 应返回一个标量"


def _test_numerical_gradient_w():
    x = np.array([1.0, 2.0, 3.0])
    true_y = np.array([3.0, 5.0, 7.0])
    result = numerical_gradient_w(0.5, 1.0, x, true_y)
    assert np.isclose(result, -14.0, atol=1e-5), "w 的中心差分梯度不正确"


def _test_gradient_step():
    x = np.array([1.0, 2.0])
    true_y = np.array([3.0, 5.0])
    new_w, new_b, new_loss = gradient_step(
        0.0, 0.0, x, true_y, learning_rate=0.1
    )
    assert np.isclose(new_w, 1.3, atol=1e-5), "w 的第一次更新不正确"
    assert np.isclose(new_b, 0.8, atol=1e-5), "b 的第一次更新不正确"
    expected_loss = np.mean((x * new_w + new_b - true_y) ** 2)
    assert np.isclose(new_loss, expected_loss), "返回的 new_loss 不正确"
    assert new_loss < np.mean(true_y**2), "更新一次后 loss 应当下降"


def _test_train_linear_regression():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    true_y = np.array([3.0, 5.0, 7.0, 9.0])
    w, b, losses = train_linear_regression(x, true_y)
    assert isinstance(losses, np.ndarray), "losses 必须是 NumPy 数组"
    assert losses.shape == (400,), "每一轮都应记录一次 loss"
    assert np.all(np.isfinite(losses)), "losses 中不能出现无穷或 NaN"
    assert losses[-1] < losses[0], "训练后的 loss 必须低于第一轮"
    assert losses[-1] < 1e-6, "最终 loss 还不够小"
    assert np.isclose(w, 2.0, atol=0.01), "最终 w 应接近 2"
    assert np.isclose(b, 1.0, atol=0.03), "最终 b 应接近 1"


def run_tests():
    tests = [
        ("1. 数组创建与 shape", _test_prepare_column),
        ("2. 切片与独立复制", _test_changed_block_copy),
        ("3. axis 与广播", _test_center_each_column),
        ("4. 矩阵乘法与线性层", _test_linear_layer),
        ("5. 均方误差 MSE", _test_mse),
        ("6. 单参数数值梯度", _test_numerical_gradient_w),
        ("7. 同时更新 w 和 b", _test_gradient_step),
        ("8. 完整线性回归训练", _test_train_linear_regression),
    ]

    passed = 0
    for name, test in tests:
        try:
            test()
        except Exception as error:
            detail = str(error) or error.__class__.__name__
            print(f"[未通过] {name}：{detail}")
        else:
            passed += 1
            print(f"[通过] {name}")

    print(f"\n结果：{passed}/{len(tests)}")
    if passed == len(tests):
        print("NumPy 阶段全部通过：下一步进入 PyTorch Tensor 与 Autograd！")
    else:
        print("先独立完成；把未通过的题号告诉我，我会针对你的写法讲解。")


if __name__ == "__main__":
    run_tests()
