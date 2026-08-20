"""NumPy 第二阶段：广播与矩阵 shape 通关练习。

只修改标有 TODO 的 6 个函数，不要修改测试代码。
尽量使用 NumPy 直接运算，不使用 for 循环逐个计算。

运行：
    conda run -n dl-study python Numpy_basic/numpy_phase2_broadcasting_challenge.py
"""

import numpy as np


def add_row_bias(matrix, bias):
    """给矩阵的每一行加上同一个一维 bias。"""
    # matrix.shape == (3, 4)，bias.shape == (4,)
    return matrix+bias


def add_column_bias(matrix, bias):
    """让 bias 中的每个数分别加到矩阵对应的一整行。"""
    # matrix.shape == (3, 4)，传入的 bias.shape == (3,)
    # TODO: 先给 bias 增加一个维度，使其 shape 变成 (3, 1)
    return matrix+bias[:,np.newaxis]


def center_each_row(matrix):
    """每一行都减去该行自己的平均值。"""
    # TODO: 求行平均值时使用 axis=1，并保留二维形状以便广播
    average=np.mean(matrix,axis=1)[:,np.newaxis]    #求出来的mean是一维数组无法进行运算
    return matrix-average


def linear_layer(inputs, weights, bias):
    """实现二维输入的线性层：outputs = inputs @ weights + bias。"""
    # inputs: (batch, in_features)
    # weights: (in_features, out_features)
    # bias: (out_features,)
    # TODO
    return inputs @ weights + bias  # 矩阵乘法 @    向量点积：np.dot() 或 @

def batched_linear_layer(inputs, weights, bias):
    """实现三维批量输入的线性层，仍然使用 inputs @ weights + bias。"""
    # inputs: (batch, sequence, in_features)
    # weights: (in_features, out_features)
    # bias: (out_features,)
    # TODO
    return inputs @ weights + bias


def predict_shapes():
    """仅根据 shape 规则填写四个结果 shape，不需要创建大型数组。

    按顺序返回：
    1. (3, 4) + (4,)        #(4,)代表无行无列 看情况可以用广播直接计算，有时需要人为增加维度
    2. (3, 4) + (3, 1)
    3. (16, 20, 512) @ (512, 128)
    4. (8, 32, 64) @ (64, 10) + (10,)
    """
    # TODO: 返回由四个 shape 元组组成的元组
    return (3,4),(3,4),(16,20,128),(8,32,10)


def _test_add_row_bias():
    matrix = np.arange(12).reshape(3, 4)
    bias = np.array([10, 20, 30, 40])
    expected = np.array(
        [[10, 21, 32, 43], [14, 25, 36, 47], [18, 29, 40, 51]]
    )
    result = add_row_bias(matrix, bias)
    assert np.array_equal(result, expected), "一维 bias 没有正确加到每一行"


def _test_add_column_bias():
    matrix = np.arange(12).reshape(3, 4)
    bias = np.array([100, 200, 300])
    expected = np.array(
        [[100, 101, 102, 103], [204, 205, 206, 207], [308, 309, 310, 311]]
    )
    result = add_column_bias(matrix, bias)
    assert np.array_equal(result, expected), "bias 需要先从 (3,) 变成 (3, 1)"


def _test_center_each_row():
    matrix = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    expected = np.array([[-1.0, 0.0, 1.0], [-10.0, 0.0, 10.0]])
    result = center_each_row(matrix)
    assert np.allclose(result, expected), "应让每一行减去它自己的平均值"
    assert np.allclose(result.mean(axis=1), np.zeros(2)), "处理后每行平均值应为 0"


def _test_linear_layer():
    inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    weights = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    bias = np.array([10.0, 20.0])
    expected = np.array([[14.0, 25.0], [20.0, 31.0]])
    result = linear_layer(inputs, weights, bias)
    assert np.allclose(result, expected), "请按 inputs @ weights + bias 计算"
    assert result.shape == (2, 2), "结果 shape 应为 (2, 2)"


def _test_batched_linear_layer():
    inputs = np.arange(24.0).reshape(2, 3, 4)
    weights = np.arange(20.0).reshape(4, 5)
    bias = np.arange(5.0)
    expected = inputs @ weights + bias
    result = batched_linear_layer(inputs, weights, bias)
    assert np.allclose(result, expected), "三维输入也可直接使用 @ 和 bias 广播"
    assert result.shape == (2, 3, 5), "结果 shape 应为 (2, 3, 5)"


def _test_predict_shapes():
    expected = ((3, 4), (3, 4), (16, 20, 128), (8, 32, 10))
    result = predict_shapes()
    assert result == expected, "至少有一个 shape 推导不正确"


def run_tests():
    tests = [
        ("1. 行方向广播", _test_add_row_bias),
        ("2. 列向量广播", _test_add_column_bias),
        ("3. axis 与广播", _test_center_each_row),
        ("4. 二维线性层", _test_linear_layer),
        ("5. 三维批量线性层", _test_batched_linear_layer),
        ("6. shape 推导", _test_predict_shapes),
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
        print("第二阶段全部通过：下一步实现 MSE 和 NumPy 线性回归！")
    else:
        print("把未通过的题号告诉我，我会针对你的写法讲解。")


if __name__ == "__main__":
    run_tests()
