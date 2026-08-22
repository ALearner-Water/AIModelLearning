"""PyTorch Tensor 第一阶段通关练习。

覆盖内容：Tensor 创建、属性、reshape、索引与切片、逐元素运算、
矩阵乘法、拼接，以及 dim/keepdim 聚合。

规则：
1. 只修改标有 TODO 的 8 个函数，不要修改测试代码。
2. 使用 PyTorch 完成，不要先转成 NumPy 计算。
3. 除测试代码外，不需要使用 for 循环。

运行：
    conda run -n dl-study python Pytorch_basic/pytorch_tensor_phase1_challenge.py
"""

import torch


def create_grid():
    """创建包含 1～12、shape 为 (3, 4)、dtype 为 float32 的 Tensor。"""
    return torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)

def tensor_profile(values):
    """把 values 转为 Tensor，返回 (维数, shape, 元素数量, dtype)。"""
    # TODO
    a=torch.tensor(values)
    return a.dim(),a.shape,torch.numel(a),a.dtype


def reshape_batch(tensor):
    """把包含 24 个元素的 Tensor 整理为 shape (2, 3, 4)。"""
    # TODO
    return torch.reshape((2,3,4))


def extract_block(tensor):
    """取得第 2 行至末行、第 2～3 列组成的子 Tensor。"""
    # TODO
    return tensor[1:,1:3]


def elementwise_results(left, right):
    """返回两个 Tensor 的逐元素相加、逐元素相乘以及 left 的平方。"""
    # TODO：返回 (added, multiplied, squared)
    return left+right,left*right,left**2


def multiply_matrices(left, right):
    """返回 left 与 right 的矩阵乘法结果。"""
    # TODO
    return left @ right


def concatenate_tensors(first, second):
    """分别沿第 0 维和第 1 维拼接两个二维 Tensor。"""
    # TODO：返回 (along_dim0, along_dim1)
    return torch.cat([first,second],dim=0),torch.cat([first,second],dim=1)


def axis_statistics(matrix):
    """返回每列之和，以及每行平均值。

    每行平均值必须保留二维形状，结果 shape 应为 (行数, 1)。
    """
    # TODO：使用 dim 和 keepdim
    return matrix.sum(dim=0),matrix.mean(dim=1,keepdim=True)        #括号里表示把谁去掉

def _test_create_grid():
    result = create_grid()
    expected = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    assert isinstance(result, torch.Tensor), "结果必须是 Tensor"
    assert torch.equal(result, expected), "数值或 shape 不正确"
    assert result.dtype == torch.float32, "dtype 必须是 torch.float32"


def _test_tensor_profile():
    result = tensor_profile([[1, 2, 3], [4, 5, 6]])
    assert isinstance(result, tuple) and len(result) == 4, "需要返回四项组成的元组"
    ndim, shape, number, dtype = result
    assert ndim == 2, "维数应为 2"
    assert shape == torch.Size([2, 3]), "shape 应为 torch.Size([2, 3])"
    assert number == 6, "元素数量应为 6"
    assert dtype == torch.int64, "由这些整数创建时，dtype 应为 torch.int64"


def _test_reshape_batch():
    source = torch.arange(24)
    result = reshape_batch(source)
    assert result.shape == (2, 3, 4), "shape 应为 (2, 3, 4)"
    assert torch.equal(result.reshape(-1), source), "reshape 后元素或顺序发生了变化"


def _test_extract_block():
    tensor = torch.arange(1, 13).reshape(3, 4)
    result = extract_block(tensor)
    expected = torch.tensor([[6, 7], [10, 11]])
    assert torch.equal(result, expected), "索引或切片范围不正确"


def _test_elementwise_results():
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    added, multiplied, squared = elementwise_results(left, right)
    assert torch.equal(added, torch.tensor([[11.0, 22.0], [33.0, 44.0]])), "加法不正确"
    assert torch.equal(multiplied, torch.tensor([[10.0, 40.0], [90.0, 160.0]])), "逐元素乘法不正确"
    assert torch.equal(squared, torch.tensor([[1.0, 4.0], [9.0, 16.0]])), "平方不正确"


def _test_multiply_matrices():
    left = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    right = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = multiply_matrices(left, right)
    expected = torch.tensor([[22.0, 28.0], [49.0, 64.0]])
    assert torch.equal(result, expected), "矩阵乘法结果不正确"
    assert result.shape == (2, 2), "结果 shape 应为 (2, 2)"


def _test_concatenate_tensors():
    first = torch.tensor([[1, 2], [3, 4]])
    second = torch.tensor([[5, 6], [7, 8]])
    dim0, dim1 = concatenate_tensors(first, second)
    expected_dim0 = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    expected_dim1 = torch.tensor([[1, 2, 5, 6], [3, 4, 7, 8]])
    assert torch.equal(dim0, expected_dim0), "沿 dim=0 拼接不正确"
    assert torch.equal(dim1, expected_dim1), "沿 dim=1 拼接不正确"


def _test_axis_statistics():
    matrix = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    column_sums, row_means = axis_statistics(matrix)
    assert torch.equal(column_sums, torch.tensor([11.0, 22.0, 33.0])), "每列之和不正确"
    assert torch.equal(row_means, torch.tensor([[2.0], [20.0]])), "每行平均值不正确"
    assert column_sums.shape == (3,), "每列之和的 shape 应为 (3,)"
    assert row_means.shape == (2, 1), "每行平均值必须保留为 (2, 1)"


def run_tests():
    tests = [
        ("1. Tensor 创建", _test_create_grid),
        ("2. Tensor 属性", _test_tensor_profile),
        ("3. reshape", _test_reshape_batch),
        ("4. 索引与切片", _test_extract_block),
        ("5. 逐元素运算", _test_elementwise_results),
        ("6. 矩阵乘法", _test_multiply_matrices),
        ("7. Tensor 拼接", _test_concatenate_tensors),
        ("8. dim 与 keepdim", _test_axis_statistics),
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
        print("Tensor 第一阶段全部通过：下一步学习 NumPy 转换、device 与 Autograd！")
    else:
        print("继续完成 TODO；不理解的题目可以把题号和代码发给我。")


if __name__ == "__main__":
    run_tests()
