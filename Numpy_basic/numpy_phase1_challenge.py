"""NumPy 第一阶段通关练习。

覆盖莫烦 NumPy 2.1～2.8：
数组创建、属性、逐元素运算、矩阵乘法、索引与切片、聚合、合并、
分割以及独立复制。

完成规则：
1. 只修改标有 TODO 的 8 个函数，不要修改测试代码。
2. 尽量使用 NumPy 操作，不使用 for 循环逐个计算。
3. 运行：conda run -n dl-study python Numpy_basic/numpy_phase1_challenge.py
4. 看到“8/8，第一阶段全部通过”后，再进入 NumPy 第二阶段。
"""

import numpy as np


def create_grid():
    """创建包含 1～12、形状为 (3, 4) 的二维数组。"""
    a=np.arange(1,13).reshape((3,4))
    return a


def array_profile(values):
    """把 values 转成数组，返回 (维数, 形状, 元素数量, 数据类型)。"""
    # TODO: 返回 ndarray 的 ndim、shape、size 和 dtype
    b=np.array(values)
    return b.ndim,b.shape,b.size,b.dtype    #属性不加括号


def elementwise_results(values):
    """返回数组逐元素乘 2 的结果，以及逐元素平方的结果。"""
    # TODO: 返回 (doubled, squared)
    a=np.array(values)
    return a*2,a**2

def multiply_matrices(left, right):
    """返回两个矩阵的逐元素乘法结果和矩阵乘法结果。"""
    # TODO: 返回 (elementwise_product, matrix_product)
    return left*right,left.dot(right)


def extract_block(matrix):
    """取得第 2 行至末行、第 2～3 列组成的子数组。"""
    # 提示：这里的“第 2 行”对应索引 1；切片的结束位置不包含在结果中
    return matrix[1:,1:3]       #取行列一起写

def axis_statistics(matrix):
    """返回整体平均值、每列之和、每行最大值。"""
    # TODO: 返回 (overall_mean, column_sums, row_maximums)
    # 提示：每列之和使用 axis=0；每行最大值使用 axis=1
    return matrix.mean(),matrix.sum(axis=0),matrix.max(axis=1)


def merge_rows(first, second):
    """把两个一维数组分别上下合并和左右合并。"""
    # TODO: 先把它们变成形状为 (1, n) 的二维数组
    # 返回 (vertical, horizontal)，目标形状分别为 (2, n) 和 (1, 2*n)
    first=first[np.newaxis,:]
    second=second[np.newaxis,:]
    return np.vstack((first,second)),np.hstack((first,second))


def split_and_copy(matrix):
    """完成分割与独立复制。

    把矩阵沿列平均分成左右两块；再独立复制左半块，将复制品左上角改为 -1。
    返回 (left, right, changed_copy)。原来的 left 不能被修改。
    """
    # TODO: 可以使用 np.hsplit() 和 copy()
    all=np.hsplit(matrix,2)
    left=all[0]
    right=all[1]
    copy=np.copy(left)
    copy[0][0]=-1
    return left,right,copy

def _test_create_grid():
    result = create_grid()
    expected = np.arange(1, 13).reshape(3, 4)
    assert isinstance(result, np.ndarray), "结果必须是 ndarray"
    assert np.array_equal(result, expected), "数值或形状不正确"


def _test_array_profile():
    result = array_profile([[1, 2, 3], [4, 5, 6]])
    assert isinstance(result, tuple) and len(result) == 4, "需要返回四项组成的元组"
    ndim, shape, size, dtype = result
    assert ndim == 2, "ndim 应为 2"
    assert shape == (2, 3), "shape 应为 (2, 3)"
    assert size == 6, "size 应为 6"
    assert np.issubdtype(dtype, np.integer), "dtype 应是整数类型"


def _test_elementwise_results():
    doubled, squared = elementwise_results([1, 2, 3, 4])
    assert np.array_equal(doubled, np.array([2, 4, 6, 8])), "乘 2 的结果不正确"
    assert np.array_equal(squared, np.array([1, 4, 9, 16])), "平方结果不正确"


def _test_multiply_matrices():
    left = np.array([[1, 2], [3, 4]])
    right = np.array([[5, 6], [7, 8]])
    elementwise, matrix_product = multiply_matrices(left, right)
    assert np.array_equal(elementwise, np.array([[5, 12], [21, 32]])), "逐元素乘法不正确"
    assert np.array_equal(matrix_product, np.array([[19, 22], [43, 50]])), "矩阵乘法不正确"


def _test_extract_block():
    matrix = np.arange(1, 13).reshape(3, 4)
    result = extract_block(matrix)
    assert np.array_equal(result, np.array([[6, 7], [10, 11]])), "索引或切片范围不正确"


def _test_axis_statistics():
    matrix = np.arange(1, 13).reshape(3, 4)
    overall_mean, column_sums, row_maximums = axis_statistics(matrix)
    assert np.isclose(overall_mean, 6.5), "整体平均值不正确"
    assert np.array_equal(column_sums, np.array([15, 18, 21, 24])), "每列之和不正确"
    assert np.array_equal(row_maximums, np.array([4, 8, 12])), "每行最大值不正确"


def _test_merge_rows():
    vertical, horizontal = merge_rows(np.array([1, 2, 3]), np.array([4, 5, 6]))
    assert np.array_equal(vertical, np.array([[1, 2, 3], [4, 5, 6]])), "上下合并不正确"
    assert np.array_equal(horizontal, np.array([[1, 2, 3, 4, 5, 6]])), "左右合并不正确"


def _test_split_and_copy():
    matrix = np.arange(12).reshape(3, 4)
    left, right, changed_copy = split_and_copy(matrix)
    expected_left = np.array([[0, 1], [4, 5], [8, 9]])
    expected_right = np.array([[2, 3], [6, 7], [10, 11]])
    assert np.array_equal(left, expected_left), "左半块不正确，或被复制品的修改影响了"
    assert np.array_equal(right, expected_right), "右半块不正确"
    expected_changed = expected_left.copy()
    expected_changed[0, 0] = -1
    assert np.array_equal(changed_copy, expected_changed), "复制品的修改结果不正确"
    assert not np.shares_memory(left, changed_copy), "复制品必须拥有独立数据"


def run_tests():
    """运行全部测试并显示每一题的结果。"""
    tests = [
        ("1. 创建与变形", _test_create_grid),
        ("2. 数组属性", _test_array_profile),
        ("3. 逐元素运算", _test_elementwise_results),
        ("4. 两种乘法", _test_multiply_matrices),
        ("5. 索引与切片", _test_extract_block),
        ("6. axis 聚合", _test_axis_statistics),
        ("7. 数组合并", _test_merge_rows),
        ("8. 分割与复制", _test_split_and_copy),
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
        print("第一阶段全部通过：可以进入 NumPy 第二阶段！")
    else:
        print("继续完成 TODO；不理解的题目可以把代码和输出发给我。")


if __name__ == "__main__":
    run_tests()
