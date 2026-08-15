"""Python -> NumPy 入门通关练习。

请完成下面 4 个函数，然后运行：
    python python_numpy_readiness_challenge.py

看到“全部通过”后，就具备开始学习 NumPy 数组的 Python 基础。
"""


def celsius_to_fahrenheit(celsius):
    """把一个摄氏温度转换为华氏温度。公式：C * 1.8 + 32。"""
    # TODO: 删除下一行，并返回计算结果。
    raise NotImplementedError


def average(values):
    """返回一组数字的平均值。可使用 sum() 和 len()。"""
    # TODO: 删除下一行，并返回平均值。
    raise NotImplementedError


def filter_at_least(values, minimum):
    """返回大于或等于 minimum 的所有数字组成的新列表。"""
    # TODO: 删除下一行，可使用循环或列表推导式。
    raise NotImplementedError


def square_all(values):
    """返回所有数字的平方组成的新列表。"""
    # TODO: 删除下一行，可使用循环或列表推导式。
    raise NotImplementedError


def run_tests():
    """基础通关测试；不需要修改这个函数。"""
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(20) == 68
    assert average([2, 4, 6, 8]) == 5
    assert filter_at_least([18, 21, 25, 19, 27], 20) == [21, 25, 27]
    assert square_all([2, 4, 6, 8]) == [4, 16, 36, 64]
    print("全部通过：可以开始学习 NumPy！")


if __name__ == "__main__":
    run_tests()
