"""Python 基础练习：为学习 NumPy 做准备。

运行方式：python python_foundation_for_numpy.py
建议先预测每一段的输出，再运行核对。
"""


def lesson_variables():
    """变量、数字和基本运算。"""
    length = 8
    width = 5
    area = length * width
    perimeter = 2 * (length + width)

    print("1. 变量与运算")
    print(f"面积：{area}，周长：{perimeter}")


def lesson_lists_and_slices():
    """列表、索引和切片；NumPy 数组也使用相似语法。"""
    scores = [82, 91, 76, 88]

    print("\n2. 列表、索引与切片")
    print("第一个成绩：", scores[0])
    print("最后一个成绩：", scores[-1])
    print("中间两个成绩：", scores[1:3])


def lesson_loops_and_conditions():
    """逐个处理数据，并按条件筛选。"""
    scores = [82, 91, 76, 88]
    passed = []

    for score in scores:
        if score >= 80:
            passed.append(score)

    print("\n3. 循环与条件判断")
    print("80 分及以上：", passed)


def normalize(values):
    """把一组数除以其中的最大值，返回新列表。"""
    maximum = max(values)
    return [value / maximum for value in values]


def lesson_functions_and_comprehensions():
    """函数和列表推导式；为 NumPy 的批量运算建立直觉。"""
    numbers = [2, 4, 6, 8]
    squares = [number**2 for number in numbers]
    normalized = normalize(numbers)

    print("\n4. 函数与列表推导式")
    print("平方：", squares)
    print("归一化：", normalized)


def self_check():
    """自动检查关键知识点。没有报错就表示示例结果正确。"""
    numbers = [2, 4, 6, 8]

    assert numbers[1:3] == [4, 6]
    assert [number**2 for number in numbers] == [4, 16, 36, 64]
    assert normalize(numbers) == [0.25, 0.5, 0.75, 1.0]

    print("\n自检通过！")


def main():
    lesson_variables()
    lesson_lists_and_slices()
    lesson_loops_and_conditions()
    lesson_functions_and_comprehensions()
    self_check()


if __name__ == "__main__":
    main()


# 独立练习（完成后可把代码发来检查）：
# 1. 创建 temperatures = [18, 21, 25, 19, 27]。
# 2. 使用循环或列表推导式，将每个温度转换为华氏度：摄氏度 * 1.8 + 32。
# 3. 筛选出大于等于 20 摄氏度的温度。
# 4. 编写 average(values) 函数，返回列表平均值。
