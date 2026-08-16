# Python → NumPy 学习指南

目标：先掌握 NumPy 所依赖的 Python 基础，再开始数组计算。

## 如何使用

每天学习 30～45 分钟：先阅读示例，手动输入代码，再完成当天练习。不要只复制粘贴。

运行 Python 文件：

```powershell
python 文件名.py
```

## 第 1 天：变量、类型与运算

```python
name = "小明"
age = 18
temperature = 23.5
is_raining = False

print(type(name))
print(age + 1)
print(temperature * 2)
```

需要理解：`str`、`int`、`float`、`bool`，以及 `+ - * / // % **`。

练习：根据长和宽计算长方形的面积与周长。

## 第 2 天：列表、索引与切片

```python
scores = [82, 91, 76, 88]

print(scores[0])
print(scores[-1])
print(scores[1:3])

scores.append(95)
scores[0] = 85
```

记住：索引从 `0` 开始；切片 `[开始:结束]` 不包含结束位置。

练习：创建 5 个温度的数据列表，输出第一个、最后一个以及中间三个温度。

## 第 3 天：循环与条件判断

```python
temperatures = [18, 21, 25, 19, 27]
warm_days = []

for temperature in temperatures:
    if temperature >= 20:
        warm_days.append(temperature)

print(warm_days)
```

需要理解：循环变量会依次取得列表中的每个元素；缩进表示代码层级。

练习：筛选成绩列表中所有大于等于 60 的成绩。

## 第 4 天：函数

```python
def average(values):
    return sum(values) / len(values)


result = average([2, 4, 6, 8])
print(result)
```

函数由输入（参数）、处理过程和输出（`return`）组成。

练习：编写 `maximum(values)`，返回列表最大值；先自己循环实现，再使用 `max()`。

## 第 5 天：列表推导式

普通循环：

```python
squares = []
for number in [1, 2, 3, 4]:
    squares.append(number ** 2)
```

列表推导式：

```python
squares = [number ** 2 for number in [1, 2, 3, 4]]
```

带条件筛选：

```python
even_numbers = [number for number in range(10) if number % 2 == 0]
```

NumPy 会把这种“对所有元素执行同一运算”的思想变得更直接。

## 第 6 天：模块与错误信息

```python
import math

print(math.sqrt(16))
```

阅读报错时先看最后一行：它通常包含错误类型和原因。

常见错误：

- `NameError`：变量名不存在或拼错。
- `TypeError`：数据类型不适合当前操作。
- `IndexError`：列表索引超出范围。
- `IndentationError`：缩进有误。
- `ModuleNotFoundError`：模块尚未安装或名称写错。

练习：故意触发上述前三种错误，观察报错，再修复。

## 第 7 天：NumPy 前置综合练习

完成 `python_numpy_readiness_challenge.py` 中的四个函数：

1. 摄氏温度转华氏温度。
2. 计算列表平均值。
3. 根据阈值筛选数字。
4. 计算列表中所有数字的平方。

运行：

```powershell
python python_numpy_readiness_challenge.py
```

输出“全部通过”后，再开始 NumPy。

## NumPy 入门时要学习的内容

```python
import numpy as np

numbers = np.array([1, 2, 3, 4])
print(numbers * 2)
print(numbers.mean())
print(numbers[numbers >= 3])
```

对应的 Python 基础分别是：列表、批量运算、函数调用、条件筛选。后续重点将是数组形状、维度、索引、数据类型、广播和聚合运算。

## 通关标准

在不看答案的情况下能够：

- 创建并修改列表，正确使用索引与切片。
- 用循环逐个处理数据。
- 用 `if` 筛选数据。
- 编写有参数和返回值的函数。
- 看懂简单的列表推导式。
- 根据报错最后一行定位常见问题。
- 独立通过通关练习中的全部断言。
