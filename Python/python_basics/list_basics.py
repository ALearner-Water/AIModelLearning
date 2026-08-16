"""Python 列表基础练习。"""


# Python 列表可以保存不同类型的元素，但实际项目通常保持元素类型一致。
items = [1, "nihao", 2.0, "hello"]

# 列表是一个整体；使用 * 解包后，sep 才会作用于每个元素。
print(*items, sep="   ")

# 创建一个长度为 15、所有元素均为 3 的列表。
repeated_numbers = [3] * 15
print(repeated_numbers)

# 直接遍历列表元素。
for item in items:
    if isinstance(item, str):
        print(item, end="  ")
    else:
        print("非字符串", end="  ")
print()

# 使用索引遍历列表。
for index in range(len(items)):
    print(items[index], end="  ")
print()
