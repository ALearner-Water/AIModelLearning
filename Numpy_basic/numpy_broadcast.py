# numpy的广播记机制
import numpy as np
# 1. 数组与数之间的计算
a=np.random.randint(0,5,size=(3,5))
print(a)
# 直接相乘
# print(a*2)

# 2. 数组与数组的运算
# 形状相同直接相加 ，形状不同不能相加
b = np.random.randint(0, 5, size=(3, 5))
# print(b)
# print(a+b)  #直接对应相加

# 例外 行(列)数相同 列(行)数为1的可以相加，每一列依次与列(行)向量相加  ——> 广播
c =np.random.randint(0, 5, size=(3, 1))
# print(c)
# print(a+c)
d = np.random.randint(0, 5, size=(1, 5))
# print(d)
# print(a+d)

# 广播原则  如果两个数组从后往前数其长度是相同的或者有其中一方是1，则可以进行运算  （广播兼容）
# 例子  shape(2,8,3)and shape(8,2)     从后往前看是3！=2
#      shape(2,8,1)and shape(1,8)     有一方始终是1
