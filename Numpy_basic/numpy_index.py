# 通过索引进行相应的操作
import numpy as np
a=np.arange(3,15).reshape(3,4)
print(a[1])  # [ 7  8  9 10] 直接取到一整行
print(a[1,1])   #取到对应值
print(a[1,:])   #只取第一行
print(a[:,1])   #只取第一列 像字符串切片一样
print(a[0,1:3]) #取第0行的第一列和第二列

#for循环
for row in a:   #取出每一行进行迭代
    print(row)

for item in a.flat: #找出每个元素  a.flat是把矩阵变成一维数组
    print(item)