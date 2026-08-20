import numpy as np
a=np.arange(2,14).reshape(3,4)
print(a)

# 找索引
print(np.argmax(a),a.argmax())  #对于已经定义好的矩阵有两种写法 其他方法也是一样
print(np.argmin(a),a.argmin())

# 求平均值  运算均可以通过axis来限定是行运算还是；列运算  1为行  0为列
print(np.mean(a))
print(np.median(a)) #求中位数

# 累加
print(np.cumsum(a))  # [ 2  5  9 14 20 27 35 44 54 65 77 90]

#做差
print(np.diff(a))  #两个元素做差

#逐行排序
b=np.array([[1,3,2],
            [4,1,9]])
print(np.sort(b))

#矩阵的转置
print(a.T)
print(np.transpose(a))

#矩阵的赋值  clip
print(np.clip(a,5,9))   #小于5的都变成5 大于9的都变成9，之间的不变