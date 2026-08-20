# numpy的基础运算
# 一维数组的运算
import numpy as np
a=np.array([20,30,40,50])   #直接做加减法  
b=np.arange(4)
# print(a,b)
# print(a-b,a+b,a**2)     #a**2是按元素进行运算，不是矩阵的幂
# print(np.sin(a))        #求三角函数
# print(b < 3)            # [ True  True  True False] 直接可以比较大小

# 矩阵的运算
c=np.array([[1,2],
            [3,4]])
d=np.arange(4).reshape(2,2)
#print(f"{c}\n{d}")
#print(f"{c*d}\n{d*c}")  #这个是逐个相乘，顺序不影响结果
dot1=np.dot(c,d)        #np.dot(a,b)才是矩阵乘法  a.dot(b)也是可以的
dot2=np.dot(d,c)
# print(dot1)
# print(dot2)

#矩阵中对个元素进行运算的方式  聚合则是1是行 0是列
r=np.random.random((2,2)) #随机出2行2列的矩阵
print(r)
# print(np.sum(r,axis=1))     #每一列求和   axis=1是在行做操作  axis=0是在列数做操作
print(np.max(r,axis=0))     #对列做操作
# print(np.min(r))
    