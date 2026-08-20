import numpy as np
#简单定义numpy
a=np.array([1,2,3],dtype=int)   #dtype来指定数据类型
print(a.dtype)
print(a)

b=np.array([[1,2,3],    #生成矩阵
            [3,4,5]])
print(b)

c=np.zeros((2,3))   #生成零矩阵
print(c)

d=np.ones((3,4))    #生成全1矩阵
print(d)

e=np.arange(10).reshape(2,5)       #与range一样都是从几到几
print(e)                           #reshape可以重新定义行列