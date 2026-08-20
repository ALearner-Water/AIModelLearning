import numpy as np
#numpy是做矩阵运算 所以先要创建列表 然后使用np.array转成numpy可以识别的数组
array=np.array([[1,2,3],
                [4,5,6]])
print(array)    #会输出矩阵
print(type(array))
print(f"numpy of dim: {array.ndim}")    #维度是多少
print(f"numpy of shape: {array.shape}") #形状是几行几列
print(f"numpy of size: {array.size}")   #有多少个元素