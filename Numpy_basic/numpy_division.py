# 进行array的分割
import numpy as np
a=np.arange(12).reshape((3,4))
print(a)
print(np.split(a,3,axis=0)) #0是行 1是列  矩阵，要分几块，按行还是列
print(np.split(a,4,axis=1)) #无法不均等分割
# 跟合并的vstuck和hstuck对应的
print(np.vsplit(a, 3))
print(np.hsplit(a, 2))

# 不均等分割
print(np.array_split(a,2,axis=0))
print(np.array_split(a,3,axis=1))
