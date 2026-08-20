# 合并两个矩阵
import numpy as np
a=np.array([1,2,3])[np.newaxis,:]
b = np.array([4, 5, 6])[np.newaxis, :]
print(np.vstack((a,b)))     #上下合并
print(np.hstack((a,b)))     #左右合并
# print(a.T)      #对于一维数组直接转置是不可以的 需要新增维度
print(a[np.newaxis,:].T)  #新增维度之后就变成矩阵了所以可以进行转置  这是增加了行的维度

# 也可以使用concatenate来要求从哪个维度来合并
c=np.concatenate((a,b,b,a),axis=0)  #要这样做a,b必须先升维 一维数组不支持 =0为上下拼接
d=np.concatenate((a,b,b,a),axis=1)  #=1为左右拼接 0是行 1是列
print(c)
print(d)