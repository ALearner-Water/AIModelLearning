# view 和 transpose
# view 要求当前内存布局与目标 shape 兼容；
# reshape 不兼容时会自动复制数据。
import torch
import numpy as np
original_data=torch.arange(1,13)
matrix = original_data.view(3, 4)  # 跟reshape一样若数据排列不连续，就不能再次变形，reshape可以，做法是直接再复制一份
transposed=matrix.transpose(0,1)    #交换0维和1维。可以直接使用matrix.T但是transpose可以用于指定维度交换

print(f"original={original_data}")
print(f"matrix={matrix}")
print(f"transposed={transposed}")

print(f"original_shape={original_data.shape},matrix_data={matrix.shape},transposed_shape={transposed.shape}")

print("\n内存地址：")
print(original_data.data_ptr())
print(matrix.data_ptr())
print(transposed.data_ptr())

print("\n是否连续：")
print(matrix.is_contiguous())
print(transposed.is_contiguous())   #转置后内存不连续
