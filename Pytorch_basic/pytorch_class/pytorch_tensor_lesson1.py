# NumPy 与 Tensor 转换
import numpy as np
import torch 

numpy_array=np.array([1,2,3],dtype=np.float32)

# 1.直接复制数据
tensor_copy = torch.tensor(numpy_array)  # 复制一份独立数据，之后互不影响。

# 2. 数据共享
tensor_shared=torch.from_numpy(numpy_array) #与 NumPy 共用同一块内存，一边修改，另一边也会变化

print(f"修改前：numpy_array={numpy_array},tensor_copy={tensor_copy},tensor_shared={tensor_shared}")
numpy_array[0]=99
print(f"修改后：numpy_array={numpy_array},tensor_copy={tensor_copy},tensor_shared={tensor_shared}")
