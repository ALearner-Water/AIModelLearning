# cat 和 stack
import torch
first=torch.tensor([[1,2],
                    [3,4]])
second=torch.tensor([[5,6],
                     [7,8]])

#cat是直接拼接
cat_dim0=torch.cat([first,second],dim=0)
cat_dim1=torch.cat([first,second],dim=1)
print(cat_dim0)
print(cat_dim1)

#stack是新增维度    必须要两个形状一样
stack_dim0=torch.stack([first,second],dim=0)
print(stack_dim0)