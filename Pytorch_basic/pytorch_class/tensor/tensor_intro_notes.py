import torch
import numpy as np
#创建张量  就是numpy里面的数组
a=torch.arange(12).reshape((3,4))
# print(a)            
# print(a.shape)      #形状
# print(torch.numel(a))   #元素总数
# b=torch.zeros((2,3,4))    #全0矩阵
# print(b)
# c=torch.ones((2,3,4))     #全1矩阵
# print(c)

#跟numpy一样可以直接相应元素+，-，*，**，/

#连结张量
e=torch.arange(12,24).reshape((3,4))
hang=torch.cat((a,e),dim=0)    #按行上下合并
lie=torch.cat((a,e),dim=1)    #按列上下合并
# print(hang)
# print(lie)
# print(a==e)         #可以使用逻辑运算符判断两个张量
# print(torch.sum(a)) #求和

#也有广播机制，访问张量也是跟numpy一样访问
one=torch.arange(3).reshape((3,1))
two=torch.arange(2).reshape((1,2))
#print(one+two)

#numpy数组torch张量的转换
q=a.numpy()     #转成numpy形式
# print(q,type(q))
p=torch.tensor(q)   #转成tensor形式
# print(p,type(p))

#转成标量
w=torch.tensor([2.0])
# print(w.item())     #将张量转成标量
# print(int(w))       #可以直接使用强制类型转换