import torch
import numpy as np
tensor=torch.tensor(range(24),dtype=torch.float32).reshape((2,3,4)).reshape((6,4))
x=torch.tensor(range(12)).reshape((3,4))
y=torch.tensor(range(8)).reshape((4,2))
print((x @ y).shape)
a=torch.tensor(range(6)).reshape((2,3))
b=torch.tensor(range(6,12)).reshape((2,3))
print(torch.cat((a,b),dim=0).shape) #行
print(torch.cat((a, b), dim=1).shape)   #列
print(torch.stack((a,b),dim=0).shape)

p=torch.tensor([1.0,2.0],requires_grad=True)
p=p*2
q=p.detach().cpu().numpy()
print(q.shape,q.dtype,q.device)
