import torch
from torch import nn
x=torch.tensor(range(1,13),dtype=torch.float32,requires_grad=True).reshape((3,4))
y = torch.tensor(range(13, 25), dtype=torch.float32,requires_grad=True).reshape((3, 4))
x_last2=x[1,2:]
reesult=x @ x.T       #shape 3,3
cat=torch.cat([x,y],dim=0)  #6,4
stack=torch.stack([x,y],dim=0) #2,3,4
print(x.transpose(0,1)) #忘记view的作用了  transport就是维度互换
print(x.dtype,x.device)
z=(2 * x + y)
z.sum().backward()
numpy_result=z.detach().cpu().numpy()
print(z.shape)
