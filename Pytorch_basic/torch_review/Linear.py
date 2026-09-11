import torch
from torch import nn
x=torch.tensor([2.0,3.0,6.0],dtype=torch.float32).reshape((1,3))
w=torch.tensor([[1.0,2.0],
                [4.0,5.0],
                [5.0,6.0]])
b=torch.tensor([[1.0,2.0]])
true_y=w@x+b
#初始化linear
model=nn.Linear(3,2)
fn_loss=nn.MSELoss()
opitimizer=torch.optim.SGD(model.parameters(),lr=0.01)

#开始训练
for i in range(500):
    pred_y=model(x)
    opitimizer.zero_grad()
    loss=fn_loss(pred_y,true_y)
    opitimizer.step()