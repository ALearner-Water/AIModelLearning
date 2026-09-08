import torch
from torch import nn
x=torch.tensor(range(2),dtype=torch.float32).reshape(2,1)
true_y=2*x+3
#创建loss和opitimizer,model对象
model=nn.Linear(1,1)    #这里的类型都是float32的
loss_fn=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
for i in range(500):
    pred_y=model(x)
    loss=loss_fn(pred_y,true_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(model.weight.item(),model.bias.item(),loss_fn(model(x),true_y).item())