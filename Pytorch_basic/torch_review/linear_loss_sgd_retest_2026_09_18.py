import torch
from torch import nn

x=torch.tensor(range(15),dtype=torch.float32).reshape(5,3)
model=nn.Linear(3,2)
y=model(x)
print(x.shape,y.shape,model.weight.shape,model.bias.shape)
true_y=torch.tensor(range(10),dtype=torch.float32).reshape(5,2)
loss1=torch.mean((y-true_y)**2)
fn_loss=nn.MSELoss()
loss2=fn_loss(y,true_y)
print(loss1.item())
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
for i in range(500):
    pred_y=model(x)
    loss=fn_loss(pred_y,true_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
loss3=fn_loss(model(x),true_y)  
print(loss3.item())
