import torch
from torch import nn
pred_y=torch.tensor([[1.0],[3.0]])
true_y=torch.tensor([[2.0],[1.0]])

mse_loss=torch.mean((pred_y-true_y)**2)
loss=nn.MSELoss()   #创建对象 不是直接赋值计算
MSE_loss=loss(pred_y,true_y)
print(MSE_loss.shape)
print(mse_loss.item())
print(MSE_loss.item())
