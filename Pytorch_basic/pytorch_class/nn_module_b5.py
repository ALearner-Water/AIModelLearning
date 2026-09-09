import torch
from torch import nn
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()      #super()找到基础模板   init()初始化基础模板
        self.linear=nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)

model = LinearModel()
x=torch.tensor([[1.0],[2.0],[3.0],[4.0]])
pred_y=model(x)
print(model)
print(x.shape)
print(pred_y.shape)
