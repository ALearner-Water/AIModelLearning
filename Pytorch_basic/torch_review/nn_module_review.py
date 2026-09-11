import torch
from torch import nn
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(2,1)
    def forward(self,x):
        return self.linear(x)
model=LinearModel()
x=torch.tensor(range(6),dtype=torch.float32).reshape(3,2)
pred_y=model(x)
print(x.shape,pred_y.shape)