#不使用nn.Rnn自己先写一个rnncell
import torch
from torch import nn

class MyRnn_cell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.linear=nn.Linear(input_size+hidden_size,hidden_size)
    def forward(self,input,h):
        combine=torch.cat(input,h,dim=1)
        output=torch.tanh(self.linear(combine))
        return output
model=MyRnn_cell(input_size=5,hidden_size=3)
print(model)
    
