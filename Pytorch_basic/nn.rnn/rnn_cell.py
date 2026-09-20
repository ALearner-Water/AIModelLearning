# 不使用nn.Rnn自己先写一个rnncell
import torch
from torch import nn

class MyRnn_cell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.linear=nn.Linear(input_size+hidden_size,hidden_size)
    def forward(self,input,h):
        combine=torch.cat((input,h),dim=1)
        output=torch.tanh(self.linear(combine))
        return output
input_size = 3
hidden_size = 5
batch_size = 2
model=MyRnn_cell(input_size,hidden_size)
x=torch.randn(batch_size,input_size)
h=torch.randn(batch_size,hidden_size)
output=model(x,h)
print(x.shape,h.shape,output.shape)
seq_len=4
h1=torch.zeros(batch_size,hidden_size)
x1=torch.randn(seq_len,batch_size,input_size)   #四个时间步  两个样本   三维向量1
print(x1.shape)
for i in range(seq_len):
    batch_output=model(x1[i],h1)
    h1=batch_output     #将上一层的输出加到下一层去
    print(batch_output.shape)
