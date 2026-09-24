import torch
from torch import nn
batch_size=2
input_size=3
hidden_size=4
seq_len=3
x=torch.randn(seq_len,batch_size,input_size)
h=torch.zeros(1,batch_size,hidden_size) #第一个是num_layer 是hidden的层数
model=nn.RNN(input_size,hidden_size,num_layers=1)
out,h_n=model(x,h)
print(out.shape,h.shape)
