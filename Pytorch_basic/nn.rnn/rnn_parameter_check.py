import torch
from torch import nn

input_size=3
hidden_size=5
num_layer=2
batch_size=7
seq_len=10
x=torch.randn(7,10,3)
h0=torch.randn(4,7,5)
model=nn.RNN(input_size=3,hidden_size=5,num_layers=2,bidirectional=True,batch_first=True)    #bidirectional=True正反都来一次
out,hn=model(x,h0)
print(x.shape,h0.shape,out.shape,hn.shape)
#x.shape=7,10,3 h0.shape=4,7,5 out.shape=7,10,10  hn.shape=4,7,5