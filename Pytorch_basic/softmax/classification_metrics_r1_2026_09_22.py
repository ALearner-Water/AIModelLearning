import torch

x = torch.tensor([[1.2, 3.4, 0.5], 
                   [2.2, 0.1, 1.8], 
                   [0.3, 0.7, 2.5], 
                   [4.0, 4.5, 4.1]])
predict=x.argmax(dim=1)
true = torch.tensor([1, 0, 1, 2])
print(x)
print(predict,predict.shape)  #[1，0，2，1]  (4,)
print(true,true.shape,true.dtype)
print((predict==true).float().mean())   #50%
