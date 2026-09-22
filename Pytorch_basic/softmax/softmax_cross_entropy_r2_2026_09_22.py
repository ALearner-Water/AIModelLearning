import torch
from torch import nn
logits=torch.tensor([[0.0,0.0],
                    [0.0,1.0986]])
# 先使用softmax将得分换算成概率
softmax=torch.softmax(logits,dim=1)
print(softmax,softmax.shape,softmax.sum(dim=1))

# 给出真实标签
labels = torch.tensor([0, 1])

# 使用真实标签提取出对应的概率
p_true=softmax[range(len(softmax)), labels]
print(p_true)

# 算出每个概率的loss
loss_each=-torch.log(p_true)
print(loss_each)
print(p_true.shape,loss_each.shape)

# 对loss求平均，算总loss
full_loss=torch.mean(loss_each)
print(full_loss.item(),full_loss.shape)

# 使用nn框架来求loss
criterion=nn.CrossEntropyLoss()
loss=criterion(logits,labels)
print(loss.item())

print(torch.allclose(full_loss, loss))
