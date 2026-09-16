import torch
from torch import nn
import torch.nn.functional as F    
# 进行卷积操作   2d卷积需要四维向量  分别是 batch 通道数 长 狂
input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]]).reshape(1,1,5,5)
# 卷积核 就是w  同上
kernul=torch.tensor([[1,2,1,],
                     [0,1,0],
                     [2,1,0]]).reshape(1,1,3,3)
# 卷积核中输入上面开始移动计算
output1=F.conv2d(input,kernul,stride=1,padding=0)
print(output1.reshape(3,3))
output2= F.conv2d(input, kernul, stride=2,padding=0)    #padding就是填充
print(output2.reshape(2,2))
