import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader

# 搭建卷积层模型  卷积层就是来提取特征的
class conv2dmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=Conv2d(1,6,kernel_size=3,stride=1,padding=0) #输入3维，因为有6个卷积核所以输出6维，然后卷积核大小是3*3 
        self.conv2=Conv2d(6,15,kernel_size=3,stride=1,padding=0)    #两层卷积层
        self.pool=MaxPool2d(2)  #增加池化层减少计算量   是2*2的
        self.linear=nn.Linear(15*5*5,10)   #最后加上全连接层可以求loss训练
    def forward(self,input):
        batch_size=input.size(0)    #获取样本数后面拉直图片成一维向量进入全连接层
        output=self.conv1(input)
        output=self.pool(output)
        output=self.conv2(output)
        output=self.pool(output)
        output=torch.flatten(output,start_dim=1) #需要将向量变成一维特征向量给全连接层
        output=self.linear(output)
        return output

dataset = torchvision.datasets.FashionMNIST(r"D:\53507\PythonAIModelLearning\Pytorch_basic\cnn",
                                       train=False,transform=torchvision.transforms.ToTensor(),download=True)
loader=DataLoader(dataset,batch_size=64)
model=conv2dmodel()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)    #将模型搬到gpu上面
print(model)
for img,target in loader:   #img是图片张量 target是标签张量
    img=img.to(device)
    output=model(img)
    print(img.shape)
    print(output.shape)