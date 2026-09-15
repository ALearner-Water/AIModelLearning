import torchvision
import torch
from torch.utils import data
from torchvision import transforms
import time

# ===========================================================数据读取阶段
trans=transforms.ToTensor()     #将图片转成pytorch里面的tensor形式
# train 表示该数据集是训练集  transform 表示将图片转换成tensor download 表示会自动下载该数据集
mnist_train = torchvision.datasets.FashionMNIST(
    root=r"D:\53507\PythonAIModelLearning\Pytorch_basic\softmax\data",  #写绝对路径
    train=True,
    transform=trans,
    download=True,)  # 从dataset里面拿到数据集
mnist_val=torchvision.datasets.FashionMNIST(root=r"D:\53507\PythonAIModelLearning\Pytorch_basic\softmax\data",
    train=False,transform=trans,download=True)

train_loader=data.DataLoader(mnist_train,batch_size=256,shuffle=True,num_workers=0)     #num_worker是多进程加载图片
val_loader = data.DataLoader(mnist_val, batch_size=256, shuffle=True, num_workers=0)
# 做计时器看读取照片需要多久
start=time.time()
for x,y in train_loader:
    continue
end=time.time()
print(end-start)    
#===============================================================处理阶段 从零实现
num_inputs=764  #因为图片是28*28的灰度图片，这里需要把他拉长成一维向量
num_outputs=10  #一共有十个类别，最后让神经网络输出10个分数
w=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True) #从均值为0 标准差为0.01的正态分布随机采样
b=torch.zeros(10,requires_grad=True)
learning_rate=0.01
def softmax(x):
    x_exp=torch.exp(x)  #分子 
    partition=x_exp.sum(1,keepdim=True) #分母
    return x_exp/partition  #这个就是softmax求概率