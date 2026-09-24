# vgg是一个搭建好的网络架构 可以在模型中进行改动来搭建自己的模型 是一个分类模型
import torch
import torchvision
from torch import nn
from torchvision.models import VGG16_Weights
vgg16_true=torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)    #可以下载已经训练好权重参数
# vgg16_false=torchvision.models.vgg16()  #不下载训练好的参数

# 加载自己的数据集
train_data = torchvision.datasets.CIFAR10(
    root=r"D:\53507\PythonAIModelLearning\Pytorch_basic\cnn",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

#在vgg16最后加入自己的线性层
print(vgg16_true)
vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))
print(vgg16_true)
