import torch
import torchvision
# 将保存好的模型加载进来

# 加载方式1
model1 = torch.load("Pytorch_basic/cnn/vgg16_method1.pth",weights_only=False)

# 加载方式2  用state_dict的方式存模型的时候是保存了模型的参数，所以直接打印model2是打印的模型参数
model2 = torch.load("Pytorch_basic/cnn/vgg16_method2.pth",weights_only=False)
print(model2)

# 那就需要使用新建模型然后再恢复成网络模型的方法来打印

# 先创建新模型
vgg16=torchvision.models.vgg16()

# 再读入字典参数
state_dict = torch.load("Pytorch_basic/cnn/vgg16_method2.pth", weights_only=False)

#最后传入参数
vgg16.load_state_dict(state_dict)