# 可以把写好的模型保存起来然后通过load调用，对于自己写的model有特殊的调用方法
import torch
import torchvision

vgg16=torchvision.models.vgg16()
print(vgg16)

# 保存方式1
torch.save(vgg16, "Pytorch_basic/cnn/vgg16_method1.pth")


# 保存方式2 将参数状态也保存进去
torch.save(vgg16.state_dict(), "Pytorch_basic/cnn/vgg16_method2.pth")
