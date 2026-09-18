import torch
from torch import nn

class MODEL(nn.Module):     #直接使用sequntial将模型打包成流水线
    def __init__(self):
        super().__init__()
        self.model0=nn.Sequential(
        nn.Conv2d(1,3,kernel_size=3,stride=1,padding=0),
        nn.MaxPool2d(2),
        nn.ReLU(),           #非线性变化，非负为原输入负数为0 隐藏层使用
        nn.Flatten(start_dim=1),
        nn.Linear(13*13*3,10),         #线性拟合     #fulling connect
        nn.Softmax(dim=1)       #非线性变化，多分类输出0-1 求和为1  输出层使用 若使用了交叉熵这里就不要重复使用
        # nn.Sigmoid(),       #非线性变化，二分类输出0或1         输出或隐藏层
        )
    def forward(self,x):
        return self.model0(x)
model=MODEL()
print(model)
test_x = torch.randn(2, 1, 28, 28)  # batch=2，单通道，28×28图片
pred = model(test_x)
print(pred.shape)
print(pred)
print(pred.sum(dim=1))  # 每一行10个概率相加，应该≈1
