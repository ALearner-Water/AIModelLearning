import torch
from torch.utils.data import Dataset    #把数据统一变成可以按编号取出的形式
from torch.utils.data import DataLoader #把数据变成一批批的样本

class LinearDataset(Dataset):
    def __init__(self,x,true_y):    #数据初始化
        self.x=x
        self.true_y=true_y

    def __len__(self):              #求数据长度
        return len(self.x)
    def __getitem__(self, index):
        x=self.x[index]
        y=self.true_y[index]
        return x,y
x=torch.tensor([[1.0],[2.0],[3.0],[4.0]])
true_y=2*x+1
dataset=LinearDataset(x,true_y)
loader=DataLoader(dataset,batch_size=2  ,shuffle=False)   #从dataset里面拿样本，每个批次放两个样本，样本不打乱  true则是打乱样本重新组批
# 打印验证是不是变成批次了
for batch_x,batch_true_y in loader:
    print(batch_x)
    print(batch_true_y)
    print("batch_x shape:", batch_x.shape)
    print("batch_true_y shape:", batch_true_y.shape)
# print("=====================================")
# print(dataset.x)
# print(dataset.true_y)
# print(len(dataset))
# print(dataset[2])
