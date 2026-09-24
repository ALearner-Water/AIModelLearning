# 验证真正掌握了没有
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

# 建模型
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)

# 定义数据集
class mydataset(Dataset):
    def __init__(self,x,true_y):
        self.x=x
        self.true_y=true_y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        initial_x=self.x[index]
        initial_y=self.true_y[index]
        return initial_x,initial_y

# 训练函数
def train(model,loader,optimizer,fn_loss,epochs):
    for i in range(epochs):
        for batch_x,batch_y in loader:
            batch_pred_y=model(batch_x)
            batch_loss=fn_loss(batch_pred_y,batch_y)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

# 验证函数
def validate(model,loader,fn_loss):
    total_loss=0.0
    total_examples=0
    with torch.no_grad():
        for batch_x,batch__y in loader:
            pred_y=model(batch_x)
            batch_loss=fn_loss(pred_y,batch__y)
            batch_size=len(batch_x)
            total_loss+=batch_loss*batch_size
            total_examples+=batch_size
    return total_loss/total_examples


# 创建对象并初始化
model=LinearModel()
optimizer=torch.optim.SGD(model.parameters(),lr=0.05)
loss=nn.MSELoss()
epochs=500
x=torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0],[6.0],[7.0]])
true_y=-2*x+5
train_x=x[:4]
train_y=true_y[:4]
val_x=x[4:]
val_y=true_y[4:]
train_dataset=mydataset(train_x,train_y)
val_dataset = mydataset(val_x, val_y)
train_loader=DataLoader(train_dataset,batch_size=3,shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)
#设置好后开始训练
train(model,train_loader,optimizer,loss,epochs)
#跑验证集
val_loss=validate(model,val_loader,loss)
train_loss=validate(model,train_loader,loss)
print(f"val_loss={val_loss},train_loss={train_loss}")

