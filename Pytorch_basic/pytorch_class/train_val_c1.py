from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn

# 计算指定数据集的loss   从单个val_x,val_y改写成val_dataloader  
def validate(model,dataloader,loss_fn):
    total_loss=0.0
    total_samples=0
    with torch.no_grad():           #这里也不需要计入计算图
        for batch_x,batch_y in dataloader:
            pred_batch_y=model(batch_x)
            batch_loss=loss_fn(pred_batch_y,batch_y)
            #每次求出传入样本的数量，最后要加权求平均loss
            total_loss+=batch_loss*len(batch_x)
            total_samples+=len(batch_x)
    return total_loss/total_samples
    
# 将训练循环封装成函数
def train(model,train_loader,optimizer,loss_fn,epoch):
    for i in range(epoch):
        for batch_x,batch_y in train_loader:
            batch_pred_y=model(batch_x)
            batch_loss=loss_fn(batch_pred_y,batch_y)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

class mydataset(Dataset):
    def __init__(self, x, true_y):
        self.x = x
        self.true_y = true_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        initial_x = self.x[index]
        inicial_y = self.true_y[index]
        return initial_x, inicial_y


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
true_y = 3 * x - 2
# 分割训练集和验证集
train_x=x[:5]
train_y=true_y[:5]
val_x=x[5:]
val_y=true_y[5:]
# ==================
print(f"train_x={train_x},shape={train_x.shape}")
print(f"val_x={val_x},shape={val_x.shape}")


# 将dataset和dataloader也拆成俩套
train_dataset = mydataset(train_x,train_y)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
val_dataset=mydataset(val_x,val_y)   #在大数据集的时候会把验证集分批
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False)
# ==================================
print(f"len_train_x={len(train_dataset)}")

model = LinearModel()
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epoch=500
# 直接调用函数训练
train(model,train_loader,optimizer,loss,epoch)

train_loss = validate(model,train_loader, loss)
val_loss = validate(model,val_loader, loss)

print(f"train_loss={train_loss}, val_loss={val_loss}")
