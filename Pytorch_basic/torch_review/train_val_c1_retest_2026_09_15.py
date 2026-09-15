import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn

class MyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linaer=nn.Linear(1,1)
    def forward(self,x):
        return self.linaer(x)
class MyDataset(Dataset):
    def __init__(self,x,true_y):
        self.x=x
        self.true_y=true_y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        initial_x=self.x[index]
        initial_y=self.true_y[index]
        return initial_x,initial_y

def train(model,fn_loss,optimizer,dataloader,epochs):
    for i in range(epochs):
        for batch_x,batch_y in dataloader:
            pred_y=model(batch_x)
            batch_loss=fn_loss(pred_y,batch_y)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

def validate(model,fn_loss,dataloader):
    with torch.no_grad():
        full_size=0
        full_loss=0.0
        for batch_x,batch_y in dataloader:
            pred_y=model(batch_x)
            batch_loss=fn_loss(pred_y,batch_y)
            full_loss+=batch_loss*len(batch_x)
            full_size+=len(batch_x)
        return full_loss/full_size

model=MyLinear()
fn_loss=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
epochs=500
x=torch.tensor(range(-3,7),dtype=torch.float32).reshape(10,1)
y=-1.5*x+2
train_x=x[:7]
train_y=y[:7]
val_x=x[7:]
val_y=y[7:]
train_dataset=MyDataset(train_x,train_y)
val_dataset=MyDataset(val_x,val_y)
train_dataloader=DataLoader(train_dataset,batch_size=4,shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
train(model,fn_loss,optimizer,train_dataloader,epochs)
val_loss=validate(model,fn_loss,val_dataloader)
train_loss=validate(model,fn_loss,train_dataloader)
print(
    f"val_loss={val_loss},train_loss={train_loss},weigth={model.linaer.weight.item()},bias={model.linaer.bias.item()}")
