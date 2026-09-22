import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn

class MYDATASET(Dataset):
    def __init__(self,x,y):
        super().__init__()
        self.x=x
        self.y=y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        initial_x=self.x[index]
        initial_y=self.y[index]
        return initial_x,initial_y

x=torch.randn(10,1,dtype=torch.float32).reshape(10,1)
y=-2*x+5
batch_size=3
model=nn.Linear(1,1)
dataset=MYDATASET(x,y)
loader=DataLoader(dataset,batch_size,shuffle=True)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
epochs=500
fn_loss=nn.MSELoss()

for i in range(epochs):
    for batch_x,batch_y in loader:
        pred_y=model(batch_x)
        loss=fn_loss(pred_y,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

pred_y=model(x)
loss=fn_loss(pred_y,y)
print(loss.item())