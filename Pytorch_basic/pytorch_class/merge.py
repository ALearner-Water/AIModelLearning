from torch.utils.data import Dataset,DataLoader
import torch
from torch import nn

class mydataset(Dataset):
    def __init__(self,x,true_y):
        self.x=x
        self.true_y=true_y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        initial_x=self.x[index]
        inicial_y=self.true_y[index]
        return initial_x,inicial_y

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)

x=torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0],[6.0],[7.0]])
true_y=3*x-2
dataset=mydataset(x,true_y)
loader=DataLoader(dataset,batch_size=4,shuffle=False)
model=LinearModel()
loss=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
for i in range(500):
    for batch_x,batch_y in loader:
        batch_pred_y=model(batch_x)
        batch_loss=loss(batch_pred_y,batch_y)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

full_loss=loss(model(x),true_y)
print(f"full_loss={full_loss},w.shape={model.linear.weight.shape},b.shape={model.linear.bias.shape}")