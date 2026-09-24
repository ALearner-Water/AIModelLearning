import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)

class dataset(Dataset):
    def __init__(self,x,true_y):
        self.x=x
        self.true_y=true_y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        initial_x=self.x[index]
        initial_y=self.true_y[index]
        return initial_x,initial_y

model=LinearModel()
fn_loss=nn.MSELoss()
optimmizer=torch.optim.SGD(model.parameters(),lr=0.03)
epochs=500

x=torch.tensor(range(-4,5),dtype=torch.float32).reshape(9,1)
true_y=3*x-4
my_dataset=dataset(x,true_y)
loader=DataLoader(my_dataset,batch_size=4,shuffle=True)
for i in range(epochs):
    for batch_x,batch_y in loader:
        batch_pred_y=model(batch_x)
        batch_loss=fn_loss(batch_pred_y,batch_y)
        optimmizer.zero_grad()
        batch_loss.backward()
        optimmizer.step()
full_pred_y=model(x)
full_loss=fn_loss(full_pred_y,true_y)
print(f"full_loss={full_loss},weight={model.linear.weight.item()},nias={model.linear.bias.item()}")
