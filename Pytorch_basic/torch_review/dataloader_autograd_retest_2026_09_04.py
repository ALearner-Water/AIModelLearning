import torch
from torch.utils.data import Dataset,DataLoader
class LinearDataset(Dataset):
    def __init__(self,x,true_y):
        self.x=x
        self.true_y=true_y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        initial_x=self.x[index]
        initial_true_y=self.true_y[index]
        return initial_x,initial_true_y
x=torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0]])
true_y=3*x-2
dataset=LinearDataset(x,true_y)
loader=DataLoader(dataset,batch_size=2,shuffle=False)
learning_rate=0.01
epochs=500
w=torch.tensor([0.0],requires_grad=True)
b=torch.tensor([0.0],requires_grad=True)
for epoch in range(epochs):
    for batch_x,batch_true_y in loader:
        pred_y=w*batch_x+b
        loss=torch.mean((pred_y-batch_true_y)**2)
        loss.backward()
        with torch.no_grad():
            w-=learning_rate*w.grad
            b-=learning_rate*b.grad
        w.grad.zero_()
        b.grad.zero_()
print(f"w={w.item()}")
print(f"b={b.item()}")
full_loss=torch.mean(((w*x+b)-true_y)**2)
print(f"full_loss={full_loss}")