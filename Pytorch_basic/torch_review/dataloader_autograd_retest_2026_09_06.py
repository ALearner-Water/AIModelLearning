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
        initial_y=self.true_y[index]
        return initial_x,initial_y
x=torch.tensor([[-3],[-2],[-1],[0],[1],[2],[3]],dtype=torch.float32)
true_y=-2*x+0.5
w=torch.tensor([0.0],requires_grad=True)
b =torch.tensor([0.0], requires_grad=True)
epochs=400
learning_rate=0.02
dataset=LinearDataset(x,true_y)
loader=DataLoader(dataset,batch_size=3,shuffle=True)
for epoch in range(epochs):
    for batch_x,batch_y in loader:
        pred_y=w*batch_x+b
        loss=torch.mean((pred_y-batch_y)**2)
        loss.backward()
        with torch.no_grad():
            w-=w.grad*learning_rate
            b-=b.grad*learning_rate
        w.grad.zero_()
        b.grad.zero_()
full_loss=torch.mean(((w*x+b)-true_y)**2)
print(f"w={w.item()},b={b.item()},full_loss={full_loss.item()}")
