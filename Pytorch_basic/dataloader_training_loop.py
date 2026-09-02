import torch
from torch.utils.data import Dataset,DataLoader

class LinearDataset(Dataset):
    def __init__(self,x,true_y):
        self.x=x
        self.true_y=true_y
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x=self.x[index]
        y=self.true_y[index]
        return x,y
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
true_y = 2 * x + 1
dataset = LinearDataset(x, true_y)
loader = DataLoader(dataset, batch_size=2, shuffle=False)
# 开始结合loss训练更新参数
epochs=500
learning_rate=0.01
w=torch.tensor(0.0,requires_grad=True)
b=torch.tensor(0.0,requires_grad=True)
for epoch in range(epochs):
    for batch_x, batch_true_y in loader:
        pred_y=batch_x*w+b
        loss=torch.mean((pred_y-batch_true_y)**2)
        loss.backward()

        with torch.no_grad():
            w-=learning_rate*w.grad
            b-=learning_rate*b.grad
        
        w.grad.zero_()
        b.grad.zero_()
#算最终的loss
with torch.no_grad():
    new_pred_y=w*x+b
    new_loss=torch.mean((new_pred_y-true_y)**2)
    print(f"loss={new_loss}") 
    print(f"w={w.item()}")
    print(f"b={b.item()}")   
