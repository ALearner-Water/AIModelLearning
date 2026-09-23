import torch
from torch import nn

class MYMODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(2,4)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(4,3)
    def forward(self,input):
        hidden1=self.linear1(input)
        hidden2=self.relu(hidden1)
        output=self.linear2(hidden2)
        return output

seed = 0
torch.manual_seed(seed)
model1=MYMODEL()
model2=nn.Linear(2,3)
fn_loss=nn.CrossEntropyLoss()
epochs=500
lr=0.01
optimizer1=torch.optim.SGD(model1.parameters(),lr=lr)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=lr)

def train(train_x,train_labels,optimizer,model,epochs,fn_loss):
    model.train()
    for i in range(epochs):
        logits=model(train_x)
        optimizer.zero_grad()
        loss=fn_loss(logits,train_labels)
        loss.backward()
        optimizer.step()

def evaluate(val_x,val_labels,model,fn_loss):
    model.eval()
    with torch.no_grad():
        logits=model(val_x)
        loss=fn_loss(logits,val_labels)
        accuracy=(torch.argmax(logits,dim=1)==val_labels).float().mean()
        return accuracy,loss
train_X = torch.tensor(
    [
        [2.0, 0.0],
        [3.0, 0.0],
        [0.0, 2.0],
        [0.0, 3.0],
        [-2.0, -2.0],
        [-3.0, -3.0],
    ],
    dtype=torch.float32,
)

train_labels = torch.tensor(
    [0, 0, 1, 1, 2, 2],
    dtype=torch.long,
)

val_X = torch.tensor(
    [
        [2.5, 0.2],
        [0.2, 2.5],
        [-2.5, -2.5],
    ],
    dtype=torch.float32,
)

val_labels = torch.tensor(
    [0, 1, 2],
    dtype=torch.long,
)

train(train_X,train_labels,optimizer1,model1,epochs,fn_loss)
train(train_X, train_labels, optimizer2, model2, epochs, fn_loss)
accuracy_1, loss1 = evaluate(val_X, val_labels, model1, fn_loss)
accuracy_1_train,loss1_train=evaluate(train_X,train_labels,model1,fn_loss)
accuracy_2, loss2 = evaluate(val_X, val_labels, model2, fn_loss)
accuracy_2_train, loss2_train = evaluate(train_X, train_labels, model2, fn_loss)
print(f"accuracy_1={accuracy_1},loss1={loss1}")
print(f"accuracy_1_train={accuracy_1_train},loss1_train={loss1_train}")
print(f"accuracy_2={accuracy_2},loss2={loss2}")
print(f"accuracy_2_train={accuracy_2_train},loss2_train={loss2_train}")
