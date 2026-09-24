import torch
from torch import nn
train_X = torch.tensor(
    [
        [[1.0], [0.8], [1.2], [0.0], [0.1], [0.0]],
        [[0.7], [1.1], [0.9], [0.1], [0.0], [-0.1]],
        [[1.3], [0.9], [0.8], [-0.1], [0.0], [0.1]],
        [[0.9], [1.2], [1.1], [0.0], [-0.1], [0.0]],
        [[-1.0], [-0.8], [-1.2], [0.0], [0.1], [0.0]],
        [[-0.7], [-1.1], [-0.9], [0.1], [0.0], [-0.1]],
        [[-1.3], [-0.9], [-0.8], [-0.1], [0.0], [0.1]],
        [[-0.9], [-1.2], [-1.1], [0.0], [-0.1], [0.0]],
    ],
    dtype=torch.float32,
)

train_labels = torch.tensor(
    [0, 0, 0, 0, 1, 1, 1, 1],
    dtype=torch.long,
)

val_X = torch.tensor(
    [
        [[1.1], [0.7], [1.0], [0.0], [-0.1], [0.0]],
        [[0.8], [1.3], [0.9], [0.1], [0.0], [0.0]],
        [[-1.1], [-0.7], [-1.0], [0.0], [-0.1], [0.0]],
        [[-0.8], [-1.3], [-0.9], [0.1], [0.0], [0.0]],
    ],
    dtype=torch.float32,
)

val_labels = torch.tensor(
    [0, 0, 1, 1],
    dtype=torch.long,
)

# 使用rnn做二分类
class MYMODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.RNN(input_size=1,hidden_size=8,batch_first=True)
        self.linear=nn.Linear(8,2)
    def forward(self, input):
        out, hn = self.rnn(input)  # 拆包 只拿out去做下一层的rnn即可
        output = self.linear(out[:, -1, :]) #取最后时间步进linear
        return output  # logits


def train(train_x,train_labels,model, fn_loss, optimizer, epochs):
    model.train()
    for i in range(epochs):
        output = model(train_x)
        optimizer.zero_grad()
        loss = fn_loss(output,train_labels)
        loss.backward()
        optimizer.step()


def evaluate(val_x,val_labels, model, fn_loss):
    model.eval()
    with torch.no_grad():
        data_size = 0
        full_loss = 0.0
        accuracy = 0.0
        logits = model(val_x)
        loss = fn_loss(logits, val_labels)
        batch_size = len(val_x)
        full_loss += loss.item() * batch_size
        data_size += batch_size
        pred = torch.argmax(logits, dim=1)  # 这里只是预测的类别下标
        batch_accuracy = ((pred == val_labels).float().mean())  # 这里就看有没有相等，转成布尔值
        accuracy += batch_accuracy.item() * batch_size
        return full_loss / data_size, accuracy / data_size


torch.manual_seed(0)
model=MYMODEL()
fn_loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
epochs=300
train(train_X,train_labels,model,fn_loss,optimizer,epochs)
loss,accuracy=evaluate(val_X,val_labels,model,fn_loss)

print(loss,accuracy)
