import torch
from torch import nn
seed=0
epochs = 500
learning_rate = 0.1
torch.manual_seed(seed)    #保证每次随机生成的权重都是一样的
model=nn.Linear(2,3)
fn_loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

def train(train_X, train_labels, optimizer, fn_loss, epochs, model):
    #开启训练模式  可以dropout和batchnorm
    model.train()
    for i in range(epochs):
        logit = model(train_X)
        loss = fn_loss(logit, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(val_X,val_labels,fn_loss,model):
    #开启验证函数 不可以dropout 但是不会关闭计算图
    model.eval()
    with torch.no_grad():
        logits=model(val_X)
        pred_y=torch.argmax(logits,dim=1)
        accuracy=(pred_y==val_labels).float().mean()
        loss=fn_loss(logits,val_labels)
        return loss,accuracy

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

train_labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

val_X = torch.tensor(
    [
        [2.5, 0.2],
        [0.2, 2.5],
        [-2.5, -2.5],
    ],
    dtype=torch.float32,
)

val_labels = torch.tensor([0, 1, 2], dtype=torch.long)

train(train_X,train_labels,optimizer,fn_loss,epochs,model=model)
val_loss,val_accuracy=evaluate(val_X,val_labels,fn_loss,model)
train_loss,train_accuracy=evaluate(train_X,train_labels,fn_loss,model)
print(val_loss.item(),val_accuracy.item())
print(train_loss.item(),train_accuracy.item())
print("=" * 40)
print("【实验配置】")
print(f"seed = {seed}")
print(f"模型结构: {model}")
print(f"learning rate = {learning_rate}")
print(f"epochs = {epochs}")
print(f"训练样本数: {len(train_X)}")
print(f"验证样本数: {len(val_X)}")
print("=" * 40)
