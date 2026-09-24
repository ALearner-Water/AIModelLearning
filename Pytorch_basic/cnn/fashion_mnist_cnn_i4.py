import torch
import torchvision
from torch import nn
from torch.utils.data import random_split,DataLoader
class MYMODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3,padding=0)
        self.conv2d_2=nn.Conv2d(in_channels=6,out_channels=15,kernel_size=3,padding=0)
        self.pooling=nn.MaxPool2d(2)
        self.relu=nn.ReLU()
        self.linear=nn.Linear(375,10)
    def forward(self,input):
        hidden_1=self.relu(self.conv2d_1(input))
        hidden_2=self.pooling(hidden_1)
        hidden_3=self.relu(self.conv2d_2(hidden_2))
        hidden_4=self.pooling(hidden_3)
        hidden_5=torch.flatten(hidden_4,start_dim=1)
        output=self.linear(hidden_5)
        return output
def train(loader,model,fn_loss,optimizer,epochs):
    model.train()
    for i in range(epochs):
        for img,labels in loader:
            img=img.to(device)
            labels=labels.to(device)
            output=model(img)
            optimizer.zero_grad()
            loss=fn_loss(output,labels)
            loss.backward()
            optimizer.step()

def evaluate(loader,model,fn_loss):
    checkpoint=None #保留第一个预测错误的地方
    model.eval()
    with torch.no_grad():
        data_size=0
        full_loss=0.0
        accuracy=0.0
        for img,labels in loader:
            img=img.to(device)
            labels=labels.to(device)
            logits=model(img)
            loss=fn_loss(logits,labels)
            batch_size=len(img)
            full_loss+=loss.item()*batch_size
            data_size+=batch_size
            pred=torch.argmax(logits,dim=1)     #这里只是预测的类别下标
            batch_accuracy=(pred==labels).float().mean()    #这里就看有没有相等，转成布尔值
            accuracy+=batch_accuracy.item()*batch_size

            #求预测错误的地方 
            if (checkpoint==None):
                for pred_label,true_label,img in zip(pred,labels,img):
                    if(pred_label!=true_label):
                        checkpoint=(img.cpu(),pred_label.item(),true_label.item()) 
                        break

        return full_loss/data_size,accuracy/data_size,checkpoint

dataset=torchvision.datasets.FashionMNIST(root=r"D:\53507\PythonAIModelLearning\Pytorch_basic\cnn",train=True,transform=torchvision.transforms.ToTensor(),download=True)
train_set, val_set = random_split(dataset, lengths=[50000, 10000], generator=torch.Generator().manual_seed(42))
train_loader=DataLoader(train_set,batch_size=64,shuffle=True)
val_loader=DataLoader(val_set,batch_size=64,shuffle=False)
model=MYMODEL()
fn_loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
fn_loss.to(device)
epochs=1
train(train_loader,model,fn_loss,optimizer,epochs)
full_loss,accuracy,chackpoint=evaluate(val_loader,model,fn_loss)
print(full_loss,accuracy)
print(chackpoint)

#看看模型训练的效果
img, label = train_set[0]  # 取第0号样本（第一张图+它的标签）
img = img.to(device)  # 加上这一行！！  但是取单张样本label是标量
img = img.unsqueeze(0)  # 在最前面增加batch维度，shape从[1,28,28] → [1,1,28,28]
print(img.shape)  # 图片tensor，shape [1,28,28]
print(label)
logit=model(img)
pred_y=torch.argmax(logit).item()
print(pred_y==label)
