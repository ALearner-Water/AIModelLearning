import torch
from torch import nn
X = torch.tensor(
    [   [2.0, 0.0],
        [3.0, 0.0],
        [0.0, 2.0],
        [0.0, 3.0],
        [-2.0, -2.0],
        [-3.0, -3.0],
    ],dtype=torch.float32,)

# 设标签
labels=torch.tensor([0,0,1,1,2,2],dtype=torch.long)

# 设参数
w=torch.zeros(2,3,requires_grad=True,dtype=torch.float32)
b=torch.zeros(3,requires_grad=True,dtype=torch.float32)
print(X.shape,labels.shape,labels.dtype,w.shape,b.shape)

for i in range(500):
    # 计算得分
    logits=X @ w + b

    # 过softmax求概率
    probs=torch.softmax(logits,dim=1)
    # print(probs)

    #通过标签取出对应概率
    true_class_probs=probs[range(len(probs)),labels]
    # print(true_class_probs,true_class_probs.shape)

    #算crossentropy
    loss=-torch.log(true_class_probs).mean()
    # print(loss,loss.shape)

    #算梯度
    loss.backward()
    # print(w.grad,b.grad)

    #更新参数
    learning_rate=0.5
    with torch.no_grad():
        w-=w.grad*learning_rate
        b-=b.grad*learning_rate
    w.grad.zero_()
    b.grad.zero_()

logits=X@w+b
probs=torch.softmax(logits,dim=1)
true_class_probs=probs[range(len(probs)),labels]
loss=-torch.log(true_class_probs).mean()
loss_fn=nn.CrossEntropyLoss()
framework_loss = loss_fn(logits, labels)
print(framework_loss.item())
print(loss.item())
print(torch.allclose(loss,framework_loss))

#计算最终的类别和accuracy
pred_labels=torch.argmax(logits,dim=1)
print(pred_labels,pred_labels.shape)
accuracy=(pred_labels==labels).float().mean()
print(accuracy)
