# x：[0, 1, 2, 3, 4]
# true_y：[1, 3, 5, 7, 9]
# 数据类型：float32
# 初始 w = 0.0
# 初始 b = 0.0
# w、b 都设置 requires_grad=True
# 学习率：0.05
# 训练轮数：200
# 损失：所有样本平方误差的平均值 MSE
# # 每20轮输出一次 epoch、loss、w、b

import torch
x=torch.tensor([0,1,2,3,4],dtype=torch.float32)
true_y=torch.tensor([1,3,5,7,9],dtype=torch.float32)
w=torch.tensor([0.0],requires_grad=True)
b=torch.tensor([0.0],requires_grad=True)
learning_rate=0.05

def mse (pred_y,true_y) :
    return torch.mean((pred_y-true_y)**2)

for i in range(1,201):
    pred_y=x*w+b
    loss=mse(pred_y,true_y)

    loss.backward()

    # 更新参数
    with torch.no_grad():  # 参数更新这项操作不需要被记录进计算图
        w-=learning_rate*w.grad
        b-=learning_rate*b.grad
        new_pred_y=w*x+b
        new_loss=mse(new_pred_y,true_y) #计算新loss可以放在这里

    # 梯度清零  清哪个就用哪个
    w.grad.zero_()
    b.grad.zero_()

    if((i%20)==0):
        print(f"w={w.item()}")
        print(f"b={b.item()}")
        print(f"new_loss={new_loss}")
        print(f"epoch={i}")
