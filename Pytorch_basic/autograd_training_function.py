# 将训练循环函数化
import torch
def train_linear_regression(x,true_y,initial_w,initial_b,learning_rate,epochs,log_every):
    # 需要有pred_y，然后算mse，然后算梯度，接着更新参数，然后循环，最后输出打印
    for i in range(1,epochs+1):
        pred_y=initial_w*x+initial_b
        loss=torch.mean((pred_y-true_y)**2)
        loss.backward()
        with torch.no_grad():
            # 更新参数不需要纳入计算图
            initial_w -= initial_w.grad * learning_rate
            initial_b -=initial_b.grad * learning_rate
            new_pred_y=initial_w*x+initial_b
            new_loss = torch.mean((new_pred_y - true_y) ** 2)
        # 梯度清零
        initial_w.grad.zero_()
        initial_b.grad.zero_()
        if ((i%log_every)==0): 
            print("epochs=:",i)
            print("loss=",new_loss.item())
            print("w=:",initial_w.item())
            print("b=:",initial_b.item())

    return initial_w,initial_b,new_loss

# 测试
x=torch.tensor([0,1,2,3,4],dtype=torch.float32)
true_y=torch.tensor([1,3,5,7,9],dtype=torch.float32)
learning_rate=0.05
epochs=200
log_every=20
initial_w=torch.tensor([0.0],requires_grad=True)
initial_b=torch.tensor([0.0],requires_grad=True)
w1,b1,loss1=train_linear_regression(x,true_y,initial_w,initial_b,learning_rate,epochs,log_every)
print(f"最终：w1={w1.item()},b1={b1.item()},loss1={loss1.item()}")
initial_b=torch.tensor([3.0],requires_grad=True)
initial_w=torch.tensor([-1.0],requires_grad=True)
w2, b2, loss2 = train_linear_regression( x, true_y,initial_w,initial_b,learning_rate, epochs, log_every)
print(f"最终：w2={w2.item()},b2={b2.item()},loss2={loss2.item()}")
