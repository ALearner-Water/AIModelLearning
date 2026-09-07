# 函数化linear regression  将 w和b的初始化在函数外简化，然后增加是否在gpu里面跑的判断
import torch 
def linear_regression(x,true_y,initial_w,initial_b,learning_rate,epochs,log_print):
    initial_w=torch.tensor(initial_w,dtype=x.dtype,device=x.device,requires_grad=True)  #与x同一类型和位置  cpu还是gpu
    initial_b = torch.tensor(initial_b, dtype=x.dtype, device=x.device, requires_grad=True)
    true_y=true_y.to(dtype=x.dtype,device=x.device) 
    def mse(pred_y,true_y):
        return torch.mean((pred_y-true_y)**2)
    for i in range(1,epochs+1):
        pred_y=x*initial_w+initial_b
        loss=mse(pred_y,true_y)
        loss.backward()
        with torch.no_grad():
            initial_w-=learning_rate*initial_w.grad
            initial_b-=learning_rate*initial_b.grad
            new_pred_y=x*initial_w+initial_b
            new_loss=mse(new_pred_y,true_y)
        initial_w.grad.zero_()
        initial_b.grad.zero_()
        if((i%log_print)==0):
            print(f"w={initial_w}")
            print(f"b={initial_b}")
            print(f"loss={new_loss}")
            print(f"epochs={i}")
    return initial_w,initial_b,new_loss

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")     #调用gpu
x = torch.tensor([0, 1, 2, 3, 4], dtype=torch.float32, device=device)
true_y = torch.tensor([1, 3, 5, 7, 9], dtype=torch.float32, device=device)
initial_w=0.0
initial_b=0.0
w1,b1,loss1=linear_regression(x,true_y,initial_w,initial_b,0.05,200,20)
w2, b2,loss2 = linear_regression(x,true_y, -1.0, 3.0 , 0.05, 200, 20)
