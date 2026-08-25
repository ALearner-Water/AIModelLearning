# 使用numpy和torch分别做梯度下降
import numpy as np
import torch
x=np.array([1,2,3])
true_y=np.array([3,5,7])
w=np.array([0.5])
b=np.array([1.0])
epsilon=0.001

def mse(pred_y,true_y):
    if (type(pred_y) == np.ndarray):
        return np.mean((pred_y-true_y)**2)
    else:
        return torch.mean((pred_y-true_y)**2)

def loss(w,b):
    pred_y=w*x+b
    return mse(pred_y,true_y)

# 先通过中心差分计算w的梯度
loss_of_right=loss(w+epsilon,b)
loss_of_left=loss(w-epsilon,b)
gradient_of_w=(loss_of_right-loss_of_left)/(2*epsilon)
print(gradient_of_w)

# 再使用自动求导算w的梯度
x=torch.from_numpy(x)
true_y=torch.from_numpy(true_y)
w=torch.tensor(w,requires_grad=True)
b=torch.tensor(b,requires_grad=True)
loss=loss(w,b)
loss.backward()
print(w.grad.item())
print(abs(gradient_of_w - w.grad.item()))
print(epsilon)
