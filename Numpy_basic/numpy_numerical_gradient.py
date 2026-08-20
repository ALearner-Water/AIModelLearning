# 计算梯度
import numpy as np
# 定义误差函数
def mse(true_y,pred_y):
    error=pred_y-true_y
    return np.mean(error**2)
true_y=np.array([4.0])
x=np.array([2.0])

def loss_of_w(w):
    pred_y=x*w
    return mse(true_y,pred_y)
w=0
epsilon=0.001
learning_rate=0.1
gradient=0
for i in range(5):
    loss_right=loss_of_w(w+epsilon)
    loss_left=loss_of_w(w-epsilon)
    # 梯度=高度/水平位置
    gradient=(loss_right-loss_left)/(2*epsilon)
    w=w-learning_rate*gradient
    print(F"gradient={gradient}")
    print(f"w={w}")
