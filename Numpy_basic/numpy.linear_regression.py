# 实现线性回归
import numpy as np
x = np.array([1.0, 2.0, 3.0, 4.0])
true_y = np.array([3.0, 5.0, 7.0, 9.0])
# 要学习w和b

def mse(true_y,pred_y):
    return np.mean((pred_y-true_y)**2)

w=0
b=0
learning_rate=0.1
epslion=0.001   #算梯度

def loss(w,b):
    pred_y=w*x+b
    return mse(true_y,pred_y)

for i in range(2000):
# w的梯度
    right_of_w=loss(w+epslion,b)
    left_of_w=loss(w-epslion,b)
    w_gradient=(right_of_w-left_of_w)/(2*epslion)

    # b的梯度
    right_of_b = loss(w, b + epslion)
    left_of_b = loss(w, b - epslion)
    b_gradient = (right_of_b - left_of_b) / (2 * epslion)

    # 有两个参数需要更新，需要两个梯度
    w=w-learning_rate*w_gradient
    b=b-learning_rate*b_gradient
    print(f"b_gradient={b_gradient}")
    print(f"w_gradient={w_gradient}")
    print(f"w={w}")
    print(f"b={b}")

print(loss(w,b))
