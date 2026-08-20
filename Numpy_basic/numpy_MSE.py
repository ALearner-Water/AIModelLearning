# 均方误差，，MSE就是：用一个数字表示模型总体错得有多严重。
# 有预测值和真实值  误差=预测-真实  误差有正负所以需要平方消除-号，这样才能求平均值
import numpy as np
# true_y=np.array([1,3])
# pred_y=np.array([2,5])
# error=true_y-pred_y
# MSE=np.mean(error**2)
# print(MSE)   写成函数更好
def mse(true_y,pred_y):
    error=true_y-pred_y
    return np.mean(error**2)
true_y = np.array([1, 3])
pred_y=np.array([2,5])
print(mse(true_y,pred_y))
