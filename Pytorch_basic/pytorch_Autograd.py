import torch
x=torch.tensor(4.0)
true_y=torch.tensor(10.0)

w=torch.tensor(1.0,requires_grad=True)    #这样才可以求w的梯度
b=torch.tensor(3.0,requires_grad=True)

pred_y=w*x+b
loss=(pred_y-true_y)**2
# 从loss开始，反向计算梯度
loss.backward()

print(f"pred_y={pred_y}")
print(f"loss={loss}")
print(f"w_grad={w.grad.item()}")  #.grad用来保存和查看梯度解结果
print(f"b_grad={b.grad.item()}")

# with是 Python 用来创建一个临时执行环境的关键字。
# 因为更新参数不需要计算梯度所以创建一个临时环境来更新
learning_rate=0.01
with torch.no_grad():
    w-=w.grad*learning_rate
    b-=b.grad*learning_rate

#更新完后计算新pred_y和loss
new_pred_y=w*x+b
new_loss=(new_pred_y-true_y)**2
print(f"new_pred_y={new_pred_y}")
print(f"new_loss={new_loss}")

#更新完后再次计算新梯度会默认累加，不会自动替换 所以每一轮更新前需要清零
print("before zero",w.grad.item(),b.grad.item())
w.grad.zero_()
b.grad.zero_()
print("after zero",w.grad.item(),b.grad.item())

new_loss.backward()

print(f"new_w_grad={w.grad.item()}")
print(f"new_b_grad={b.grad.item()}")