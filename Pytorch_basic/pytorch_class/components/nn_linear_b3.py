import torch
from torch import nn  

# 引出nn.Linear
X = torch.tensor(
    [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0]
    ]
)
w = torch.tensor([[2.0], [-1.0]])
b = torch.tensor([0.5])
pred_y = X @ w + b
# print(f"X.shape={X.shape}")
# print(f"w.shape={w.shape}")
# print(f"b.shape={b.shape}")
# print(f"pred_y.shape={pred_y.shape}")
# print(f"pred_y={pred_y}")
linear = nn.Linear(2, 1)
linear_pred_y = linear(X)
print("linear weight:", linear.weight.shape)
print("linear bias:", linear.bias.shape)
print("linear output:", linear_pred_y.shape)
print(linear_pred_y)
