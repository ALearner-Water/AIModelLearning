import numpy
import torch
x1=numpy.array([2],dtype=numpy.float64).reshape(1,1)
x2=torch.tensor([2],dtype=torch.float32,requires_grad=True)

ep=10**-4
def y(x):
    return x**3+2*x

ay_1=y(x1+ep)
ay_2=y(x1-ep)
y2=y(x2)

loss1=numpy.mean((ay_1-ay_2)/(2*ep))
y2.backward()

print(loss1, x2.grad.item(), abs(loss1 - x2.grad.item()))
