# CPU、GPU 与 device
import torch
# 确认是用gpu还是cpu
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tensor默认在cpu
x=torch.tensor([1.0,2.0,3.0])
print(f"移动前x:{x.device}")
# 将x搬到gpu作运算
x=x.to(device)
print(f"移动后x:{x.device}")
# 运算完若需要cpu工具进行下一步运算则要搬回来
result_cpu=(x*2).cpu()
print(f"结果位置：{result_cpu.device}")
#若要转成numpy数组 则需要  先分离梯度detach(),再搬回cpu(),再转成numpy数组
result_numpy=(x*2).detach().cpu().numpy()
print(result_numpy,type(result_numpy))