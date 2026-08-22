"""PyTorch Tensor 第二阶段通关练习。

覆盖内容：NumPy/Tensor 转换、共享内存、dtype、view、transpose、
cat 与 stack、CPU/GPU device，以及安全转回 NumPy。

规则：
1. 只修改标有 TODO 的 6 个函数，不修改测试代码。
2. 不需要学习 Autograd 原理；第 6 题中的 detach 暂时按固定写法使用。
3. 运行：
   conda run -n dl-study python Pytorch_basic/pytorch_tensor_phase2_challenge.py
"""

import numpy as np
import torch


def numpy_to_shared_tensor(array):
    """把 NumPy 数组转成与原数组共享内存的 Tensor。"""
    # TODO：不要复制数据
    return torch.from_numpy(array)


def convert_to_float32(tensor):
    """返回 dtype 为 torch.float32 的 Tensor。"""
    # TODO
    return tensor.to(torch.float32)


def view_and_transpose(tensor):
    """先把 12 个元素 view 成 (3, 4)，再交换第 0、1 维。

    返回 (viewed, transposed)，目标 shape 分别是 (3, 4) 与 (4, 3)。
    """
    # TODO
    a=tensor.view((3,4))
    b=a.transpose(0,1)
    return a,b


def cat_and_stack(first, second):
    """沿 dim=0 分别使用 cat 和 stack 组合两个二维 Tensor。"""
    # TODO：返回 (concatenated, stacked)
    return torch.cat([first,second],dim=0),torch.stack([first,second],dim=0)


def move_to_training_device(tensor):
    """CUDA 可用时移到 cuda，否则留在 cpu，并返回移动后的 Tensor。"""
    # TODO：先确定 device，再使用 tensor.to(device)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tensor.to(device)


def tensor_to_numpy(tensor):
    """把可能在 GPU、可能记录梯度的 Tensor 安全地转成 NumPy 数组。"""
    # TODO：依次使用 detach、cpu、numpy
    return tensor.detach().cpu().numpy( )


def _test_numpy_to_shared_tensor():
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = numpy_to_shared_tensor(array)
    assert isinstance(result, torch.Tensor), "结果必须是 Tensor"
    assert result.dtype == torch.float32, "dtype 应与原 NumPy 数组一致"
    array[0] = 99.0
    assert result[0].item() == 99.0, "转换结果必须与 NumPy 数组共享内存"


def _test_convert_to_float32():
    source = torch.tensor([1, 2, 3], dtype=torch.int64)
    result = convert_to_float32(source)
    assert result.dtype == torch.float32, "结果 dtype 必须是 torch.float32"
    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0])), "数值不能改变"
    assert source.dtype == torch.int64, "不要原地改变传入 Tensor 的 dtype"


def _test_view_and_transpose():
    source = torch.arange(12)
    viewed, transposed = view_and_transpose(source)
    assert viewed.shape == (3, 4), "viewed 的 shape 应为 (3, 4)"
    assert transposed.shape == (4, 3), "transposed 的 shape 应为 (4, 3)"
    assert torch.equal(viewed, torch.arange(12).reshape(3, 4)), "view 后元素顺序不正确"
    assert torch.equal(transposed, viewed.T), "transpose 结果不正确"
    assert viewed.data_ptr() == source.data_ptr(), "view 不应复制底层数据"


def _test_cat_and_stack():
    first = torch.tensor([[1, 2], [3, 4]])
    second = torch.tensor([[5, 6], [7, 8]])
    concatenated, stacked = cat_and_stack(first, second)
    expected_cat = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    expected_stack = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert torch.equal(concatenated, expected_cat), "cat(dim=0) 结果不正确"
    assert torch.equal(stacked, expected_stack), "stack(dim=0) 结果不正确"
    assert concatenated.shape == (4, 2), "cat 不会新增维度"
    assert stacked.shape == (2, 2, 2), "stack 应新增一个维度"


def _test_move_to_training_device():
    source = torch.tensor([1.0, 2.0, 3.0])
    result = move_to_training_device(source)
    expected_type = "cuda" if torch.cuda.is_available() else "cpu"
    assert isinstance(result, torch.Tensor), "结果必须是 Tensor"
    assert result.device.type == expected_type, f"当前环境应使用 {expected_type}"
    assert torch.equal(result.cpu(), source), "移动设备后数值不能改变"


def _test_tensor_to_numpy():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = torch.tensor([1.5, 2.5, 3.5], device=device, requires_grad=True)
    result = tensor_to_numpy(source)
    assert isinstance(result, np.ndarray), "结果必须是 NumPy ndarray"
    assert np.allclose(result, np.array([1.5, 2.5, 3.5])), "转换后的数值不正确"


def run_tests():
    tests = [
        ("1. NumPy 转共享 Tensor", _test_numpy_to_shared_tensor),
        ("2. dtype 转换", _test_convert_to_float32),
        ("3. view 与 transpose", _test_view_and_transpose),
        ("4. cat 与 stack", _test_cat_and_stack),
        ("5. CPU/GPU device", _test_move_to_training_device),
        ("6. Tensor 安全转 NumPy", _test_tensor_to_numpy),
    ]

    passed = 0
    for name, test in tests:
        try:
            test()
        except Exception as error:
            detail = str(error) or error.__class__.__name__
            print(f"[未通过] {name}：{detail}")
        else:
            passed += 1
            print(f"[通过] {name}")

    print(f"\n结果：{passed}/{len(tests)}")
    if passed == len(tests):
        print("Tensor 阶段全部通过：下一步进入 Autograd！")
    else:
        print("继续完成 TODO；不理解的题目可以把题号和代码发给我。")


if __name__ == "__main__":
    run_tests()
