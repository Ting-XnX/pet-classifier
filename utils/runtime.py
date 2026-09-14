"""检测 CUDA 实际可执行性，而非仅检测驱动能否枚举 GPU。"""
import torch

def select_device():
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        probe=torch.ones(1,device='cuda')
        probe=probe+1
        torch.cuda.synchronize()
        del probe
    except RuntimeError as error:
        raise RuntimeError(
            'CUDA 驱动可见，但 PyTorch 无法在当前显卡执行运算。'
            '请使用支持该显卡架构的 PyTorch 构建；RTX 5060 本实验验证使用 CUDA 12.8 构建。'
        ) from error
    return torch.device('cuda')
