import torch
# 添加缺失的函数
@torch.no_grad()
def offset2bincount(offset):
    """将offset转换为bincount"""
    offset = torch.cat([torch.tensor([0], device=offset.device, dtype=offset.dtype), offset])
    return offset[1:] - offset[:-1]


@torch.no_grad()
def bincount2offset(bincount):
    """将bincount转换为offset"""
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset2batch(offset):
    """将offset转换为batch索引"""
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch):
    """将batch索引转换为offset"""
    return torch.cumsum(batch.bincount(), dim=0).long()

