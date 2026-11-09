"""A pure-PyTorch stand-in for the parts of torch_scatter that ULTRA uses.

ULTRA imports torch_scatter, a compiled extension. On a machine with no
prebuilt wheel for its PyTorch version (Google Colab often moves ahead of the
wheels), building it from source takes ten minutes or more. Everything ULTRA's
entity-level NBFNet needs from it is a scatter sum, mean, max or min, which
PyTorch has had natively since 1.12, so this package provides those with the
same signatures and the same semantics (empty slots are filled with zero).

It is only installed when the real wheel is unavailable. Results agree with the
compiled extension to float rounding; the Colab notebook checks this against
the recorded benchmark before trusting it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

__version__ = '0.0.0+puretorch'


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    # same rule as torch_scatter.utils.broadcast
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


def _out(src, index, dim, out, dim_size):
    if out is not None:
        return out
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    return src.new_zeros(size)


def scatter_sum(src, index, dim: int = -1, out=None, dim_size: Optional[int] = None):
    index = broadcast(index, src, dim)
    out = _out(src, index, dim, out, dim_size)
    return out.scatter_add_(dim, index, src)


scatter_add = scatter_sum


def scatter_mean(src, index, dim: int = -1, out=None, dim_size: Optional[int] = None):
    total = scatter_sum(src, index, dim, out, dim_size)
    d = dim if dim >= 0 else src.dim() + dim
    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, min(d, index.dim() - 1), None, total.size(d))
    count = broadcast(count.clamp(min=1), total, d)
    if total.is_floating_point():
        return total / count
    return torch.div(total, count, rounding_mode='floor')


def _extreme(src, index, dim, out, dim_size, which):
    index = broadcast(index, src, dim)
    res = _out(src, index, dim, out, dim_size)
    res = res.scatter_reduce(dim, index, src, reduce=which, include_self=False)
    # argument: position along dim of the winning element, -1 for empty slots,
    # matching torch_scatter's convention of src.size(dim) for empty slots
    d = dim if dim >= 0 else src.dim() + dim
    pos = torch.arange(src.size(d), device=src.device)
    shape = [1] * src.dim()
    shape[d] = -1
    pos = pos.view(shape).expand_as(src)
    hit = src == res.gather(d, index)
    fill = torch.full(res.size(), src.size(d), dtype=torch.long, device=src.device)
    cand = torch.where(hit, pos, torch.full_like(pos, src.size(d)))
    arg = fill.scatter_reduce(d, index, cand, reduce='amin', include_self=True)
    return res, arg


def scatter_max(src, index, dim: int = -1, out=None,
                dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return _extreme(src, index, dim, out, dim_size, 'amax')


def scatter_min(src, index, dim: int = -1, out=None,
                dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return _extreme(src, index, dim, out, dim_size, 'amin')


def scatter_mul(src, index, dim: int = -1, out=None, dim_size: Optional[int] = None):
    index = broadcast(index, src, dim)
    res = _out(src, index, dim, out, dim_size)
    return res.fill_(1).scatter_reduce(dim, index, src, reduce='prod', include_self=True)


def scatter(src, index, dim: int = -1, out=None, dim_size: Optional[int] = None,
            reduce: str = 'sum'):
    if reduce in ('sum', 'add'):
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    if reduce == 'mul':
        return scatter_mul(src, index, dim, out, dim_size)
    if reduce == 'max':
        return scatter_max(src, index, dim, out, dim_size)[0]
    if reduce == 'min':
        return scatter_min(src, index, dim, out, dim_size)[0]
    raise ValueError(f'unknown reduce {reduce!r}')


from . import composite  # noqa: E402,F401  (ULTRA's variadic module imports it)
