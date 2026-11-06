"""Softmax over scatter groups, as torch_scatter.composite provides."""
from __future__ import annotations

import torch

from . import broadcast, scatter_max, scatter_sum


def scatter_softmax(src, index, dim: int = -1, dim_size=None):
    index_b = broadcast(index, src, dim)
    mx = scatter_max(src, index, dim, dim_size=dim_size)[0]
    shifted = src - mx.gather(dim, index_b)
    e = shifted.exp()
    s = scatter_sum(e, index, dim, dim_size=dim_size) + 1e-16
    return e / s.gather(dim, index_b)


def scatter_log_softmax(src, index, dim: int = -1, eps: float = 1e-12, dim_size=None):
    index_b = broadcast(index, src, dim)
    mx = scatter_max(src, index, dim, dim_size=dim_size)[0]
    shifted = src - mx.gather(dim, index_b)
    s = scatter_sum(shifted.exp(), index, dim, dim_size=dim_size)
    return shifted - torch.log(s.gather(dim, index_b) + eps)
