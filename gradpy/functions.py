from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING: from .tensor import Tensor

import numpy as np
from . import backend

if TYPE_CHECKING: Data = np.ndarray | Tensor | list | float | int

def tensor(data:Data, requires_grad=False, dtype=np.float32) -> Tensor:
    from .tensor import Tensor
    if isinstance(data, Tensor):
        return data
    else:
        return Tensor(data=np.array(data, dtype=dtype), requires_grad=requires_grad)

def random(size, requires_grad=False) -> Tensor:
    return tensor(data=((np.random.random(size)*2.0)-1.0), requires_grad=requires_grad)

def ones(size, requires_grad=False) -> Tensor:
    return tensor(data=np.ones(size), requires_grad=requires_grad)

def split(a:Tensor, shape:tuple) -> Tensor:
    return backend.split_forwards(a, shape)

def norm_size(a:Tensor, b:Tensor) -> Tuple[Tensor, Tensor]:
    if a.shape != b.shape:
        if a.size < b.size:
            a = split(a, b.shape)
        if a.size > b.size:
            b = split(b, a.shape)
    return a, b

def add(a:Data, b:Data) -> Tensor:
    a, b = tensor(a), tensor(b)
    a, b = norm_size(a, b)
    return backend.add_forwards(a, b)

def sub(a:Data, b:Data) -> Tensor:
    a, b = tensor(a), tensor(b)
    a, b = norm_size(a, b)
    return backend.sub_forwards(a, b)

def mul(a:Data, b:Data) -> Tensor:
    a, b = tensor(a), tensor(b)
    a, b = norm_size(a, b)
    return backend.mul_forwards(a, b)

def div(a:Data, b:Data) -> Tensor:
    a, b = tensor(a), tensor(b)
    a, b = norm_size(a, b)
    return backend.div_forwards(a, b)

def pow(a:Data, b:Data) -> Tensor:
    a, b = tensor(a), tensor(b)
    a, b = norm_size(a, b)
    return backend.pow_forwards(a, b)

def matmul(a:Data, b:Data) -> Tensor:
    a, b = tensor(a), tensor(b)
    if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f'shapes {a.shape} and {b.shape} not compatible')
    return backend.matmul_forwards(a, b)

def dot(a:Data, b:Data):
    a, b = tensor(a), tensor(b)
    if a.shape != b.shape: 
        raise ValueError(f'shapes {a.shape} and {b.shape} not compatible')
    return backend.dot_forwards(a, b)

def sum(a:Data) -> Tensor:
    a = tensor(a)
    return backend.sum_forwards(a)

def sum_keepdims(a:Data, axis:int) -> Tensor:
    a = tensor(a)
    return backend.sum_keepdims_forwards(a, axis)

def abs(a:Data) -> Tensor:
    a = tensor(a)
    return backend.abs_forwards(a)

def transpose(a:Data) -> Tensor:
    a = tensor(a)
    return backend.transpose_forwards(a)

def max(a:Data, b:Data) -> Tensor:
    a, b = tensor(a), tensor(b)
    a, b = norm_size(a, b)
    return backend.max_forwards(a, b)

def min(a:Data, b:Data) -> Tensor:
    a, b = tensor(a), tensor(b)
    a, b = norm_size(a, b)
    return backend.min_forwards(a, b)

def log(a:Data) -> Tensor:
    a = tensor(a)
    return backend.log_forwards(a)

def sigmoid(a:Data) -> Tensor: 
    a = tensor(a)
    return 1/(1+np.e**(-1*a))

def selu(a:Data) -> Tensor:
    a = tensor(a)
    return a*1/(1+np.e**(-1*a))

def relu(a:Data) -> Tensor:
    a = tensor(a)
    b = tensor(0)
    return max(b, a)

def softmax(a:Data) -> Tensor:
    a = tensor(a)
    return (np.e**(a)/sum(np.e**(a)))