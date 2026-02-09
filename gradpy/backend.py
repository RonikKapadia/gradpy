from __future__ import annotations
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING: from .tensor import Tensor

import numpy as np

# split
def split_forwards(a:Tensor, shape:int|Tuple) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = np.ones(shape) * a.data, 
        children = [a], 
        grad_fn = split_backwards, 
        requires_grad = True,
        grad_fn_args = [shape],
    )

def split_backwards(self:Tensor, grad):
    a, = self.children
    shape, = self.grad_fn_args
    axis = np.argmax(np.array(self.shape) != np.array(shape))
    if a.requires_grad:
        a.backward(np.sum(grad, axis=axis, keepdims=True))

# addition
def add_forwards(a:Tensor, b:Tensor) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = a.data + b.data,
        children = [a, b],
        grad_fn = add_backwards,
        requires_grad = True,
    )

def add_backwards(self:Tensor, grad):
    a, b = self.children
    if a.requires_grad:
        a.backward(grad*np.ones(b.data.shape))
    if b.requires_grad:
        b.backward(grad*np.ones(a.data.shape))

# subtraction
def sub_forwards(a:Tensor, b:Tensor) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = a.data - b.data,
        children = [a, b],
        grad_fn = sub_backwards,
        requires_grad = True,
    )

def sub_backwards(self:Tensor, grad):
    a, b = self.children
    if a.requires_grad:
        a.backward(grad*np.ones(b.data.shape))
    if b.requires_grad:
        b.backward(-1*grad*np.ones(b.data.shape))

# multiplication
def mul_forwards(a:Tensor, b:Tensor) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = a.data * b.data,
        children = [a, b],
        grad_fn = mul_backwards,
        requires_grad = True,
    )

def mul_backwards(self:Tensor, grad):
    a, b = self.children
    if a.requires_grad:
        a.backward((grad)*(b.data))
    if b.requires_grad:
        b.backward((grad)*(a.data))

# division
def div_forwards(a:Tensor, b:Tensor) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = a.data / b.data,
        children = [a, b],
        grad_fn = div_backwards,
        requires_grad = True,
    )

def div_backwards(self:Tensor, grad):
    a, b = self.children
    if a.requires_grad:
        a.backward((grad)*(1/b.data))
    if b.requires_grad:
        b.backward((grad)*((-a.data)/b.data**2))

# pow
def pow_forwards(a:Tensor, b:Tensor) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = a.data ** b.data,
        children = [a, b],
        grad_fn = pow_backwards,
        requires_grad = True,
    )

def pow_backwards(self:Tensor, grad):
    a, b = self.children
    if a.requires_grad:
        a.backward((grad)*(b.data*a.data**(b.data-1)))
    if b.requires_grad:
        b.backward((grad)*((a.data**b.data)*np.log(a.data)))

# matmul
def matmul_forwards(a:Tensor, b:Tensor) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = np.matmul(a.data, b.data),
        children = [a, b],
        grad_fn = matmul_backwards,
        requires_grad = True,
    )

def matmul_backwards(self:Tensor, grad):
    a, b = self.children
    if a.requires_grad:
        a.backward(np.matmul(grad * np.ones(self.shape), b.data.T))
    if b.requires_grad:
        b.backward(np.matmul(a.data.T, grad * np.ones(self.shape)))

# dot
def dot_forwards(a:Tensor, b:Tensor) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = np.dot(a.data, b.data),
        children = [a, b],
        grad_fn = dot_backwards,
        requires_grad = True,
    )

def dot_backwards(self:Tensor, grad):
    a, b = self.children
    if a.requires_grad:
        a.backward((grad)*(b.data))
    if b.requires_grad:
        b.backward((grad)*(a.data))

# sum
def sum_forwards(a:Tensor) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = np.sum(a.data),
        children = [a],
        grad_fn = sum_backwards,
        requires_grad = True,
    )

def sum_backwards(self:Tensor, grad):
    a, = self.children
    if a.requires_grad:
        a.backward(grad)

# sum keepdims
def sum_keepdims_forwards(a:Tensor, axis:int) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = np.sum(a.data, axis=axis, keepdims=True),
        children = [a],
        grad_fn = sum_keepdims_backwards,
        requires_grad = True,
    )

def sum_keepdims_backwards(self:Tensor, grad):
    a, = self.children
    if a.requires_grad:
        a.backward(grad)

# abs
def abs_forwards(a:Tensor) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = np.abs(a.data),
        children = [a],
        grad_fn = abs_backwards,
        requires_grad = True,
    )

def abs_backwards(self:Tensor, grad):
    a, = self.children
    if a.requires_grad:
        a.backward((grad)*(a.data/np.abs(a.data)))

# transpose
def transpose_forwards(a:Tensor) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = np.transpose(a.data),
        children = [a],
        grad_fn = transpose_backwards,
        requires_grad = True,
    )

def transpose_backwards(self:Tensor, grad):
    a, = self.children
    if a.requires_grad:
        a.backward(np.transpose(grad))

# max
def max_forwards(a:Tensor, b:Tensor) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = np.maximum(a.data, b.data),
        children = [a, b],
        grad_fn = max_backwards,
        requires_grad = True,
    )

def max_backwards(self:Tensor, grad):
    a, b = self.children
    if a.requires_grad:
        a.backward(((a.data > b.data)*1.0) + (a.data == b.data)*0.5)
    if b.requires_grad:
        b.backward(((b.data > a.data)*1.0) + (a.data == b.data)*0.5)

# min
def min_forwards(a:Tensor, b:Tensor) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = np.minimum(a.data, b.data),
        children = [a, b],
        grad_fn = min_backwards,
        requires_grad = True,
    )

def min_backwards(self:Tensor, grad):
    a, b = self.children
    if a.requires_grad:
        a.backward(((a.data < b.data)*1.0) + (a.data == b.data)*0.5)
    if b.requires_grad:
        b.backward(((b.data < a.data)*1.0) + (a.data == b.data)*0.5)

# log
def log_forwards(a:Tensor) -> Tensor:
    from .tensor import Tensor
    return Tensor(
        data = np.log(a.data),
        children = [a],
        grad_fn = log_backwards,
        requires_grad = True,
    )

def log_backwards(self:Tensor, grad):
    a, = self.children
    if a.requires_grad:
        a.backward((grad)*(1/a.data))