from __future__ import annotations
from typing import Tuple, List, Callable

import numpy as np
from .functions import *

class Tensor():
    def __init__(self, data:np.ndarray, requires_grad=False, children:list[Tensor]=[], grad_fn:Callable|None=None, grad_fn_args:List=[]):
        self.data = data
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad_fn_args = grad_fn_args
        self.children = children

        if not self.children and self.requires_grad:
            self.init_grad()

    def init_grad(self):
        self.requires_grad = True
        self.grad = np.zeros(self.data.shape)
        self.dw = 0.0

    def backward(self, grad:float|np.ndarray=1.0):
        if self.requires_grad:
            if self.grad_fn:
                self.grad_fn(self, grad)
            else:
                self.grad += grad

    def optimize(self, lr=1e-3, momentum=0.9):
        if self.requires_grad:
            if self.children:
                for child in self.children: child.optimize(lr)
            else:
                self.dw = momentum*self.dw - self.grad*lr
                self.data += self.dw

    def zero_grad(self):
        if self.requires_grad:
            if self.children:
                for child in self.children: child.zero_grad()
            else:
                self.grad *= 0.0
                self.dw *= 0.0

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size
    
    def view(self, shape:Tuple[int, ...]):
        new = Tensor(
            data = np.reshape(self.data, shape),
            requires_grad=self.requires_grad,
            children=self.children,
            grad_fn=self.grad_fn,
        )
        if not new.children and new.requires_grad:
            new.grad = np.reshape(self.grad, shape)

        if new.data.base is None or new.grad.base is None:
            raise ValueError("Numpy reshape returning copy not view")
        return new
    
    def __repr__(self):
        return f'{self.data}'
    
    def __getitem__(self, key):
        if isinstance(key, int) and len(self.data.shape) == 1:
            key = slice(key,key+1)
        new = Tensor(
            data = self.data[key],
            requires_grad=self.requires_grad,
            children=self.children,
            grad_fn=self.grad_fn,
        )
        if not new.children and new.requires_grad:
            new.grad = self.grad[key]
        return new
    
    def __setitem__(self, key, new):
        if not self.requires_grad:
            self.data[key] = new.data
        else:
            raise ValueError("Can't __setitem__ with a tensor that requires_grad")
    
    def __add__(self, other):
        return add(self, other)
    
    def __radd__(self, other):
        return add(other, self)
    
    def __sub__(self, other):
        return sub(self, other)
    
    def __rsub__(self, other):
        return sub(other, self)
    
    def __mul__(self, other):
        return mul(self, other)
    
    def __rmul__(self, other):
        return mul(other, self)
    
    def __truediv__(self, other):
        return div(self, other)
    
    def __rtruediv__(self, other):
        return div(other, self)
    
    def __pow__(self, other):
        return pow(self, other)
    
    def __rpow__(self, other):
        return pow(other, self)
    
    def __iadd__(self, other):
        return add(self, other)
    
    def __isub__(self, other):
        return sub(self, other)
    
    def __imul__(self, other):
        return mul(self, other)
    
    def __idiv__(self, other):
        return div(self, other)
    
    def __ipow__(self, other):
        return pow(self, other)
    
    def __neg__(self):
        return sub(0, self)
    
    def __pos__(self):
        return self