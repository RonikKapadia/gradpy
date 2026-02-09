from .tensor import Tensor
from .functions import *

class Linear():
    def __init__(self, in_size:int, out_size:int, requires_grad=False, init_params=True):
        self.in_size = in_size
        self.out_size = out_size
        self.requires_grad = requires_grad

        if init_params:
            self.init_params()
        
    def init_params(self):
        self.weights = random((self.in_size, self.out_size), requires_grad=self.requires_grad)
        self.bias = random((1, self.out_size), requires_grad=self.requires_grad)

    def load_params(self, params):
        weights, bias = params
        self.weights = weights
        self.bais = bias

    def __call__(self, x:Tensor) -> Tensor:
        return matmul(x, self.weights) + self.bias