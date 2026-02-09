# GradPy

**GradPy** is a minimal automatic differentiation library (~500 lines) that brings PyTorch-like capabilities to NumPy. I created this project to understand how automatic differentiation works under the hood, and to provide a lightweight alternative for educational purposes and small experiments.

| Feature | GradPy | PyTorch |
|---------|--------|---------|
| Backend | NumPy | C++ / CUDA |
| Speed | Good for learning | Production-ready |
| Dependencies | NumPy only | torch, etc. |
| GPU Support | ❌ | ✅ |
| Code Size | ~500 lines | 100k+ lines |

**GradPy** is perfect for:
- Learning automatic differentiation from scratch
- Understanding how PyTorch works internally
- Lightweight projects where PyTorch is overkill
- Educational purposes

This project was inspired by [PyTorch](https://pytorch.org/), and projects like [micrograd](https://github.com/karpathy/micrograd) and [tinygrad](https://github.com/tinygrad/tinygrad).

## Installation

```bash
git clone https://github.com/RonikKapadia/gradpy.git
cd gradpy
pip install numpy
```

## Quick Start

#### Basic Tensor Operations

```python
import gradpy as gp

# Create tensors
a = gp.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = gp.tensor([4.0, 5.0, 6.0], requires_grad=True)

# Perform operations
c = a + b
d = c * 2

# Compute gradients
d.backward()

print(f"a.grad = {a.grad}")  # [2. 2. 2.]
print(f"b.grad = {b.grad}")  # [2. 2. 2.]
```

#### Building a Model with Classes

```python
import gradpy as gp

class Model:
    def __init__(self, requires_grad=True):
        # Define layers
        self.l1 = gp.Linear(2, 5, requires_grad=requires_grad)
        self.l2 = gp.Linear(5, 5, requires_grad=requires_grad)
        self.l3 = gp.Linear(5, 1, requires_grad=requires_grad)
    
    def __call__(self, x):
        # Forward pass with activations
        x = self.l1(x)
        x = gp.sigmoid(x)
        x = self.l2(x)
        x = gp.sigmoid(x)
        x = self.l3(x)
        x = gp.sigmoid(x)
        return x

# Create model
model = Model()

# Forward pass
input_data = gp.tensor([[0.5, 0.3]])
output = model(input_data)
print(f"Output: {output}")
```

#### The Training Loop

```python
import gradpy as gp
import numpy as np
from tqdm import tqdm

# Generate some data (XOR problem)
DATA_SIZE = 1000
BATCH_SIZE = 10
data_in = np.random.random((DATA_SIZE, 2))
data_in[data_in > 0.5] = 1
data_in[data_in < 0.5] = 0
data_out = np.expand_dims(np.logical_and(data_in[:,0], data_in[:,1])*1.0, -1)

# Create model
model = Model()

# Training loop
for step in tqdm(range(DATA_SIZE // BATCH_SIZE)):
    # Get batch
    x = gp.tensor(data_in[step*BATCH_SIZE:(step+1)*BATCH_SIZE])
    t = gp.tensor(data_out[step*BATCH_SIZE:(step+1)*BATCH_SIZE])
    
    # Forward pass
    y = model(x)
    
    # Compute loss (MSE)
    loss = gp.sum((t - y) ** 2)
    
    # Backward pass
    loss.backward()
    
    # Update weights (SGD with momentum)
    loss.optimize(lr=1e-3, momentum=0.9)
    
    # Zero gradients
    loss.zero_grad()
```

## Tips

1. **Always use `requires_grad=True`** for parameters you want to optimize
2. **Call `zero_grad()`** after each optimization step to clear gradients
3. **Use `view()`** instead of reshape to maintain gradient connections
4. **Start with small learning rates** (1e-3 to 1e-4) and adjust as needed
5. **Use momentum** (0.9 is a good default) for faster convergence

## API Reference

#### Tensor Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| `gp.tensor(data, requires_grad=False)` | Create a tensor | `x = gp.tensor([1, 2, 3])` |
| `gp.random(size, requires_grad=False)` | Random uniform [-1, 1] | `w = gp.random((3, 4))` |
| `gp.ones(size, requires_grad=False)` | Tensor of ones | `b = gp.ones((1, 5))` |
| `x.backward(grad=1.0)` | Compute gradients | `loss.backward()` |
| `x.optimize(lr=1e-3, momentum=0.9)` | SGD update | `loss.optimize(lr=0.01)` |
| `x.zero_grad()` | Zero gradients | `loss.zero_grad()` |

#### Mathematical Operations

```python
# Arithmetic
z = x + y    # Addition
z = x - y    # Subtraction
z = x * y    # Element-wise multiplication
z = x / y    # Division
z = x ** 2   # Power

# Matrix operations
z = gp.matmul(x, y)  # Matrix multiplication
z = gp.dot(x, y)     # Dot product
z = gp.transpose(x)  # Transpose

# Reductions
z = gp.sum(x)              # Sum all elements
z = gp.sum_keepdims(x, 0)  # Sum along axis, keep dims

# Other
z = gp.abs(x)        # Absolute value
z = gp.log(x)        # Natural logarithm
z = gp.max(x, y)     # Element-wise maximum
z = gp.min(x, y)     # Element-wise minimum
```


#### View and Reshape

```python
x = gp.tensor(np.random.randn(2, 3, 4))
y = x.view((6, 4))  # Reshape while preserving gradients
```

#### Indexing

```python
x = gp.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
y = x[0]      # Get first row
z = x[0, 1]   # Get element at [0, 1]
```

#### Activation Functions

```python
y = gp.relu(x)      # ReLU: max(0, x)
y = gp.sigmoid(x)   # Sigmoid: 1 / (1 + exp(-x))
y = gp.selu(x)      # SELU: x * sigmoid(x)
y = gp.softmax(x)   # Softmax: exp(x) / sum(exp(x))
```

#### Neural Network Layers

```python
# Linear layer (fully connected)
layer = gp.Linear(in_size=784, out_size=128, requires_grad=True)
output = layer(input_tensor)  # Computes: input @ W + b
```

## Architecture

```
gradpy/
├── __init__.py      # Main exports
├── tensor.py        # Tensor class with autograd
├── functions.py     # Tensor creation and operations
├── backend.py       # Gradient computation (forward and backward functions)
└── classes.py       # Neural network layers
```

## Examples

```bash
# Run the example notebooks
jupyter notebook gradpy.ipynb
jupyter notebook torch.ipynb  # Comparison with PyTorch
```

## License

MIT License - feel free to use this for any project!

