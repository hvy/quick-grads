# Easy Numerical Gradient Check

A single function, `check_gradient()` to perform [numerical gradient check](https://en.wikipedia.org/wiki/Numerical_differentiation) on NumPy and [CuPy](https://github.com/cupy/cupy) ndarrays as well as [Chainer](https://github.com/chainer/chainer) variable objects.

The gradient check can be used to "check" the correctness of an analytical gradient function by comparing the output results with numerically computed ones. If the analytical gradients are wrong, `check_gradient()` raises an error. Otherwise, it does nothing. This gradient check basically wraps the gradient check of Chainer, but allows the user to skip having to implement `chainer.function.Function`s or `chainer.function_node.FunctionNode`s.

## Example

Basic usage with NumPy, elementwise operations.

```python
from gradient_check import check_gradient

def f(x):
    y = 0.5 * x * x
    return y

def df(x, gy):
    gx = x * gy
    return gx

def df_wrong(x, gy):
    gx = x * x * gy
    return gx

check_gradient(f, df, input_shapes=((3, 4),))  # No error
check_gradient(f, df_wrong, input_shapes=((3, 4),))  # Raises an AssertionError
```

Similar CuPy example with matrix multiplication,.

```python
from gradient_check import check_gradient

def f(x, y):
    z = x @ y
    return z

def df(x, y, gz):
    gx = gz @ y.T
    gy = x.T @ gz
    return gx, gy

check_gradient(f, df, input_shapes=((3, 4), (4, 2)), device=0)
```

You can replace `df` in the previous example with a function that returns [chainer.Variable](https://docs.chainer.org/en/stable/reference/core/generated/chainer.Variable.html#chainer.Variable).

```python
import chainer

def df(x, y, gz):
    gx = chainer.functions.matmul(gz, y, transb=True)
    gy = chainer.functions.matmul(x, gz, transa=True)
    return gx, gy
```

## Requirements

- Python 3.5+
  - Chainer 3.0.0+
  - NumPy 1.13.0+
  - (Optional) CuPy 2.0.0+

 ### Install Requirements
 ```bash
 pip install chainer
 pip install numpy
 pip install cupy
 ```
