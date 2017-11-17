import numpy as np

import gradient_check as gc
from chainer import functions as F


def f(inputs):
    a, b = inputs
    return  np.matmul(a, b),


def df(inputs, grad_outputs):
    a, b = inputs
    gy, = grad_outputs
    return F.matmul(gy, b), F.matmul(a.T, gy)


def f2(inputs):
    a, = inputs
    return  a * 5,


def df2(inputs, grad_outputs):
    gy, = grad_outputs
    return 5 * gy,

if __name__ == '__main__':
    gc.check_backward(f, df, ((4, 4),(4, 4)))
    #gc.check_backward(f2, df2, ((3, 4),))
