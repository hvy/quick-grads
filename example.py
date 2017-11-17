import numpy as np

import gradient_check as gc
from chainer import functions as F


def f(a, b):
    return  np.matmul(a, b),


def df(a, b, gy):
    return F.matmul(gy, b.T), F.matmul(a.T, gy)


def f2(inputs):
    a, = inputs
    return  a * 5,


def df2(inputs, grad_outputs):
    gy, = grad_outputs
    return 5 * gy,

if __name__ == '__main__':
    gc.check_backward(f, df, ((4, 4),(4, 4)))
    #gc.check_backward(f2, df2, ((3, 4),))
