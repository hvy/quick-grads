

import gradient_check as gc
from chainer import cuda
from chainer import functions as F


def f(a, b):
    xp = cuda.get_array_module(a)
    return  xp.matmul(a, b), xp.matmul(b, a)


def df(a, b, gy0, gy1):
    xp = cuda.get_array_module(a)
    # return F.matmul(gy0, b.T) + F.matmul(b.T, gy1), F.matmul(gy1, a.T) + F.matmul(a.T, gy0)
    return None, F.matmul(gy1, a.T) + F.matmul(a.T, gy0)
    # return None, xp.matmul(a.T, gy0) + xp.matmul(gy1, a.T)


if __name__ == '__main__':
    gc.check_backward(f, df, ((10, 10),(10, 10)), gpu=0, rtol=1e-2, atol=1e-2, no_grads=(True, False))
    # gc.check_backward(f2, df2, ((10, 16),(16, 12)), gpu=0, rtol=1e-2, atol=1e-2)
    #gc.check_backward(f2, df2, ((3, 4),))
