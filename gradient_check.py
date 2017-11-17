# TODO: Allow old style functions, e.g. backward with ndarray. THis allows user
# to not depend on Chainer!
# TODO: Allow GPU, maybe this is implicit?
# TODO: Test Both Variable and ndarray, add options to specity?

import numpy as np

from chainer import function_node
from chainer import gradient_check


def _as_iterable(x):
    if isinstance(x, (tuple, list)):
        return x
    return x,


def _as_function_node(f, df):

    class _FunctionNode(function_node.FunctionNode):
        def forward(self, inputs):
            self.retain_inputs(tuple(range(len(inputs))))
            out = _as_iterable(f(*inputs))
            return out

        def backward(self, indexes, grad_outputs):
            inputs = self.get_retained_inputs()
            return _as_iterable(df(*(inputs + grad_outputs)))

    def func(*inputs):
        return _FunctionNode().apply(_as_iterable(inputs))[0]

    return func


def _rnd_ndarray(shape, dtype='f'):
    return np.random.uniform(-1, 1, shape).astype(dtype)



def check_backward(f, df, input_shapes, atol=1e-5, rtol=1e-4):
    inputs = tuple(_rnd_ndarray(shape) for shape in input_shapes)
    outputs = f(*inputs)
    output_shapes = [output.shape for output in outputs]
    grad_outputs = tuple(_rnd_ndarray(shape) for shape in output_shapes)

    fn = _as_function_node(f, df)
    gradient_check.check_backward(fn, inputs, grad_outputs)
