import numpy

from chainer import cuda
from chainer import function
from chainer import function_node
from chainer import gradient_check
from chainer import Variable


def _as_iterable(x):
    if hasattr(x, '__iter__'):
        return x
    return x,


def _get_array_module(device):
    if device < 0:
        return numpy
    elif cuda.available:
        cuda.get_device_from_id(device).use()
        return cuda.cupy
    raise ValueError('Tried to use the GPU but CuPy was not available.')


def _sample_ndarrays(sampler, shapes):
    if callable(sampler):
        return tuple(sampler(shape) for shape in shapes)
    elif len(sampler) == len(shapes):
        return tuple(
            _sampler(shape) for _sampler, shape in zip(sampler, shapes))
    raise ValueError(
        'Array specific generators were given but the number of'
        'generators did not match the number of inputs.')


def _as_function(f, df):
    class _Function(function.Function):

        def forward(self, inputs):
            return _as_iterable(f(*inputs))

        def backward(self, inputs, grad_outputs):
            return _as_iterable(df(*inputs, *grad_outputs))

    return lambda *inputs: _Function()(*inputs)


def _as_function_node(f, df):
    class _FunctionNode(function_node.FunctionNode):

        def forward(self, inputs):
            self.retain_inputs(tuple(range(len(inputs))))
            return _as_iterable(f(*inputs))

        def backward(self, indexes, grad_outputs):
            inputs = self.get_retained_inputs()
            return _as_iterable(df(*inputs, *grad_outputs))

    def _func(*inputs):
        output = _FunctionNode().apply(_as_iterable(inputs))
        if isinstance(output, tuple) and len(output) == 1:
            output, = output
        return output
    return _func


def check_backward(f, df, input_shapes, make_inputs=None,
                   make_grad_outputs=None, inputs=None, grad_outputs=None,
                   device=-1, eps=1e-3, atol=1e-5, rtol=1e-4, no_grads=None):
    """Check the gradient computations of the given function.

    Inputs and upstream gradients are randomly generated to test for the
    correctness of the gradient function by computing numerical gradients.

    Args:
        f (function): Function to differentiate.
        df (function): Gradient computation of the function `f`.
        input_shapes (tuple): Tuple of shapes of the inputs to `f`. The length
            of the tuple must equal the number of inputs.
        gpu (int): Device id. If `<0`, the device is CPU and NumPy ndarrays
            are used as inputs. If `>=0`, the corresponding GPU is used along
            with CuPy to generate the data. Data is randomly generated unless
            `make_data` is specified.
        make_data (function):

    """
    def _make_ndarrays(_sampler, _shapes):
        if _sampler is None:
            xp = _get_array_module(device)  # NumPy or CuPy
            _sampler = lambda shape: xp.random.uniform(-1, 1, shape, dtype='f')
        return _sample_ndarrays(_sampler, _shapes)

    if inputs is None:
        inputs = _make_ndarrays(make_inputs, input_shapes)

    if grad_outputs is None:
        output_shapes = [output.shape for output in f(*inputs)]
        grad_outputs = _make_ndarrays(make_grad_outputs, output_shapes)

    # Use `chainer.function.Function` or `chainer.function_node.FunctionNode`
    # depnding on the class types of the outputs of `df`.
    grads = df(*inputs, *grad_outputs)
    grads = [g for g in grads if g is not None]

    is_grads_variable = [isinstance(g, Variable) for g in grads]
    is_grads_ndarray = [
        isinstance(g, (numpy.ndarray, cuda.ndarray)) for g in grads]

    if all(is_grads_variable):
        func = _as_function_node(f, df)
    elif all(is_grads_ndarray):
        func = _as_function(f, df)
    else:
        raise ValueError('Cannot mix Variable and NumPy/CuPy gradients.')

    gradient_check.check_backward(
        func, inputs, grad_outputs, eps=eps, atol=atol, rtol=rtol,
        no_grads=no_grads)
