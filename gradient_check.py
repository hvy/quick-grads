import numpy

from chainer import cuda
from chainer import function
from chainer import function_node
from chainer import gradient_check
from chainer import Variable


def _as_iterable(x):
    if isinstance(x, (tuple, list)):
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


def check_gradient(f, df, input_shapes, make_inputs=None,
                   make_grad_outputs=None, inputs=None, grad_outputs=None,
                   device=-1, eps=1e-3, atol=1e-5, rtol=1e-4, no_grads=None):
    """Checks the gradient computations for the given function.

    This function raises an error in case the gradients computed by `df`
    differ from the numerically computed gradients based on `f` and inputs
    offsetted by `eps`, with more than a certain tolerance. Otherwise, this
    function does not raise an exception and simply passes.

    Inputs and upstream gradients are randomly generated based on the
    `input_shapes` and the outputs computed by `f`, unless generators or
    ndarray data is explicitly given.

    Args:
        f (function):
            Function to differentiate.
        df (function):
            Gradient function, that takes the same inputs as `f` as well as
            output gradients for each output of `f`.
        input_shapes (tuple):
            Tuple of shapes of the inputs to `f`. The length of the tuple must
            equal the number of inputs.
        make_inputs (function):
            Optional sampler function to generate input data. The sampler must
            take a shape (tuple of ints) and return a NumPy or CuPy ndarray.
            If `None`, a random uniform sampler is used to generate the data
            on the device specified by `device`.
        make_grad_outputs (function):
            Optional sampler function to generate upstream gradients. Acts
            similary to `make_inputs`.
        inputs (numpy.ndarray or cupy.ndarray):
            Optional input data. If given, `input_shapes` and `make_inputs`
            are ignored.
        grad_outputs (numpy.ndarray or cupy.ndarray):
            Optional upstream gradients. If given, `make_grad_outputs` are
            ignored. The shapes must match the output of `f`.
        device (int):
            Device id. If `<0`, the device is CPU and NumPy ndarrays
            are generated by the samplers. If `>=0`, the GPU with the
            corresponding id is used to generate CuPy ndarrays instead.
        eps (float):
            Numerical gradient offset.
        atol (float):
            Absolute tolerance for the numerical gradient check.
        rtol (float):
            Relative tolerance for the numericla gradient check.
        no_grads (tuple of bool):
            Optional tuple with boolean values. If specified, the length of the
            tuple must equal the number of inputs to `f`. For each boolean
            value in the tuple, the corresponding gradient is ignored in the
            gradient check.

    .. admonition:: Example

        Matrix multiplication example.

        >>> def f(a, b):
        >>>     y = a @ b
        >>>     return y
        >>>
        >>> def df(a, b, gy):
        >>>     ga = gy @ b.T
        >>>     gb = a.T @ gy
        >>>     return ga, gb
        >>>
        >>> check_gradient(f, df, input_shapes=((3, 6), (6, 8)))

    """
    def _make_ndarrays(_sampler, _shapes):
        if _sampler is None:
            xp = _get_array_module(device)  # NumPy or CuPy

            def _sampler(shape):
                return xp.random.uniform(-1, 1, shape).astype('f')
        return _sample_ndarrays(_sampler, _shapes)

    if inputs is None:
        inputs = _make_ndarrays(make_inputs, input_shapes)

    if grad_outputs is None:
        output_shapes = [output.shape for output in _as_iterable(f(*inputs))]
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
