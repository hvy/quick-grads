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


def _as_function(f, df):
    print('Function')
    class _Function(function.Function):
        def forward(self, inputs):
            return _as_iterable(f(*inputs))

        def backward(self, inputs, grad_outputs):
            return _as_iterable(df(*inputs, *grad_outputs))

    def _func(*inputs):
        return _Function()(*inputs)
    return _func


def _as_function_node(f, df):
    print('FunctionNode')
    class _FunctionNode(function_node.FunctionNode):
        def forward(self, inputs):
            self.retain_inputs(tuple(range(len(inputs))))
            return _as_iterable(f(*inputs))

        def backward(self, indexes, grad_outputs):
            inputs = self.get_retained_inputs()
            return _as_iterable(df(*(inputs + grad_outputs)))

    def _func(*inputs):
        y = _FunctionNode().apply(_as_iterable(inputs))
        if len(y) == 1:
            return y[0]
        return y
    return _func


def _make_rnd_data(shape, dtype='f', xp=numpy):
    return xp.random.uniform(-1, 1, shape).astype(dtype)


def make_rnd_data_numpy(shape, dtype='f'):
    """Generates a NumPy ndarray with random data of given shape.

    Args:
        shape (tuple): Shape of ndarray to generate.
        dtype (str or type): Dtype of array to generate. Default is 'f'.

    Return:
        numpy.ndarray: Random NumPy ndarray.

    """
    return _make_rnd_data(shape, dtype, numpy)


def make_rnd_data_cupy(shape, dtype='f'):
    """Generates a CuPy ndarray with random data of given shape.

    Args:
        shape (tuple): Shape of ndarray to generate.
        dtype (str or type): Dtype of array to generate. Default is 'f'.

    Return:
        numpy.ndarray: Random CuPy ndarray.

    """
    return _make_rnd_data(shape, dtype, cuda.cupy)


def check_backward(f, df, input_shapes, make_inputs=None,
                   make_grad_outputs=None, inputs=None, grad_outputs=None,
                   gpu=-1, atol=1e-5, rtol=1e-4, eps=1e-3, no_grads=None):
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
    if make_inputs is None or make_grad_outputs is None:
        if gpu < 0:
            xp = numpy
        else:
            if cuda.available:
                cuda.get_device_from_id(gpu).use()
                xp = cuda.cupy
            else:
                raise ValueError(
                    'Tried to use the GPU but CuPy was not available.')
        if make_inputs is None:
            make_inputs = \
                lambda shape: _make_rnd_data(shape, dtype='f', xp=xp)
        if make_grad_outputs is None:
            make_grad_outputs = \
                lambda shape: _make_rnd_data(shape, dtype='f', xp=xp)

    assert make_inputs is not None
    assert make_grad_outputs is not None

    def _generate_ndarray(_generator, _shapes):
        if callable(_generator):
            arr = tuple(_generator(shape) for shape in _shapes)
        elif len(_generator) == len(_shapes):
            arr = tuple(
                _generator(shape) for shape in zip(_generator, _shapes))
        else:
            raise ValueError(
                'Array specific generators were given but the number of'
                'generators did not match the number of inputs.')
        return arr

    if inputs is None:
        inputs = _generate_ndarray(make_inputs, input_shapes)

    outputs = f(*inputs)
    output_shapes = [output.shape for output in outputs]
    if grad_outputs is None:
        grad_outputs = _generate_ndarray(make_grad_outputs, output_shapes)

    # Use `chainer.function.Function` or `chainer.function_node.FunctionNode`
    # based on the class types of the outputs of `df`.
    grads = df(*inputs, *grad_outputs)
    grads = filter(lambda g: g is not None, grads)
    is_grad_variables = [isinstance(g, Variable) for g in grads]
    is_grad_ndarrays = [
        isinstance(g, (numpy.ndarray, cuda.ndarray)) for g in grads]

    if all(is_grad_variables):
        func = _as_function_node(f, df)
    elif all(is_grad_ndarrays):
        func = _as_function(f, df)
    else:
        raise ValueError('Cannot mix Variables and numpy/cupy ndarrays.')

    gradient_check.check_backward(
        func, inputs, grad_outputs, eps=eps, atol=atol, rtol=rtol,
        no_grads=no_grads)
