import lasagne
from lasagne.layers.conv import BaseConvLayer
from lasagne import init
from lasagne import nonlinearities
from lasagne.utils import as_tuple
from lasagne.layers.conv import conv_output_length
from lasagne.layers.base import Layer
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.abstract_conv import BaseAbstractConv2d, get_conv_output_shape
import theano
import numpy
from theano.tensor import as_tensor_variable, patternbroadcast
from theano.tensor import get_scalar_constant_value, NotScalarConstantError
from six import reraise, integer_types
import numpy
try:
    from scipy.signal.signaltools import _valfrommode, _bvalfromboundary
    from scipy.signal.sigtools import _convolve2d
    imported_scipy_signal = True
except ImportError:
    imported_scipy_signal = False
import sys
from theano.gof import Apply, Op
import warnings


class BaseConvLayerR(Layer):
    """
    lasagne.layers.BaseConvLayer(incoming, num_filters, filter_size,
    stride=1, pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
    n=None, **kwargs)

    Convolutional layer base class

    Base class for performing an `n`-dimensional convolution on its input,
    optionally adding a bias and applying an elementwise nonlinearity. Note
    that this class cannot be used in a Lasagne network, only its subclasses
    can (e.g., :class:`Conv1DLayer`, :class:`Conv2DLayer`).

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. Must
        be a tensor of 2+`n` dimensions:
        ``(batch_size, num_input_channels, <n spatial dimensions>)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or an `n`-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or an `n`-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of `n` integers allows different symmetric padding
        per dimension.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If ``True``, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be an
        `n`-dimensional tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a tensor of 2+`n` dimensions with shape
        ``(num_filters, num_input_channels, <n spatial dimensions>)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, <n spatial dimensions>)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    flip_filters : bool (default: True)
        Whether to flip the filters before sliding them over the input,
        performing a convolution (this is the default), or not to flip them and
        perform a correlation. Note that for some other convolutional layers in
        Lasagne, flipping incurs an overhead and is disabled by default --
        check the documentation when using learned weights from another layer.

    n : int or None
        The dimensionality of the convolution (i.e., the number of spatial
        dimensions of each feature map and each convolutional filter). If
        ``None``, will be inferred from the input shape.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 untie_biases=False,
                 W=init.GlorotUniform(), R=init.GlorotUniform, b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 n=None, **kwargs):
        super(BaseConvLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        if n is None:
            n = len(self.input_shape) - 2
        elif n != len(self.input_shape) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." %
                             (n, self.input_shape, n+2, n))
        self.n = n
        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, n, int)
        self.flip_filters = flip_filters
        self.stride = as_tuple(stride, n, int)
        self.untie_biases = untie_biases


        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = as_tuple(0, n)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, n, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        self.R = self.add_param(R, self.get_W_shape(), name="R")

        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters,) + self.output_shape[2:]
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels) + self.filter_size

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, pad)))

    def get_output_for(self, input, **kwargs):
        conved = self.convolve(input, **kwargs)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + T.shape_padleft(self.b, 1)
        else:
            activation = conved + self.b.dimshuffle(('x', 0) + ('x',) * self.n)

        return self.nonlinearity(activation)

    def convolve(self, input, **kwargs):
        """
        Symbolically convolves `input` with ``self.W``, producing an output of
        shape ``self.output_shape``. To be implemented by subclasses.

        Parameters
        ----------
        input : Theano tensor
            The input minibatch to convolve
        **kwargs
            Any additional keyword arguments from :meth:`get_output_for`

        Returns
        -------
        Theano tensor
            `input` convolved according to the configuration of this layer,
            without any bias or nonlinearity applied.
        """
        raise NotImplementedError("BaseConvLayer does not implement the "
                                  "convolve() method. You will want to "
                                  "use a subclass such as Conv2DLayer.")


class Conv2DLayerR(BaseConvLayerR):
    """
    lasagne.layers.Conv2DLayer(incoming, num_filters, filter_size,
    stride=(1, 1), pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
    convolution=theano.tensor.nnet.conv2d, **kwargs)

    2D convolutional layer

    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 2-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or a 2-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of two integers allows different symmetric padding
        per dimension.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 4D tensor with shape
        ``(num_filters, num_input_channels, filter_rows, filter_columns)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    flip_filters : bool (default: True)
        Whether to flip the filters before sliding them over the input,
        performing a convolution (this is the default), or not to flip them and
        perform a correlation. Note that for some other convolutional layers in
        Lasagne, flipping incurs an overhead and is disabled by default --
        check the documentation when using learned weights from another layer.

    convolution : callable
        The convolution implementation to use. Usually it should be fine to
        leave this at the default value.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), R=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 convolution=conv2dR, **kwargs):
        super(Conv2DLayerR, self).__init__(incoming, num_filters, filter_size,
                                          stride, pad, untie_biases, W, b,
                                          nonlinearity, flip_filters, n=2,
                                          **kwargs)
        self.convolution = convolution

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        conved = self.convolution(input, self.W, self.R,
                                  self.input_shape, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        return conved




def conv2dR(input,
           filters, Rfilters,
           input_shape=None,
           filter_shape=None,
           border_mode='valid',
           subsample=(1, 1),
           filter_flip=True):
    """This function will build the symbolic graph for convolving a mini-batch of a
    stack of 2D inputs with a set of 2D filters. The implementation is modelled
    after Convolutional Neural Networks (CNN).

    Refer to :func:`nnet.conv2d <theano.tensor.nnet.conv2d>` for a more detailed documentation.
    """
    input = as_tensor_variable(input)
    filters = as_tensor_variable(filters)
    Rfilters = as_tensor_variable(Rfilters)

    conv_op = AbstractConv2dR(imshp=input_shape,
                             kshp=filter_shape,
                             border_mode=border_mode,
                             subsample=subsample,
                             filter_flip=filter_flip)
    return conv_op(input, filters, Rfilters)


class BaseAbstractConv2dR(Op):
    """Base class for AbstractConv

    Define an abstract convolution op that will be replaced with the
    appropriate implementation

    Parameters
    ----------
     imshp: None, tuple/list of len 4 of int or Constant variable
        The shape of the input parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.
        imshp is defined w.r.t the forward conv.

     kshp: None, tuple/list of len 4 of int or Constant variable
        The shape of the filters parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.
        kshp is defined w.r.t the forward conv.

     border_mode: str, int or tuple of two int
        Either of the following:

        ``'valid'``: apply filter wherever it completely overlaps with the
            input. Generates output of shape: input shape - filter shape + 1
        ``'full'``: apply filter wherever it partly overlaps with the input.
            Generates output of shape: input shape + filter shape - 1
        ``'half'``: pad input with a symmetric border of ``filter rows // 2``
            rows and ``filter columns // 2`` columns, then perform a valid
            convolution. For filters with an odd number of rows and columns, this
            leads to the output shape being equal to the input shape.
        ``int``: pad input with a symmetric border of zeros of the given
            width, then perform a valid convolution.
        ``(int1, int2)``: pad input with a symmetric border of ``int1`` rows
            and ``int2`` columns, then perform a valid convolution.

    subsample: tuple of len 2
        Factor by which to subsample the output.
        Also called strides elsewhere.

    filter_flip: bool
        If ``True``, will flip the filter rows and columns
        before sliding them over the input. This operation is normally referred
        to as a convolution, and this is the default. If ``False``, the filters
        are not flipped and the operation is referred to as a
        cross-correlation.

    """
    check_broadcast = False
    __props__ = ('border_mode', 'subsample', 'filter_flip', 'imshp', 'kshp')

    def __init__(self,
                 imshp=None, kshp=None,
                 border_mode="valid", subsample=(1, 1),
                 filter_flip=True):

        if isinstance(border_mode, integer_types):
            border_mode = (border_mode, border_mode)
        if isinstance(border_mode, tuple):
            pad_h, pad_w = map(int, border_mode)
            border_mode = (pad_h, pad_w)
        if border_mode == (0, 0):
            border_mode = 'valid'
        if not ((isinstance(border_mode, tuple) and min(border_mode) >= 0) or
                border_mode in ('valid', 'full', 'half')):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a pair of'
                ' integers'.format(border_mode))

        self.imshp = tuple(imshp) if imshp else (None,) * 4
        for imshp_i in self.imshp:
            if imshp_i is not None:
                # Components of imshp should be constant or ints
                try:
                    get_scalar_constant_value(imshp_i,
                                              only_process_constants=True)
                except NotScalarConstantError:
                    reraise(ValueError,
                            ValueError("imshp should be None or a tuple of "
                                       "constant int values"),
                            sys.exc_info()[2])

        self.kshp = tuple(kshp) if kshp else (None,) * 4


        for kshp_i in self.kshp:
            if kshp_i is not None:
                # Components of kshp should be constant or ints
                try:
                    get_scalar_constant_value(kshp_i,
                                              only_process_constants=True)
                except NotScalarConstantError:
                    reraise(ValueError,
                            ValueError("kshp should be None or a tuple of "
                                       "constant int values"),
                            sys.exc_info()[2])

        self.border_mode = border_mode
        self.filter_flip = filter_flip

        if len(subsample) != 2:
            raise ValueError("subsample must have two elements")
        self.subsample = tuple(subsample)

    def flops(self, inp, outp):
        """ Useful with the hack in profilemode to print the MFlops"""
        # if the output shape is correct, then this gives the correct
        # flops for any direction, sampling, padding, and border mode
        inputs, filters, R = inp
        outputs, = outp
        assert inputs[1] == filters[1]
        # nb mul and add by output pixel
        flops = filters[2] * filters[3] * 2
        # nb flops by output image
        flops *= outputs[2] * outputs[3]
        # nb patch multiplied
        flops *= inputs[1] * filters[0] * inputs[0]
        return flops

    def do_constant_folding(self, node):
        # Disable constant folding since there is no implementation.
        # This may change in the future.
        return False

    def conv2d(self, img, kern, mode="valid"):
        """
        Basic slow python implementatation for DebugMode
        """

        if not imported_scipy_signal:
            raise NotImplementedError(
                "AbstractConv perform requires the python package"
                " for scipy.signal to be installed.")
        if not (mode in ('valid', 'full')):
            raise ValueError(
                'invalid mode {}, which must be either '
                '"valid" or "full"'.format(mode))

        out_shape = get_conv_output_shape(img.shape, kern.shape, mode, [1, 1])
        out = numpy.zeros(out_shape, dtype=img.dtype)
        val = _valfrommode(mode)
        bval = _bvalfromboundary('fill')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            for b in xrange(img.shape[0]):
                for n in xrange(kern.shape[0]):
                    for im0 in xrange(img.shape[1]):
                        # some cast generates a warning here
                        out[b, n, ...] += _convolve2d(img[b, im0, ...],
                                                      kern[n, im0, ...],
                                                      1, val, bval, 0)
        return out


class AbstractConv2dR(BaseAbstractConv2dR):
    """ Abstract Op for the forward convolution.
    Refer to :func:`BaseAbstractConv2d <theano.tensor.nnet.abstract_conv.BaseAbstractConv2d>`
    for a more detailed documentation.
    """

    def __init__(self,
                 imshp=None,
                 kshp=None, Rshp=None,
                 border_mode="valid",
                 subsample=(1, 1),
                 filter_flip=True):
        super(AbstractConv2dR, self).__init__(imshp, kshp,
                                             border_mode, subsample,
                                             filter_flip)

    def make_node(self, img, kern, R):

        # Make sure both inputs are Variables with the same Type
        if not isinstance(img, theano.Variable):
            img = as_tensor_variable(img)
        if not isinstance(kern, theano.Variable):
            kern = as_tensor_variable(kern)
        ktype = img.type.clone(dtype=kern.dtype,
                               broadcastable=kern.broadcastable)
        kern = ktype.filter_variable(kern)
        R = ktype.filter_variable(R)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')

        broadcastable = [img.broadcastable[0],
                         kern.broadcastable[0],
                         False, False]
        output = img.type.clone(broadcastable=broadcastable)()
        return Apply(self, [img, kern, R], [output])

    def perform(self, node, inp, out_):
        img, kern, R = inp
        img = numpy.asarray(img)
        kern = numpy.asarray(kern)
        o, = out_
        mode = self.border_mode

        if not ((isinstance(mode, tuple) and min(mode) >= 0) or
                mode in ('valid', 'full', 'half')):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a pair of'
                ' integers'.format(mode))

        if mode == "full":
            mode = (kern.shape[2] - 1, kern.shape[3] - 1)
        elif mode == "half":
            mode = (kern.shape[2] // 2, kern.shape[3] // 2)
        if isinstance(mode, tuple):
            pad_h, pad_w = map(int, mode)
            mode = "valid"
            new_img = numpy.zeros((img.shape[0], img.shape[1],
                                   img.shape[2] + 2 * pad_h,
                                   img.shape[3] + 2 * pad_w), dtype=img.dtype)
            new_img[:, :, pad_h:img.shape[2] + pad_h, pad_w:img.shape[3] + pad_w] = img
            img = new_img
        if not self.filter_flip:
            kern = kern[:, :, ::-1, ::-1]
        conv_out = self.conv2d(img, kern, mode="valid")
        conv_out = conv_out[:, :, ::self.subsample[0], ::self.subsample[1]]

        o[0] = node.outputs[0].type.filter(conv_out)

    def R_op(self, inputs, eval_points):
        rval = None
        if eval_points[0] is not None:
            rval = self.make_node(eval_points[0], inputs[1], inputs[2]).outputs[0]
        if eval_points[1] is not None:
            if rval is None:
                rval = self.make_node(inputs[0], eval_points[1], inputs[2]).outputs[0]
            else:
                rval += self.make_node(inputs[0], eval_points[1], inputs[2]).outputs[0]
        return [rval]

    def grad(self, inp, grads):
        bottom, weights, R = inp
        top, = grads
        d_bottom = AbstractConv2d_gradInputsR(self.imshp, self.kshp,
                                             self.border_mode,
                                             self.subsample,
                                             self.filter_flip)(
            R, top, bottom.shape[-2:])
        d_weights = AbstractConv2d_gradWeightsR(self.imshp, self.kshp,
                                               self.border_mode,
                                               self.subsample,
                                               self.filter_flip)(

            bottom, top, weights.shape[-2:])

        # Make sure that the broadcastable pattern of the inputs is used
        # for the gradients, even if the grad opts are not able to infer
        # that the dimensions are broadcastable.
        # Also make sure that the gradient lives on the same device than
        # the corresponding input.
        d_bottom = patternbroadcast(d_bottom, bottom.broadcastable)
        d_bottom = bottom.type.filter_variable(d_bottom)
        d_weights = patternbroadcast(d_weights, weights.broadcastable)
        d_weights = weights.type.filter_variable(d_weights)
        return d_bottom, d_weights

    def infer_shape(self, node, input_shapes):
        imshp = input_shapes[0]
        kshp = input_shapes[1]

        # replace symbolic shapes with known constant shapes
        if self.imshp is not None:
            imshp = [imshp[i] if self.imshp[i] is None else self.imshp[i]
                     for i in range(4)]
        if self.kshp is not None:
            kshp = [kshp[i] if self.kshp[i] is None else self.kshp[i]
                    for i in range(4)]
        res = get_conv_output_shape(imshp, kshp, self.border_mode,
                                    self.subsample)
        return [res]


class AbstractConv2d_gradWeightsR(BaseAbstractConv2d):
    """Gradient wrt. filters for `AbstractConv2d`.
    Refer to :func:`BaseAbstractConv2d <theano.tensor.nnet.abstract_conv.BaseAbstractConv2d>`
    for a more detailed documentation.

    :note: You will not want to use this directly, but rely on
           Theano's automatic differentiation or graph optimization to
           use it as needed.

    """
    def __init__(self,
                 imshp=None,
                 kshp=None,
                 Rshp=None,
                 border_mode="valid",
                 subsample=(1, 1),
                 filter_flip=True):
        super(AbstractConv2d_gradWeightsR, self).__init__(imshp, kshp,
                                                         border_mode,
                                                         subsample,
                                                         filter_flip)

    # Update shape/height_width
    def make_node(self, img, topgrad, shape):
        # Make sure both inputs are Variables with the same Type
        if not isinstance(img, theano.Variable):
            img = as_tensor_variable(img)
        if not isinstance(topgrad, theano.Variable):
            topgrad = as_tensor_variable(topgrad)
        gtype = img.type.clone(dtype=topgrad.dtype,
                               broadcastable=topgrad.broadcastable)
        topgrad = gtype.filter_variable(topgrad)

        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')

        shape = as_tensor_variable(shape)
        broadcastable = [topgrad.broadcastable[1],
                         img.broadcastable[1],
                         False, False]
        output = img.type.clone(broadcastable=broadcastable)()
        return Apply(self, [img, topgrad, shape], [output])

    def perform(self, node, inp, out_):
        img, topgrad, shape = inp
        img = numpy.asarray(img)
        topgrad = numpy.asarray(topgrad)

        o, = out_

        mode = self.border_mode
        if not ((isinstance(mode, tuple) and min(mode) >= 0) or
                mode in ('valid', 'full', 'half')):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a pair of'
                ' integers'.format(mode))

        if mode == "full":
            mode = (shape[0] - 1, shape[1] - 1)
        elif mode == "half":
            mode = (shape[0] // 2, shape[1] // 2)
        if isinstance(mode, tuple):
            pad_h, pad_w = map(int, mode)
            mode = "valid"
            new_img = numpy.zeros((img.shape[0], img.shape[1],
                                   img.shape[2] + 2 * pad_h,
                                   img.shape[3] + 2 * pad_w), dtype=img.dtype)
            new_img[:, :, pad_h:img.shape[2] + pad_h, pad_w:img.shape[3] + pad_w] = img
            img = new_img

        if self.subsample[0] > 1 or self.subsample[1] > 1:
            new_shape = (topgrad.shape[0], topgrad.shape[1],
                         img.shape[2] - shape[0] + 1,
                         img.shape[3] - shape[1] + 1)
            new_topgrad = numpy.zeros((new_shape), dtype=topgrad.dtype)
            new_topgrad[:, :, ::self.subsample[0], ::self.subsample[1]] = topgrad
            topgrad = new_topgrad

        topgrad = topgrad.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1]
        img = img.transpose(1, 0, 2, 3)
        kern = self.conv2d(img, topgrad, mode="valid")
        if self.filter_flip:
            kern = kern.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1]
        else:
            kern = kern.transpose(1, 0, 2, 3)
        o[0] = node.outputs[0].type.filter(kern)

    # def grad(self, inp, grads):
    #     bottom, top = inp[:2]
    #     weights, = grads
    #     d_bottom = AbstractConv2d_gradInputsR(self.imshp, self.kshp,
    #                                          self.border_mode,
    #                                          self.subsample,
    #                                          self.filter_flip)(
    #                                              weights,
    #                                              top,
    #                                              bottom.shape[-2:])
    #     d_top = AbstractConv2dR(self.imshp,
    #                            self.kshp,
    #                            self.border_mode,
    #                            self.subsample,
    #                            self.filter_flip)(bottom, weights)
    #     # Make sure that the broadcastable pattern of the inputs is used
    #     # for the gradients, even if the grad opts are not able to infer
    #     # that the dimensions are broadcastable.
    #     # Also make sure that the gradient lives on the same device than
    #     # the corresponding input.
    #     d_bottom = patternbroadcast(d_bottom, bottom.broadcastable)
    #     d_bottom = bottom.type.filter_variable(d_bottom)
    #     d_top = patternbroadcast(d_top, top.broadcastable)
    #     d_top = top.type.filter_variable(d_top)
    #
    #     d_height_width = (theano.gradient.DisconnectedType()(),)
    #     return (d_bottom, d_top) + d_height_width

    def connection_pattern(self, node):
        return [[1], [1], [0]]  # no connection to height, width

    def infer_shape(self, node, input_shapes):
        # We use self.kshp (that was passed when creating the Op) if possible,
        # or fall back to the `shape` input of the node.
        # TODO: when there is no subsampling, try to infer the kernel shape
        # from the shapes of inputs.
        imshp = input_shapes[0]
        topshp = input_shapes[1]
        kshp = self.kshp[:] if self.kshp is not None else [None] * 4
        fallback_kshp = [topshp[1], imshp[1], node.inputs[2][0], node.inputs[2][1]]
        kshp = [fallback_kshp[i] if kshp[i] is None else kshp[i]
                for i in range(4)]
        return [kshp]


class AbstractConv2d_gradInputsR(BaseAbstractConv2dR):
    """Gradient wrt. inputs for `AbstractConv2d`.
    Refer to :func:`BaseAbstractConv2d <theano.tensor.nnet.abstract_conv.BaseAbstractConv2d>`
    for a more detailed documentation.

    :note: You will not want to use this directly, but rely on
           Theano's automatic differentiation or graph optimization to
           use it as needed.

    """

    def __init__(self,
                 imshp=None,
                 kshp=None,
                 border_mode="valid",
                 subsample=(1, 1),
                 filter_flip=True):
        super(AbstractConv2d_gradInputsR, self).__init__(imshp, kshp,
                                                        border_mode,
                                                        subsample,
                                                        filter_flip)

    # Update shape/height_width
    def make_node(self, kern, R, topgrad, shape):
        # Make sure both inputs are Variables with the same Type
        if not isinstance(kern, theano.Variable):
            kern = as_tensor_variable(kern)
        if not isinstance(topgrad, theano.Variable):
            topgrad = as_tensor_variable(topgrad)
        gtype = kern.type.clone(dtype=topgrad.dtype,
                                broadcastable=topgrad.broadcastable)
        topgrad = gtype.filter_variable(topgrad)

        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')

        shape = as_tensor_variable(shape)
        broadcastable = [topgrad.type.broadcastable[0],
                         kern.type.broadcastable[1],
                         False, False]
        output = kern.type.clone(broadcastable=broadcastable)()
        return Apply(self, [kern, topgrad, shape], [output])

    def perform(self, node, inp, out_):
        okern, kern, topgrad, shape = inp
        kern = numpy.asarray(kern)
        topgrad = numpy.asarray(topgrad)
        o, = out_

        mode = self.border_mode
        if not ((isinstance(mode, tuple) and min(mode) >= 0) or
                mode in ('valid', 'full', 'half')):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a pair of'
                ' integers'.format(mode))

        pad_h, pad_w = 0, 0
        if mode == "full":
            pad_h, pad_w = (kern.shape[2] - 1, kern.shape[3] - 1)
        elif mode == "half":
            pad_h, pad_w = (kern.shape[2] // 2, kern.shape[3] // 2)
        elif isinstance(mode, tuple):
            pad_h, pad_w = map(int, self.border_mode)
        if self.subsample[0] > 1 or self.subsample[1] > 1:
            new_shape = (topgrad.shape[0], topgrad.shape[1],
                         shape[0] + 2 * pad_h - kern.shape[2] + 1,
                         shape[1] + 2 * pad_w - kern.shape[3] + 1)
            new_topgrad = numpy.zeros((new_shape), dtype=topgrad.dtype)
            new_topgrad[:, :, ::self.subsample[0], ::self.subsample[1]] = topgrad
            topgrad = new_topgrad
        kern = kern.transpose(1, 0, 2, 3)
        if self.filter_flip:
            topgrad = topgrad[:, :, ::-1, ::-1]
        img = self.conv2d(topgrad, kern, mode="full")
        if self.filter_flip:
            img = img[:, :, ::-1, ::-1]
        if pad_h > 0 or pad_w > 0:
            img = img[:, :, pad_h:img.shape[2] - pad_h, pad_w:img.shape[3] - pad_w]
        o[0] = node.outputs[0].type.filter(img)


    def connection_pattern(self, node):
        return [[1], [1], [0]]  # no connection to height, width

    def infer_shape(self, node, input_shapes):
        # We use self.imshp (that was passed when creating the Op) if possible,
        # or fall back to the `shape` input of the node.
        # TODO: when there is no subsampling, try to infer the image shape
        # from the shapes of inputs.
        kshp = input_shapes[0]
        topshp = input_shapes[1]
        imshp = self.imshp[:] if self.imshp is not None else [None] * 4
        fallback_imshp = [topshp[0], kshp[1], node.inputs[2][0],
                          node.inputs[2][1]]
        imshp = [fallback_imshp[i] if imshp[i] is None else imshp[i]
                 for i in range(4)]
        return [imshp]
