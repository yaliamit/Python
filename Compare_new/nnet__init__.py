from .nnet import (
    CrossentropyCategorical1Hot, CrossentropyCategorical1HotGrad,
    CrossentropySoftmax1HotWithBiasDx, CrossentropySoftmaxArgmax1HotWithBias,
    LogSoftmax, Prepend_scalar_constant_to_each_row,
    Prepend_scalar_to_each_row, Softmax,
    SoftmaxGrad, SoftmaxWithBias, binary_crossentropy,
    categorical_crossentropy, crossentropy_categorical_1hot,
    crossentropy_categorical_1hot_grad, crossentropy_softmax_1hot,
    crossentropy_softmax_1hot_with_bias,
    crossentropy_softmax_1hot_with_bias_dx,
    crossentropy_softmax_argmax_1hot_with_bias,
    crossentropy_softmax_max_and_argmax_1hot,
    crossentropy_softmax_max_and_argmax_1hot_with_bias,
    crossentropy_to_crossentropy_with_softmax,
    crossentropy_to_crossentropy_with_softmax_with_bias,
    graph_merge_softmax_with_crossentropy_softmax, h_softmax,
    logsoftmax, logsoftmax_op, prepend_0_to_each_row, prepend_1_to_each_row,
    prepend_scalar_to_each_row, relu, softmax, softmax_grad, softmax_graph,
    softmax_op, softmax_simplifier, softmax_with_bias, elu)
from . import opt
from .conv import ConvOp
from .Conv3D import *
from .ConvGrad3D import *
from .ConvTransp3D import *
from .sigm import (softplus, sigmoid, sigmoid_inplace,
                   scalar_sigmoid, ultra_fast_sigmoid,
                   hard_sigmoid)
from .bn import batch_normalization


import warnings
from .abstract_conv import conv2dR as abstract_conv2dR
from .abstract_conv import conv2d as abstract_conv2d

#from Compare_new.abstract_conv import conv2dR as abstract_conv2dR
# from .abstract_conv import conv2d as abstract_conv2d

def conv2dR(input,
           filters, Rfilters, Wzer,
            #Rzer,
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


    return abstract_conv2dR(input, filters, Rfilters, Wzer,
                            #Rzer,
                            input_shape, filter_shape,
                           border_mode, subsample, filter_flip)





def conv2d(input, filters, input_shape=None, filter_shape=None,
           border_mode='valid', subsample=(1, 1), filter_flip=True,
           image_shape=None, **kwargs):
    """
    This function will build the symbolic graph for convolving a mini-batch of a
    stack of 2D inputs with a set of 2D filters. The implementation is modelled
    after Convolutional Neural Networks (CNN).


    Parameters
    ----------
    input: symbolic 4D tensor
        Mini-batch of feature map stacks, of shape
        (batch size, input channels, input rows, input columns).
        See the optional parameter ``input_shape``.

    filters: symbolic 4D tensor
        Set of filters used in CNN layer of shape
        (output channels, input channels, filter rows, filter columns).
        See the optional parameter ``filter_shape``.

    input_shape: None, tuple/list of len 4 of int or Constant variable
        The shape of the input parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.

    filter_shape: None, tuple/list of len 4 of int or Constant variable
        The shape of the filters parameter.
        Optional, possibly used to choose an optimal implementation.
        You can give ``None`` for any element of the list to specify that this
        element is not known at compile time.

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
        are not flipped and the operation is referred to as a cross-correlation.

    image_shape: None, tuple/list of len 4 of int or Constant variable
        Deprecated alias for input_shape.

    kwargs: Any other keyword arguments are accepted for backwards
            compatibility, but will be ignored.

    Returns
    -------
    Symbolic 4D tensor
        Set of feature maps generated by convolutional layer. Tensor is
        of shape (batch size, output channels, output rows, output columns)

    Notes
    -----
        If cuDNN is available, it will be used on the
        GPU. Otherwise, it is the *CorrMM* convolution that will be used
        "caffe style convolution".

        This is only supported in Theano 0.8 or the development
        version until it is released.

    """

    if 'imshp_logical' in kwargs or 'kshp_logical' in kwargs:
        raise ValueError(
            "Keyword arguments 'imshp_logical' and 'kshp_logical' for conv2d "
            "are not supported anymore (and have not been a reliable way to "
            "perform upsampling). That feature is still available by calling "
            "theano.tensor.nnet.conv.conv2d() for the time being.")
    if len(kwargs.keys()) > 0:
        warnings.warn(str(kwargs.keys()) +
                      " are now deprecated in "
                      "`tensor.nnet.abstract_conv.conv2d` interface"
                      " and will be ignored.",
                      stacklevel=2)

    if image_shape is not None:
        warnings.warn("The `image_shape` keyword argument to "
                      "`tensor.nnet.conv2d` is deprecated, it has been "
                      "renamed to `input_shape`.",
                      stacklevel=2)
        if input_shape is None:
            input_shape = image_shape
        else:
            raise ValueError("input_shape and image_shape should not"
                             " be provided at the same time.")

    return abstract_conv2d(input, filters, input_shape, filter_shape,
                           border_mode, subsample, filter_flip)
