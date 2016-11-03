
from theano.tensor.nnet.abstract_conv import BaseAbstractConv2d, get_conv_output_shape
import theano
import theano.tensor as T
from theano.tensor import as_tensor_variable, patternbroadcast
import numpy
from theano.gof import Apply


try:
    from scipy.signal.signaltools import _valfrommode, _bvalfromboundary
    from scipy.signal.sigtools import _convolve2d
    imported_scipy_signal = True
except ImportError:
    imported_scipy_signal = False



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




class AbstractConv2dR(BaseAbstractConv2d):
    """ Abstract Op for the forward convolution.
    Refer to :func:`BaseAbstractConv2d <theano.tensor.nnet.abstract_conv.BaseAbstractConv2d>`
    for a more detailed documentation.
    """

    def __init__(self,
                 imshp=None,
                 kshp=None,
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

        d_R=d_weights
        # Make sure that the broadcastable pattern of the inputs is used
        # for the gradients, even if the grad opts are not able to infer
        # that the dimensions are broadcastable.
        # Also make sure that the gradient lives on the same device than
        # the corresponding input.
        d_bottom = patternbroadcast(d_bottom, bottom.broadcastable)
        d_bottom = bottom.type.filter_variable(d_bottom)
        d_weights = patternbroadcast(d_weights, weights.broadcastable)
        d_weights = weights.type.filter_variable(d_weights)
        d_R = patternbroadcast(d_R, R.broadcastable)
        d_R = R.type.filter_variable(d_R)
        return d_bottom, d_weights, d_R

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


class AbstractConv2d_gradInputsR(BaseAbstractConv2d):
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
    def make_node(self, kern, topgrad, shape):
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
