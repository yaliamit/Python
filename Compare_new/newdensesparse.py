import numpy as np
import theano.tensor as T

import lasagne
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers.base import Layer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano
from theano import printing
import theano.sparse as sparse
import sparse_new
import scipy.sparse as sp
__all__ = [
    "NewDenseLayer",
]

class SparseDenseLayer(Layer):
    """
    lasagne.layers.DenseLayer(incoming, num_units,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)

    A fully connected layer.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of units of the layer

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, DenseLayer
    >>> l_in = InputLayer((100, 20))
    >>> l1 = DenseLayer(l_in, num_units=50)

    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """
    def __init__(self, incoming, num_units, W=None, R=None,
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(SparseDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))
        if (W is None):
            W=theano.shared(sp.csc_matrix(np.float32(np.eye(num_inputs, num_units))))
        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if (R is None):
            R=theano.shared(sp.csc_matrix(np.float32(np.eye(num_inputs, num_units))))
        self.R = self.add_param(R, (num_inputs, num_units), name="R")#, trainable=False)


        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        activation = sparse_new.new_structured_dot(input, self.W, self.R)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


