import numpy as np
import theano.tensor as T

import lasagne
from lasagne import init
from lasagne import nonlinearities
import newdot
from lasagne.layers.base import Layer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano import printing
import theano
__all__ = [
    "NewDenseLayer",
]

class NewDenseLayer(Layer):
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
    def __init__(self, incoming, num_units, W=init.GlorotUniform(), R=init.GlorotUniform(),
                 Wzero=init.Uniform(range=(0.,1.)),Rzero=init.Uniform(range=(0.,1.)),
                 b=init.Constant(0.), prob=(.5,.5),
                 nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(NewDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        self.R = self.add_param(R, (num_inputs, num_units), name="R")#, trainable=False)
        self.prob = prob
        self.Wzero=self.add_param(Wzero,(num_inputs,num_units),name="Wzero", trainable=False)
        self.Rzero=self.add_param(Rzero,(num_inputs,num_units),name="Rzero", trainable=False)
        self.Wzero=(self.Wzero<self.prob[0] or self.W==0)
        self.Rzero=(self.Rzero<self.prob[0] or self.R==0)
        self.W=self.W*self.Wzero
        self.R=self.R*self.Rzero
        # self.prob[1]=0 no gradient on R
        if (self.prob[1]==0.):
            self.Rzero=self.Rzero<0

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
        activation = newdot.newdot(input, self.W,self.R, self.prob, self.Wzero, self.Rzero)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


