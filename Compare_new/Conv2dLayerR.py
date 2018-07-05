
from lasagne import init
from lasagne import nonlinearities
from lasagne.utils import as_tuple
from lasagne.layers.conv import conv_output_length
from lasagne.layers.base import Layer
import theano.tensor as T
import numpy as np

try:
    from scipy.signal.signaltools import _valfrommode, _bvalfromboundary
    from scipy.signal.sigtools import _convolve2d
    imported_scipy_signal = True
except ImportError:
    imported_scipy_signal = False





__all__ = [
    "Conv2DLayerR",
]


class BaseConvLayerR(Layer):
    """

    """
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 untie_biases=False,
                 W=init.GlorotUniform(),
                 R=init.GlorotUniform(),
                 Wzer=init.Uniform(range=(0.,1.)),
                 Rzer=init.Uniform(range=(0.,1.)),
                 prob=np.array((.5,.5)),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 n=None, **kwargs):
        super(BaseConvLayerR, self).__init__(incoming, **kwargs)
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

        self.prob=prob

        self.Wzer=self.add_param(Wzer,self.get_W_shape(),name="Wzero", trainable=False)
        self.Wzer=self.Wzer<self.prob[0]
        #WZ=T.copy(self.Wzer)
        self.W=self.W*self.Wzer
        if (self.prob[1]<0):
            self.R=np.zeros((2,2))
            #self.Wzer=T.zeros((2,2))
            self.Rzer=T.zeros((2,2))
            self.R = self.add_param(R, (2,2), name="R", trainable=False)
            self.Rzer=self.add_param(Rzer,(2,2),name="Rzero", trainable=False)
        else:
            self.R = self.add_param(R, self.get_W_shape(), name="R")
            self.Rzer=self.add_param(Rzer,self.get_W_shape(),name="Rzero", trainable=False)
        p=.8
        self.Rzer=self.Rzer<self.prob[0]
        #RZ=self.Wzer*(self.Rzer<p)+(1-self.Wzer)*(self.Rzer<(1-p))

        #self.Rzer=RZ
        self.R=self.R*self.Rzer
        #self.prob[1]=0 no gradient on R

        if (self.prob[1]==0.):
            self.Rzer=self.Rzer<0



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


    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(),
                 R=init.GlorotUniform(),
                 Wzer=init.Uniform(range=(0.,1.)),Rzer=init.Uniform(range=(0.,1.)),
                 b=init.Constant(0.), prob=np.array((.5,.5)),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 convolution=T.nnet.conv2dR, **kwargs):
        super(Conv2DLayerR, self).__init__(incoming, num_filters, filter_size,
                                          stride, pad, untie_biases, W,
                                           R, Wzer, Rzer, prob,
                                           b,
                                          nonlinearity, flip_filters, n=2,
                                          **kwargs)
        self.convolution = convolution

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        conved = self.convolution(input, self.W,
                                  self.R, self.Wzer, self.Rzer,
                                  self.input_shape, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        return conved


