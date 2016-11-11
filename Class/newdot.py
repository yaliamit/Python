__author__ = 'amit'

import numpy
import theano
import theano.tensor as T

class NewDotOp(theano.Op):
    __props__ = ()


    def make_node(self, *inputs):
        inputs = list(map(theano.as_tensor_variable, inputs))

        if len(inputs) != 3:
            raise TypeError(
                'AffineOP: 3 arguments required, %d given ' %
                len(inputs))

        i_broadcastables = [input.type.broadcastable for input in inputs]
        bx, by = i_broadcastables
        # if len(by) == 2:  # y is a matrix

        # elif len(by) == 1:  # y is vector
        #     bz = bx[:-1]

        i_dtypes = [input.type.dtype for input in inputs]
        outputs = [theano.tensor(theano.scal.upcast(*i_dtypes), bz)]
        return theano.Apply(self, inputs, outputs)

    def perform(self, node, inp, out):
        x, y = inp
        u, = out

        # the asarray is here because dot between two vectors
        # gives a numpy float object but we need to return a 0d
        # ndarray
        u[0] = numpy.asarray(numpy.dot(x, y)+z)

    def grad(self, inp, grads):

        x, y = inp
        gz, = grads
        xdim, ydim, gdim = x.type.ndim, y.type.ndim, gz.type.ndim

        R=np.random.normal(0,.1,x.shape)
        # # grad is scalar, so x is vector and y is vector
        # if gdim == 0:
        #     xgrad = gz * y
        #     #ygrad = gz * x
        #
        # # x is vector, y is matrix, grad is vector
        # elif xdim == 1 and ydim == 2:
        #     xgrad = dot(gz, y.T)
        #     ygrad = outer(x.T, gz)

        # # x is matrix, y is vector, grad is vector
        if xdim == 2 and ydim == 1:
            xgrad = T.outer(gz, y.T)
            #ygrad = T.dot(x.T, gz)
            ygrad=T.dot(R,gz)


        # # x is matrix, y is matrix, grad is matrix
        # elif xdim == ydim == 2:
        #     xgrad = T.dot(gz, y.T)
        #     ygrad = T.dot(x.T, gz)


        # If x or y contain broadcastable dimensions but only one of
        # them know that a matching dimensions is broadcastable, the
        # above code don't always return the right broadcast pattern.
        # This cause problem down the road. See gh-1461.
        if xgrad.broadcastable != x.broadcastable:
            xgrad = patternbroadcast(xgrad, x.broadcastable)
        if ygrad.broadcastable != y.broadcastable:
            ygrad = patternbroadcast(ygrad, y.broadcastable)

        rval = xgrad, ygrad

        for elem in rval:
            assert elem.dtype.find('float') != -1

        return rval


newmulOp = NewMulOp()

#Using itypes and otypes


