__author__ = 'amit'

import numpy
import theano
import theano.tensor as T
import theano
import theano.tensor.basic
import theano.gradient
from theano.compat import izip
from theano.configparser import config
from theano import gof
from theano.gof import Apply, Constant, Op, Variable

from theano.tensor import elemwise
from theano.tensor.var import (AsTensorError, TensorVariable,
                               TensorConstant,
                               _tensor_py_operators)
from theano.tensor.type import TensorType, values_eq_approx_always_true
from theano.tensor.type_other import NoneConst
from theano import scalar as scal
from functools import partial
from theano import compile, printing
from theano.printing import pprint, min_informative_str
# For history
from theano.compile import Rebroadcast, Shape, shape


class NewDotOp(theano.Op):
    __props__ = ()


    def make_node(self, *inputs):
        inputs = list(map(theano.tensor.basic.as_tensor_variable, inputs))
        if len(inputs) != 3:
            raise TypeError(
                'AffineOP: 3 arguments required, %d given ' %
                len(inputs))

        i_broadcastables = [input.type.broadcastable for input in inputs]
        bx, by, br = i_broadcastables
        if len(by) == 2:  # y is a matrix
            bz = bx[:-1] + by[-1:]
        elif len(by) == 1:  # y is vector
            bz = bx[:-1]
        # if len(by) == 2:  # y is a matrix

        # elif len(by) == 1:  # y is vector
        #bz = bx[:-1]

        i_dtypes = [input.type.dtype for input in inputs]
        outputs = [theano.tensor.basic.tensor(scal.upcast(*i_dtypes), bz)]
        return theano.Apply(self, inputs, outputs)

    def perform(self, node, inp, out):
        x, y, R = inp
        u, = out

        # the asarray is here because dot between two vectors
        # gives a numpy float object but we need to return a 0d
        # ndarray
        u[0] = numpy.asarray(numpy.dot(x, y))

    def grad(self, inp, grads):

        x, y, R = inp
        gz, = grads
        xdim, ydim, gdim = x.type.ndim, y.type.ndim, gz.type.ndim


        # grad is scalar, so x is vector and y is vector
        if gdim == 0:
            xgrad = gz * y
            ygrad = gz * x

        # x is vector, y is matrix, grad is vector
        elif xdim == 1 and ydim == 2:
            #xgrad = T.dot(gz, y.T)
            xgrad = T.dot(gz, R.T)
            ygrad = T.outer(x.T, gz)

        # # x is matrix, y is vector, grad is vector
        if xdim == 2 and ydim == 1:
            #xgrad = T.outer(gz, y.T)
            xgrad = T.outer(gz, R.T)
            ygrad = T.dot(x.T, gz)



        # x is matrix, y is matrix, grad is matrix
        elif xdim == ydim == 2:
            #xgrad = T.dot(gz, y.T)
            xgrad = T.dot(gz, R.T)
            ygrad = T.dot(x.T,gz)

        zgrad=T.zeros(T.shape(R))
        # If x or y contain broadcastable dimensions but only one of
        # them know that a matching dimensions is broadcastable, the
        # above code don't always return the right broadcast pattern.
        # This cause problem down the road. See gh-1461.
        if xgrad.broadcastable != x.broadcastable:
            xgrad = theano.tensor.basic.patternbroadcast(xgrad, x.broadcastable)
        if ygrad.broadcastable != y.broadcastable:
            ygrad = theano.tensor.basic.patternbroadcast(ygrad, y.broadcastable)

        rval = xgrad, ygrad, zgrad

        for elem in rval:
            assert elem.dtype.find('float') != -1

        return rval


newdot = NewDotOp()

#Using itypes and otypes


