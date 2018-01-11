__author__ = 'amit'

import numpy
import theano.tensor as T
import theano
import theano.tensor.basic
import theano.gradient
from theano.tensor.type_other import NoneConst
from theano import scalar as scal
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams



class NewDotOp(theano.Op):
    __props__ = ()


    def make_node(self, *inputs):
        inputs = list(map(theano.tensor.basic.as_tensor_variable, inputs))
        i_broadcastables = [input.type.broadcastable for input in inputs]
        bx, by, br, bp, byz, brz, brd = i_broadcastables
        if len(by) == 2:  # y is a matrix
            bz = bx[:-1] + by[-1:]
        elif len(by) == 1:  # y is vector
            bz = bx[:-1]

        self.srng = RandomStreams(numpy.random.randint(1,high=2147462579))
        self.prob=inputs[3]

        i_dtypes = [input.type.dtype for input in inputs[0:3]]
        outputs = [theano.tensor.basic.tensor(scal.upcast(*i_dtypes), bz)]
        return theano.Apply(self, inputs[0:3]+inputs[4:7], outputs)
        #return theano.Apply(self, inputs, outputs)

    def perform(self, node, inp, out):
        x, y, R, Wzer, Rzer, OD = inp
        u, = out

        # the asarray is here because dot between two vectors
        # gives a numpy float object but we need to return a 0d
        # ndarray
        u[0] = numpy.asarray(numpy.dot(x, y))

    def grad(self, inp, grads):

        x, y, R, Wzer, Rzer, OD = inp
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
            if (Rzer.data.shape[0] == 1):
                xgrad = T.dot(gz, y.T)
            else:
                xgrad = T.dot(gz, R.T)
                if (OD.data[0]>1):
                    xx=T.reshape(xgrad,tuple(([0],)+OD))
                    xxs=T.nnet.conv2d(xx,T.ones((OD.data[1],5,5)))
                    xgrad=xxs
            yygrad=T.dot(x.T,gz)
            zzgrad=yygrad

            ygrad=yygrad*Wzer
            if (Rzer.data.shape[0] == 1):
                zgrad=theano.gradient.grad_undefined(self,2,R)
            else:
                #zzgrad=T.dot(x.T,T.nnet.relu(gz))
                zgrad=zzgrad*Rzer


        #zgrad=T.zeros(T.shape(R))
        # If x or y contain broadcastable dimensions but only one of
        # them know that a matching dimensions is broadcastable, the
        # above code don't always return the right broadcast pattern.
        # This cause problem down the road. See gh-1461.
        if xgrad.broadcastable != x.broadcastable:
            xgrad = theano.tensor.basic.patternbroadcast(xgrad, x.broadcastable)
        if ygrad.broadcastable != y.broadcastable:
            ygrad = theano.tensor.basic.patternbroadcast(ygrad, y.broadcastable)
        if (Rzer.data.shape[0]!=1):
            if zgrad.broadcastable != R.broadcastable:
                zgrad = theano.tensor.basic.patternbroadcast(zgrad, R.broadcastable)
        Wz=theano.gradient.grad_undefined(self,3,Wzer)
        Rz=theano.gradient.grad_undefined(self,4,Rzer)
        ODz=theano.gradient.grad_undefined(self,5,OD)
        return xgrad, ygrad, zgrad, Wz, Rz, ODz


newdot = NewDotOp()

#Using itypes and otypes


