import sys

import theano
import theano.tensor as T
import lasagne
import numpy as np
import time
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.compile.nanguardmode import NanGuardMode


def build_model(input_var=None, input_shape=(None, ), n_latent=2,
                n_hidden=128, n_layers=1, batch_size=100):
    print("Building model ...")
    network, (z_mu_net, z_ls_net), z_net, trans_param_net = build_transf_var_autoencoder(
        input_var, input_shape, n_latent=n_latent, n_hidden=n_hidden,
        n_layers=n_layers)

    def loss_fn(y_pred, y_true, z_mu, z_ls, val=False):
            eps = 1e-6
            y_pred = y_pred.clip(eps, 1 - eps)
            kl_div = 1 + 2 * z_ls - T.sqr(z_mu) - T.exp(2 * z_ls)
            kl_div = T.sum(kl_div) / batch_size
            logpxz = lasagne.objectives.binary_crossentropy(y_pred, y_true)
            logpxz = logpxz.sum() / batch_size
            loss = logpxz - 0.5 * kl_div
            if val:
                return loss, logpxz
            return loss

    outputs = lasagne.layers.get_output([network, z_mu_net, z_ls_net])
    prediction = outputs[0]
    z_mu = outputs[1]
    z_ls = outputs[2]
    output_var = T.tensor4('outputs')
    loss = loss_fn(prediction, output_var, z_mu, z_ls)

    params = lasagne.layers.get_all_params(network, trainable=True)
    grad = T.grad(loss, params)
    updates = lasagne.updates.adadelta(
                grad, params)
    outputs = lasagne.layers.get_output(
        [network, z_mu_net, z_ls_net, z_net, trans_param_net], deterministic=True)
    test_prediction = outputs[0]
    z_mu = outputs[1]
    z_ls = outputs[2]
    z = outputs[3]
    trans_param = outputs[4]

    test_loss = loss_fn(test_prediction, output_var, z_mu, z_ls, True)

    print("Compiling functions ...")
    train_fn = theano.function([input_var, output_var], loss, updates=updates)
    val_fn = theano.function([input_var, output_var], test_loss)
    predict_fn = theano.function([input_var], test_prediction)
    z_vect = T.matrix()
    trans_par = T.matrix()
    generated_x = lasagne.layers.get_output(network, {z_net: z_vect,
                                                      trans_param_net: trans_par})
    decode_fn = theano.function([z_vect, trans_par], generated_x)
    encode_fn = theano.function([input_var], z)
    param_fn = theano.function([input_var], trans_param)
    return decode_fn, encode_fn, train_fn, val_fn, predict_fn, param_fn



class GaussianSampleLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng \
            else RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape = (self.input_shapes[0][0] or inputs[0].shape[0],
                 self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.rng.normal(shape)


def build_transf_var_autoencoder(input_var=None, input_shape=(None, ),
                                 n_hidden=128, n_layers=1, n_latent=2):

    x = lasagne.layers.InputLayer(
        shape=(None,) + input_shape,
        input_var=input_var)
    for i in range(n_layers):
        x = lasagne.layers.DenseLayer(
            x, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)

    # x2 = lasagne.layers.DropoutLayer(x2, p=drop_prob)

    transf_param = lasagne.layers.DenseLayer(
        x, num_units=6, nonlinearity=lasagne.nonlinearities.tanh)
    transf_param = lasagne.layers.standardize(
        transf_param, np.array([-1, 0, 0, 0, -1, 0], dtype='float32'),
        np.ones(6, dtype='float32'))

    z_mean = lasagne.layers.DenseLayer(
        x, num_units=n_latent, nonlinearity=None)
    relu_shift = 6  # 10
    z_logsigma = lasagne.layers.DenseLayer(
        x, num_units=n_latent,
        nonlinearity=lambda a: T.nnet.relu(a + relu_shift) - relu_shift)

    z = GaussianSampleLayer(z_mean, z_logsigma)
    x = z

    for i in range(n_layers):
        x = lasagne.layers.DenseLayer(
            x, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)

    x = lasagne.layers.DenseLayer(
        x, num_units=np.prod(input_shape),
        nonlinearity=lasagne.nonlinearities.sigmoid)

    x = lasagne.layers.ReshapeLayer(
        x, shape=([0],) + input_shape)

    l_out = lasagne.layers.TransformerLayer(x, transf_param)

    return l_out, (z_mean, z_logsigma), z, transf_param


# 2 latent, 128 hidden, 1 layer, 130
# 2 latent, 256 hidden, 1 layer, 125
# 2 latent, 512 hidden, 1 layer, 116
# 2 latent, 64 hidden, 2 layers, 113
# 2 latent, 128 hidden, 2 layers, 110
# 2 latent, 256 hidden, 2 layers, 110
# 2 latent, 64 hidden, 3 layers, 112
# 2 latent, 128 hidden, 3 layers, 108
# 2 latent, 256 hidden, 3 layers, 108


# 14 latent, 128 hidden 1 layer, 118
# 14 latent, 256 hidden 1 layer, 113
# 14 latent, 512 hidden 1 layer, 114
# 14 latent, 128 hidden 2 layer, 107
# 14 latent, 256 hidden 2 layers, 100
# 14 latent, 512 hidden 2 layers, 106
# 14 latent, 128 hidden 3 layers, 114
# 14 latent, 256 hidden 3 layers, ?

# 18 latent, 128 hidden 1 layer, 109
# 18 latent, 256 hidden 1 layer, 110

# WRONG DUE TO A PREVIOUS MISTAKE
# 18 latent, 128 hidden 1 layer, 126
# 18 latent, 256 hidden 1 layer, 120
# 18 latent, 512 hidden 1 layer, 125
# 18 latent, 128 hidden 2 layers, 112
# 18 latent, 256 hidden 2 layers, 108
# 18 latent, 512 hidden 2 layers, 115
# 18 latent, 128 hidden 3 layers, 113

