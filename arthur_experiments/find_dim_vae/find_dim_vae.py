from data import load_mnist
from train_utils import train_model
from train_utils import iterate_minibatches
import theano
import theano.tensor as T
import lasagne
import numpy as np
import time
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.compile.nanguardmode import NanGuardMode
from PIL import Image


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


def build_var_autoencoder(
    input_var=None, input_shape=(None, ), n_latent=2, n_hidden=128):

    # For 2 latent best result achieved for 128 (both train and test at ~15200)
    # For 8 latent, 128 -> 124

    x = lasagne.layers.InputLayer(
        shape=(None,) + input_shape,
        input_var=input_var)
    x = lasagne.layers.DenseLayer(
        x, num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.rectify)

    z_mean = lasagne.layers.DenseLayer(
        x, num_units=n_latent, nonlinearity=None)

    relu_shift = 6
    z_logsigma = lasagne.layers.DenseLayer(
        x, num_units=n_latent,
        nonlinearity=lambda a: T.nnet.relu(a + relu_shift) - relu_shift)

    z = GaussianSampleLayer(z_mean, z_logsigma)

    x = lasagne.layers.DenseLayer(
        z, num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.rectify)
    x = lasagne.layers.DenseLayer(
        x, num_units=np.prod(input_shape),
        nonlinearity=lasagne.nonlinearities.sigmoid)
    l_out = lasagne.layers.ReshapeLayer(
        x, shape=([0],) + input_shape)
    return l_out, (z_mean, z_logsigma), z


def train_model(X_train, y_train, X_val, y_val,
                train_fn, val_fn, num_epochs, batch_size):
    print("Starting training...")
    quit = False
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size,
                                         shuffle=True):
            inputs, targets = batch
            try:
                train_err += train_fn(inputs, targets)
            except KeyboardInterrupt:
                quit = True
                break
            train_batches += 1
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size,
                                         shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        if np.isnan(train_err) or quit:
            break

def train_model(X_train, y_train, X_val, y_val, train_fn, val_fn, num_epochs,
                batch_size=128, comp_acc=False):
    print("Starting training...")
    quit = False
    if comp_acc:
        best_val = 1000
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train,
                                         batch_size, shuffle=True):
            inputs, targets = batch
            try:
                train_err += train_fn(inputs, targets)
            except KeyboardInterrupt:
                quit = True
                break
            train_batches += 1
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val,
                                         batch_size, shuffle=False):
            inputs, targets = batch
            if comp_acc:
                err, acc = val_fn(inputs, targets)
                val_acc += acc
            else:
                try:
                    err = val_fn(inputs, targets)
                except KeyboardInterrupt:
                    break
            val_err += err
            val_batches += 1
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        if comp_acc:
            print("  validation CE:\t\t{:.2f}".format(
                  val_acc / val_batches))
            if val_acc / val_batches < best_val:
                best_val = val_acc / val_batches
        if np.isnan(train_err) or quit:
            break
    if val_acc:
        return best_val

num_epochs = 50
#num_epochs = 4
eps = 1e-6
batch_size = 100
print("Loading data...")

(X_train, y_train), (X_val, y_val) = load_mnist()
prod_shape = np.prod(X_train.shape[1:])

#latent_values = [2, 4, 8, 16, 32, 64, 128, 256, 512]
latent_values = [16, 20, 24, 28, 32, 48, 64]
hidden_values = [128, 256, 512]

#latent_values = [2, 4]
#hidden_values = [30]

input_var = T.tensor4('inputs')
output_var = T.tensor4('outputs')
hist = np.zeros((len(latent_values), len(hidden_values)))

for i, n_latent in enumerate(latent_values):
    for j, n_hidden in enumerate(hidden_values):
        print("Building model for", n_latent, n_hidden, "...", i, j)
        network, (z_mu_net, z_ls_net), z_net = build_var_autoencoder(
            input_var, X_train.shape[1:], n_latent=n_latent,
            n_hidden=n_hidden)

        def loss_fn(y_pred, y_true, z_mu, z_ls):
                y_pred = y_pred.clip(eps, 1 - eps)
                kl_div = 1 + 2 * z_ls - T.sqr(z_mu) - T.exp(2 * z_ls)
                kl_div = T.sum(kl_div) / batch_size
                logpxz = lasagne.objectives.binary_crossentropy(y_pred, y_true)
                logpxz = logpxz.sum() / batch_size
                loss = logpxz - 0.5 * kl_div
                return loss

        def val_fn(y_pred, y_true):
            y_pred = y_pred.clip(eps, 1 - eps)
            ce = lasagne.objectives.binary_crossentropy(y_pred, y_true)
            return ce.sum() / batch_size

        outputs = lasagne.layers.get_output([network, z_mu_net, z_ls_net, z_net])
        prediction = outputs[0]
        z_mu = outputs[1]
        z_ls = outputs[2]
        z = outputs[3]

        loss = loss_fn(prediction, output_var, z_mu, z_ls)

        params = lasagne.layers.get_all_params(network, trainable=True)
        grad = T.grad(loss, params)
        #grad = lasagne.updates.total_norm_constraint(grad, 1)
        updates = lasagne.updates.adadelta(grad, params)

        outputs = lasagne.layers.get_output(
            [network, z_mu_net, z_ls_net, z_net], deterministic=True)

        test_prediction = outputs[0]
        z_mu = outputs[1]
        z_ls = outputs[2]
        z = outputs[3]
        test_loss = loss_fn(test_prediction, output_var, z_mu, z_ls)
        test_ce = val_fn(test_prediction, output_var)

        print("Compiling functions ...")
        vect = T.matrix()
        generated_x = lasagne.layers.get_output(network, {z_net: vect})
        decode_fn = theano.function([vect], generated_x)
        encode_fn = theano.function([input_var], z)
        train_fn = theano.function([input_var, output_var], loss, updates=updates)
        # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        val_fn = theano.function([input_var, output_var], [test_loss, test_ce])
        predict_fn = theano.function([input_var], test_prediction)

        hist[i, j] = train_model(X_train, X_train, X_val, X_val, train_fn,
                                 val_fn, num_epochs, batch_size, True)
        print("best for this turn:", hist[i, j])

print(hist)
np.savetxt("results_raw.txt", hist)
max_hist = hist.max(axis=1)
np.savetxt("results_best.txt", max_hist)

import matplotlib.pyplot as plt

plt.plot(np.array(latent_values), max_hist)
plt.savefig("plot.png")
