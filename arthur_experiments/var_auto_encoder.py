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
    input_var=None, input_shape=(None, ), n_latent=26, n_hidden = 256):
    drop_prob = 0.

    # For 2 latent best result achieved for 128 (both train and test at ~152)
    # For 8 latent, 128 -> 124

    x1 = lasagne.layers.InputLayer(
        shape=(None,) + input_shape,
        input_var=input_var)
    x2 = lasagne.layers.DenseLayer(
        x1, num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.rectify)
    x2 = lasagne.layers.DenseLayer(
        x2, num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.rectify)
    # x2 = lasagne.layers.DropoutLayer(x2, p=drop_prob)

    z_mean = lasagne.layers.DenseLayer(
        x2, num_units=n_latent, nonlinearity = None)

    relu_shift = 6  # 10
    z_logsigma = lasagne.layers.DenseLayer(
        x2, num_units=n_latent,
        nonlinearity=lambda a: T.nnet.relu(a + relu_shift) - relu_shift)

    z = GaussianSampleLayer(z_mean, z_logsigma)

    x3 = lasagne.layers.DenseLayer(
        z, num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.rectify)
    x3 = lasagne.layers.DenseLayer(
        x3, num_units=n_hidden,
        nonlinearity=lasagne.nonlinearities.rectify)
    #x3 = lasagne.layers.DropoutLayer(x3, p=drop_prob)
    x4 = lasagne.layers.DenseLayer(
        x3, num_units=np.prod(input_shape),
        nonlinearity=lasagne.nonlinearities.sigmoid)
    l_out = lasagne.layers.ReshapeLayer(
        x4, shape=([0],) + input_shape)
    return l_out, (z_mean, z_logsigma), z


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
batch_size = 100
n_latent = 20
n_hidden = 256
#24 512 -> CE 73
#24 512 -> CE 74

print("Loading data...")


(X_train, y_train), (X_val, y_val) = load_mnist()
# X_train = np.floor(X_train + 0.5)
# X_val = np.floor(X_val + 0.5)
# print(np.unique(X_train))
#X_train=X_train[0:1000]
#X_val=X_val[0:1000]
prod_shape = np.prod(X_train.shape[1:])

input_var = T.tensor4('inputs')
output_var = T.tensor4('outputs')

print("Building model ...")
network, (z_mu_net, z_ls_net), z_net = build_var_autoencoder(
    input_var, X_train.shape[1:], n_latent=n_latent, n_hidden=n_hidden)


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


outputs = lasagne.layers.get_output([network, z_mu_net, z_ls_net, z_net])
prediction = outputs[0]
z_mu = outputs[1]
z_ls = outputs[2]
z = outputs[3]
# prediction = lasagne.layers.get_output(network)
# z_mu = lasagne.layers.get_output(z_mu_net)
# z_ls = lasagne.layers.get_output(z_ls_net)
loss = loss_fn(prediction, output_var, z_mu, z_ls)

params = lasagne.layers.get_all_params(network, trainable=True)


grad = T.grad(loss, params)
#grad = lasagne.updates.total_norm_constraint(grad, 1)
updates = lasagne.updates.adadelta(grad, params)
#momentum(grad, params, learning_rate=0.1)
# adadelta(
# adam(loss, params, learning_rate=1e-4)
# adadelta(#adadelta(# nesterov_momentum(
#            grad, params)#, learning_rate=0.0001, momentum=0.)
# adadelta(loss, params)
#0.01 - 206.17
#adadelta - 206.5

outputs = lasagne.layers.get_output(
    [network, z_mu_net, z_ls_net, z_net], deterministic=True)
test_prediction = outputs[0]
z_mu = outputs[1]
z_ls = outputs[2]
z = outputs[3]
test_loss = loss_fn(test_prediction, output_var, z_mu, z_ls, True)

print("Compiling functions ...")
vect = T.matrix()
generated_x = lasagne.layers.get_output(network, {z_net: vect})
decode_fn = theano.function([vect], generated_x)
encode_fn = theano.function([input_var], z)
train_fn = theano.function([input_var, output_var], loss, updates=updates)
    #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
val_fn = theano.function([input_var, output_var], test_loss)
predict_fn = theano.function([input_var], test_prediction)


score = train_model(X_train, X_train, X_val, X_val, train_fn,
            val_fn, num_epochs, batch_size, True)
print("Best score", score)


z_trains = []
for batch in iterate_minibatches(X_train, X_train, 100, shuffle=False):
    inputs, targets = batch
    z_trains += [encode_fn(inputs)]
z_train = np.concatenate(z_trains, axis=0)
print("Z1 ", np.mean(z_train[:, 0]), np.std(z_train[:, 0]))
print("Z2 ", np.mean(z_train[:, 1]), np.std(z_train[:, 1]))
# print(np.corrcoef(np.transpose(z_train)))

print("Saving samples ...")

n = 10
decoded_imgs = predict_fn(X_val[:n])
img = np.array([])
for i in range(n):
    # display original
    img_1 = np.concatenate([X_val[i].reshape(28, 28),
                            decoded_imgs[i].reshape(28, 28)],
                           axis=0)
    if i == 0:
        img = img_1
    else:
        img = np.concatenate([img, img_1], axis=1)

img = img.reshape(2 * 28, n * 28, 1)
img = 1.-np.concatenate([img, img, img], axis=2)

from scipy.misc import imsave
imsave('sample_val.jpg', img)


print("Saving manifold (random generated samples if n_latent != 2 ...")
n = 10

z_train_m_0 = np.mean(z_train[:, 0])
z_train_sd_0 = np.std(z_train[:, 0])
z_train_m_1 = np.mean(z_train[:, 1])
z_train_sd_1 = np.std(z_train[:, 1])

x = np.linspace(z_train_m_0 - 3 * z_train_sd_0, z_train_m_0 + 3 * z_train_sd_0, n)
y = np.linspace(z_train_m_1 - 3 * z_train_sd_1, z_train_m_1 + 3 * z_train_sd_1, n)

x = np.linspace(1 / n, 1 - 1 / n, n)
y = np.linspace(1 / n, 1 - 1 / n, n)

pred_mat = []
for k, i in enumerate(x):
    preds_line = []
    for m, j in enumerate(y):
        from scipy.stats import norm
        if n_latent ==2:
            true_i = norm.ppf(i)
            true_j = norm.ppf(j)
            data = np.array([true_i, true_j], dtype='float32')
        else:
            data = norm.rvs(size=n_latent).astype('float32')
        data = data.reshape(1, n_latent)
        pred = decode_fn(data)[0][0]
        preds_line += [pred]
    pred_mat += [np.concatenate(preds_line, axis=0)]
manifold = np.concatenate(pred_mat, axis=1)

manifold = 1.-manifold[np.newaxis, :]
print(manifold.shape)

img = np.concatenate([manifold, manifold, manifold], axis=0)
img = img.transpose(1, 2, 0)
imsave('manifold.jpg', img)

print("Saving encoding ...")


# encode_train = np.zeros((X_train.shape[0], n_latent))
# y_enc_train = np.zeros(y_train.shape)
# for i, batch in enumerate(iterate_minibatches(X_train, y_train, 100, shuffle=False)):
#     inputs, targets = batch
#     encode_train[(i * 100): ((i + 1) * 100), :] = encode_fn(inputs)
#     y_enc_train[(i * 100): ((i + 1) * 100)] = targets
#
# encode_val = np.zeros((X_val.shape[0], n_latent))
# y_enc_val = np.zeros(y_val.shape)
# for i, batch in enumerate(iterate_minibatches(X_val, y_val, 100, shuffle=False)):
#     inputs, targets = batch
#     encode_val[(i * 100):((i + 1) * 100), :] = encode_fn(inputs)
#     y_enc_val[(i * 100): ((i + 1) * 100)] = targets
#
# import pandas as pd
# encode_train_df = pd.DataFrame(encode_train)
# encode_val_df = pd.DataFrame(encode_val)
# encode_train_df["y"] = y_enc_train
# encode_val_df["y"] = y_enc_val
# encode_train_df.to_csv("encoded_mnist_train.csv")
# encode_val_df.to_csv("encoded_mnist_val.csv")
