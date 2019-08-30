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
                                 n_hidden=256, n_control=3**2, n_layers=1,
                                 n_latent=2):

    x = lasagne.layers.InputLayer(
        shape=(None,) + input_shape,
        input_var=input_var)
    for i in range(n_layers):
        x = lasagne.layers.DenseLayer(
            x, num_units=n_hidden,
            nonlinearity=lasagne.nonlinearities.rectify)

    transf_param = lasagne.layers.DenseLayer(
        x, num_units=2 * n_control, nonlinearity=lasagne.nonlinearities.tanh)
    sample_line = np.linspace(-1, 1, np.sqrt(n_control))
    mesh = np.meshgrid(sample_line, sample_line)
    identity = np.concatenate(mesh).flatten().astype('float32')

    scale = 1
    # scaling and then translation
    transf_param = lasagne.layers.standardize(
        transf_param, np.zeros(2 * n_control),
        scale=scale * np.ones(2 * n_control))
    # /!\ the layer is dividing the input by scale
    transf_param = lasagne.layers.standardize(
        transf_param, -identity, scale=np.ones(2 * n_control))

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

    l_out = lasagne.layers.TPSTransformerLayer(
        x, transf_param, control_points=n_control)

    return l_out, (z_mean, z_logsigma), z, transf_param



if __name__ == "__main__":
    num_epochs = 50
    batch_size = 100
    n_control = 3**2
    n_hidden = 256
    n_layer = 2
    n_latent = 12
    id_ = str(39)
    print("Loading data...")


    (X_train, y_train), (X_val, y_val) = load_mnist("../../data")
    prod_shape = np.prod(X_train.shape[1:])

    input_var = T.tensor4('inputs')
    output_var = T.tensor4('outputs')

    print("Building model ...")
    network, (z_mu_net, z_ls_net), z_net, trans_param_net = build_transf_var_autoencoder(
        input_var, X_train.shape[1:], n_layer=n_layer, n_hidden=n_hidden, n_control=n_control,
        n_latent=n_latent)


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

    # prediction = lasagne.layers.get_output(network)
    # z_mu = lasagne.layers.get_output(z_mu_net)
    # z_ls = lasagne.layers.get_output(z_ls_net)

    loss = loss_fn(prediction, output_var, z_mu, z_ls)

    params = lasagne.layers.get_all_params(network, trainable=True)


    grad = T.grad(loss, params)
    #grad = lasagne.updates.total_norm_constraint(grad, 1)
    updates = lasagne.updates.adadelta(
    #adam(loss, params, learning_rate=1e-4)
    #adadelta(#adadelta(# nesterov_momentum(
                grad, params)#, learning_rate=0.0001, momentum=0.)
    # adadelta(loss, params)

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
        #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    val_fn = theano.function([input_var, output_var], test_loss)
    predict_fn = theano.function([input_var], test_prediction)
    z_vect = T.matrix()
    trans_par = T.matrix()
    generated_x = lasagne.layers.get_output(network, {z_net: z_vect,
                                                      trans_param_net: trans_par})
    decode_fn = theano.function([z_vect, trans_par], generated_x)
    encode_fn = theano.function([input_var], z)
    param_fn = theano.function([input_var], trans_param)

    train_model(X_train, X_train, X_val, X_val, train_fn,
                val_fn, num_epochs, batch_size, comp_acc=True)

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
    img = np.concatenate([img, img, img], axis=2)
    from scipy.misc import imsave
    print("Saving samples ...")
    imsave('output/sample_val_' + id_ + '.jpg', img)

    z_trains = []
    mean_trans = np.zeros((100, 2 * n_control))
    for batch in iterate_minibatches(X_train, X_train, 100, shuffle=False):
        inputs, targets = batch
        z_trains += [encode_fn(inputs)]
        mean_trans += param_fn(inputs)

    mean_trans = np.sum(mean_trans, axis=0) / X_train.shape[0]
    z_train = np.concatenate(z_trains, axis=0)
    print(np.mean(z_train[:, 0]), np.std(z_train[:, 0]))
    print(np.mean(z_train[:, 1]), np.std(z_train[:, 1]))
    print(np.corrcoef(np.transpose(z_train)))
    print("Average transform:", mean_trans)



    print("if 2 latent variables")
    print("Saving manifold ...")
    n = 20
    x = np.linspace(1 / n, 1 - 1 / n, n)
    y = np.linspace(1 / n, 1 - 1 / n, n)

    pred_mat = []


    identity = mean_trans.reshape(1, 2 * n_control).astype('float32')
    for k, i in enumerate(x):
        preds_line = []
        for m, j in enumerate(y):
            from scipy.stats import norm
            if n_latent == 2:
                true_i = norm.ppf(i)
                true_j = norm.ppf(j)
                data = np.array([true_i, true_j], dtype='float32')
            else:
                data = norm.rvs(size=n_latent).astype('float32')
            data = data.reshape(1, n_latent)
            pred = decode_fn(data, identity)[0][0]
            preds_line += [pred]
        pred_mat += [np.concatenate(preds_line, axis=0)]

    manifold = np.concatenate(pred_mat, axis=1)
    manifold = manifold[np.newaxis, :]
    img = np.concatenate([manifold, manifold, manifold], axis=0)
    img = img.transpose(1, 2, 0)
    imsave('output/manifold_' + id_ + '.jpg', img)

    sample_line =  np.linspace(-1, 1, np.sqrt(n_control))
    mesh = np.meshgrid(sample_line, sample_line)
    identity = np.concatenate(mesh).flatten().astype('float32').reshape(1, -1)
    for k, i in enumerate(x):
        preds_line = []
        for m, j in enumerate(y):
            from scipy.stats import norm
            if n_latent == 2:
                true_i = norm.ppf(i)
                true_j = norm.ppf(j)
                data = np.array([true_i, true_j], dtype='float32')
            else:
                data = norm.rvs(size=n_latent).astype('float32')
            data = data.reshape(1, n_latent)
            pred = decode_fn(data, identity)[0][0]
            preds_line += [pred]
        pred_mat += [np.concatenate(preds_line, axis=0)]

    manifold = np.concatenate(pred_mat, axis=1)
    manifold = manifold[np.newaxis, :]
    img = np.concatenate([manifold, manifold, manifold], axis=0)
    img = img.transpose(1, 2, 0)
    imsave('output/manifold_2_' + id_ + '.jpg', img)

