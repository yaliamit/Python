    import sys
    sys.path.append("../..")

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

    import vae.var_auto_encoder as vae

    def run_exp(data, module, n_epoch=50, batch_size=100, id_="-1",
                n_latent=2, n_hidden=128, n_layers=1):
    (X_train, y_train), (X_val, y_val) = data
    prod_shape = np.prod(X_train.shape[1:])

    input_var = T.tensor4('inputs')
    output_var = T.tensor4('outputs')

    print("Building model ...")
    network, (z_mu_net, z_ls_net), z_net = module.build_var_autoencoder(
    input_var, X_train.shape[1:], n_latent=n_latent, n_hidden=n_hidden,
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
    # updates = lasagne.updates.momentum(grad, params, learning_rate=0.0001)
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
    img = np.concatenate([img, img, img], axis=2)

    from scipy.misc import imsave
    imsave('output/sample_val_'+id_+'.jpg', img)


    print("Saving manifold (random generated samples if n_latent != 2 ...")
    n = 20

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
        if n_latent == 2:
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

    manifold = manifold[np.newaxis, :]
    print(manifold.shape)

    img = np.concatenate([manifold, manifold, manifold], axis=0)
    img = img.transpose(1, 2, 0)
    imsave('output/manifold_'+id_+'.jpg', img)

    print("Saving encoding ...")


    encode_train = np.zeros((X_train.shape[0], n_latent))
    y_enc_train = np.zeros(y_train.shape)
    for i, batch in enumerate(iterate_minibatches(X_train, y_train, 100, shuffle=False)):
    inputs, targets = batch
    encode_train[(i * 100): ((i + 1) * 100), :] = encode_fn(inputs)
    y_enc_train[(i * 100): ((i + 1) * 100)] = targets

    encode_val = np.zeros((X_val.shape[0], n_latent))
    y_enc_val = np.zeros(y_val.shape)
    for i, batch in enumerate(iterate_minibatches(X_val, y_val, 100, shuffle=False)):
    inputs, targets = batch
    encode_val[(i * 100):((i + 1) * 100), :] = encode_fn(inputs)
    y_enc_val[(i * 100): ((i + 1) * 100)] = targets

    import pandas as pd
    encode_train_df = pd.DataFrame(encode_train)
    encode_val_df = pd.DataFrame(encode_val)
    encode_train_df["y"] = y_enc_train
    encode_val_df["y"] = y_enc_val
    encode_train_df.to_csv("encoded_mnist_train.csv")
    encode_val_df.to_csv("encoded_mnist_val.csv")



if __name__ == "__main__":

    print("Loading data...")

    data = load_mnist("../../data")

    num_epochs = 50
    batch_size = 100
    id_ = str(21)
    n_latent = 30
    n_hidden = 128
    n_layers = 1

    run_exp(data, vae, n_epoch=num_epochs, batch_size=batch_size,
            id_=id_, n_latent=n_latent, n_hidden=n_hidden, n_layers=n_layers)

