import theano.tensor as T
import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import vae.var_auto_encoder as vae
import tvae_1.prev_transf_var_auto_encoder as tvae
import tps_vae.tps_vae as tps_vae

sys.path.append("..")

from data import load_mnist
from train_utils import train_model
from train_utils import iterate_minibatches


def run_exp(data, module, n_epoch=50, batch_size=100, id_="-1",
            n_latent=2, n_hidden=128, n_layers=1, save_encoding=False, spatial=None, n_spatial = 0):
    id_ = str(id_)
    (X_train, y_train), (X_val, y_val) = data

    input_var = T.tensor4('inputs')

    funcs = module.build_model(
        input_var, X_train.shape[1:], n_latent=n_latent, n_hidden=n_hidden,
        n_layers=n_layers, batch_size=batch_size)

    if spatial is None:

        decode_fn, encode_fn, train_fn, val_fn, predict_fn = funcs
    else:
        decode_fn, encode_fn, train_fn, val_fn, predict_fn, param_fn = funcs

    score = train_model(X_train, X_train, X_val, X_val, train_fn,
                        val_fn, num_epochs, batch_size, True)
    print("Best score", score)

    z_trains = []
    mean_trans = np.zeros((batch_size, n_spatial))
    for batch in iterate_minibatches(X_train, X_train, 100, shuffle=False):
        inputs, targets = batch
        z_trains += [encode_fn(inputs)]
        if spatial is not None:
            mean_trans += param_fn(inputs)
    z_train = np.concatenate(z_trains, axis=0)
    if spatial is not None: 
        mean_trans = np.sum(mean_trans, axis=0) / X_train.shape[0]

    print("Z1 ", np.mean(z_train[:, 0]), np.std(z_train[:, 0]))
    print("Z2 ", np.mean(z_train[:, 1]), np.std(z_train[:, 1]))
    corr = np.corrcoef(np.transpose(z_train))
    dist_corr = np.sqrt(np.mean((corr - np.identity(n_latent))**2))
    print(dist_corr)

    def computeChiSquared(z):
        n = z.shape[0]
        i = 0
        res = 0
        while i + 1 < n:
            res += np.sum((z[i + 1] - z[i])**2)
            i += 2
        return res / 2

    import scipy.stats as st

    chi2_0 = np.sum(z_train**2)
    dof_0 = n_latent * z_train.shape[0]
    disp_0 = chi2_0 / dof_0
    print("chi2 from 0: ", chi2_0, "on ", dof_0, "degrees of freedom")
    print("Dispertion :", disp_0)

    chi2 = computeChiSquared(z_train)
    dof = n_latent * (z_train.shape[0] // 2)
    disp_dist = chi2 / dof
    print("chi2: ", chi2, "on ", dof, "degrees of freedom")
    print("Dispersion: ", disp_dist)
    print("p-value:", st.chi2.sf(chi2, dof))

    print("Saving samples ...")

    n = 10
    decoded_imgs = predict_fn(X_val[:n])
    img = np.array([])
    for i in range(n):
        # display original
        img_1 = np.concatenate([X_val[i].reshape(28, 28),
                                decoded_imgs[i].reshape(28, 28)], axis=0)
        if i == 0:
            img = img_1
        else:
            img = np.concatenate([img, img_1], axis=1)

    img = img.reshape(2 * 28, n * 28, 1)
    img = np.concatenate([img, img, img], axis=2)

    from scipy.misc import imsave
    imsave('output/sample_val_' + id_ + '.jpg', img)

    print("Saving manifold (random generated samples if n_latent != 2 ...")
    n = 20

    x = np.linspace(1 / n, 1 - 1 / n, n)
    y = np.linspace(1 / n, 1 - 1 / n, n)

    pred_mat = []
    if spatial is not None:
        identity = mean_trans.reshape(1, n_spatial).astype('float32')

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
            if spatial is None:
                pred = decode_fn(data)[0][0]
            else:
                pred = decode_fn(data, identity)[0][0]

            preds_line += [pred]
        pred_mat += [np.concatenate(preds_line, axis=1)]
    manifold = np.concatenate(pred_mat, axis=0)

    manifold = manifold[np.newaxis, :]
    print(manifold.shape)
    line_sample = pred_mat[0][np.newaxis, :]
    line_sample = np.concatenate(3 * [line_sample], axis=0)
    line_sample = line_sample.transpose(1, 2, 0)
    img = np.concatenate([manifold, manifold, manifold], axis=0)
    img = img.transpose(1, 2, 0)
    imsave('output/manifold_' + id_ + '.jpg', img)
    imsave('output/new_sample_' + id_ + '.jpg', line_sample)

    # NEAREST NEIGHBORS METRICS
    # Dist between images
    # train - test
    print("Nearest neighbors metrics computations ...")
    print("- in the image space -")
    nn = NearestNeighbors(n_neighbors=10, algorithm="brute")
    nn.fit(X_train.reshape(X_train.shape[0], -1))
    dist, _ = nn.kneighbors(X_val.reshape(X_val.shape[0], -1))
    dst_test_train = np.mean(dist)

    # gen - train
    n_gen = 10000
    z_gen = np.random.standard_normal((n_gen, n_latent)).astype("float32")
    if spatial is None:
        x_gen = decode_fn(z_gen)
    else:
        x_gen = decode_fn(z_gen, np.repeat(identity, n_gen, axis=0))
    dist, _ = nn.kneighbors(x_gen.reshape(X_val.shape[0], -1))
    dst_gen_train = np.mean(dist)

    print("- in the embedded space -")
    nn = NearestNeighbors(n_neighbors=10, algorithm="brute")
    nn.fit(z_train)
    z_val = encode_fn(X_val)
    dist, _ = nn.kneighbors(z_val)
    dst_test_train_emb = np.mean(dist)

    dist, _ = nn.kneighbors(z_gen)
    dst_gen_train_emb = np.mean(dist)

    print("Saving dist in embedded space ...")
    perm = np.random.permutation(X_train.shape[0])[:n_gen]
    plt.scatter(z_train[perm, 0], z_train[perm, 1], c=y_train[perm], s=5)
    plt.savefig("output/dist_train_" + id_ + ".png")
    plt.scatter(z_val[:, 0], z_val[:, 1], c=y_val, s=5)
    plt.savefig("output/dist_val_" + id_ + ".png")

    pca = PCA(n_components=2)
    z_train_pca = pca.fit_transform(z_train)
    plt.scatter(z_train_pca[perm, 0], z_train_pca[perm, 1], c=y_train[perm], s=5)
    plt.savefig("output/dist_train_pca_" + id_ + ".png")
    z_val_pca = pca.fit_transform(z_val)
    plt.scatter(z_val_pca[:, 0], z_val_pca[:, 1], c=y_val, s=5)
    plt.savefig("output/dist_val_pca_" + id_ + ".png")



    if save_encoding:
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

    with open('file.txt', 'a') as file:
        txt = "\n n_ep: " + str(n_epoch)
        txt = txt + " b_s: " + str(batch_size)
        txt = txt + " id_" + id_
        txt = txt + " n_lat: " + str(n_latent)
        txt = txt + " n_hid: " + str(n_hidden)
        txt = txt + " n_layers: " + str(n_layers) + "\n"
        txt = txt + " score: " + "{0:.2f}".format(score)
        txt = txt + " dist corr: " + "{0:.4f}".format(dist_corr)
        txt = txt + " disp 0: " + "{0:.2f}".format(disp_0)
        txt = txt + " disp_dist: " + "{0:.2f}".format(disp_dist)
        txt = txt + " test_train_im: " + "{0:.2f}".format(dst_test_train)
        txt = txt + " gen_train_im: " + "{0:.2f}".format(dst_gen_train)
        txt = txt + " test_train_emb: " + "{0:.4f}".format(dst_test_train_emb)
        txt = txt + " gen_train_emb: " + "{0:.4f}".format(dst_gen_train_emb)
        file.write(txt)


if __name__ == "__main__":

    print("Loading data...")

    data = load_mnist("../../data")

    num_epochs = 50
    batch_size = 100

    # VAE
    # run_exp(data, vae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_=18, n_latent=2, n_hidden=64, n_layers=2)
    # run_exp(data, vae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_=19, n_latent=8, n_hidden=512, n_layers=1)
    # run_exp(data, vae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_=20, n_latent=20, n_hidden=128, n_layers=1)
    # run_exp(data, vae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_=21, n_latent=30, n_hidden=128, n_layers=1)

    # TVAE
    # run_exp(data, tvae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_=26, n_latent=2, n_hidden=256, n_layers=2,
    #         spatial="tvae", n_spatial=6)
    # run_exp(data, tvae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_=27, n_latent=14, n_hidden=256, n_layers=2,
    #         spatial="tvae", n_spatial=6)
    # run_exp(data, tvae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_=28, n_latent=24, n_hidden=256, n_layers=2,
    #         spatial="tvae", n_spatial=6)

    # TPS_VAE
    run_exp(data, tps_vae, n_epoch=num_epochs, batch_size=batch_size,
            id_=38, n_latent=2, n_hidden=256, n_layers=2,
            spatial="tps_vae", n_spatial=18)
    run_exp(data, tps_vae, n_epoch=num_epochs, batch_size=batch_size,
            id_=39, n_latent=12, n_hidden=256, n_layers=2,
            spatial="tps_vae", n_spatial=18)
