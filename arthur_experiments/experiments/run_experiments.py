# This script is responsible for managing the experiments, it regroups
# the routines shared by all models: data, training, plot generation, etc.
# Change the end of the file to run new experiments


import theano.tensor as T
import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Import all the different architecture / model
#   Standard Variational Autoencoder
import var_auto_encoder as vae
# Transformer Variational Autoencoder, affine transformation, no prior on spatial
import prev_transf_var_auto_encoder as tvae
# Transformer Variational Autoencoder, affine transformation, diagonal normal prior on spatial
import tvae_aff_mu_sig.tvae as tvae_aff_mu_sig
from  tvae_aff_sig import tvae as tvae_aff_sig
# Transformer Variational Autoencoder, affine transformation reparametrized with 5 d.o.f, diagonal normal prior on spatial
import tvae_aff_mu_sig_repar.tvae as tvae_aff_mu_sig_repar
# Transformer Variational Autoencoder, TPS transformation, no prior on spatial
import tps_vae.tps_vae as tps_vae
import prev_transf_var_auto_encoder_3 as tvae_sig_3
import conv_tvae.tvae as conv_tvae
import conv_vae.vae as conv_vae
# import tvae_1_mu_sig.tvae as tvae_mu_sig
import tps_vae_nn_prior.tps_vae as tps_vae_nn_prior

sys.path.append("..")

from data import load_mnist
from train_utils import train_model
from train_utils import iterate_minibatches


def run_exp(data, module, n_epoch=50, batch_size=100, id_="-1",
            n_latent=2, n_hidden=128, n_layers=1, save_encoding=False, spatial="None", n_spatial = 0, n_hidden_prior=10, save_distances=True):
    id_ = str(id_)
    (X_train, y_train), (X_val, y_val) = data

    input_var = T.tensor4('inputs')
    print(X_train.shape)
    funcs = module.build_model(
        input_var, X_train.shape[1:], n_latent=n_latent, n_hidden=n_hidden,
        n_layers=n_layers, batch_size=batch_size)#, n_spatial=n_spatial, n_hidden_prior=n_hidden_prior)

    if spatial == 'None':
        decode_fn, encode_fn, train_fn, val_fn, predict_fn = funcs
    elif "mu_sig" in spatial:
        decode_fn, encode_fn, train_fn, val_fn, predict_fn, param_fn, mu_spatial, sigma_spatial = funcs
    elif not "sig" in spatial or "tps" in spatial:
        decode_fn, encode_fn, train_fn, val_fn, predict_fn, param_fn = funcs
    else:
        decode_fn, encode_fn, train_fn, val_fn, predict_fn, param_fn, sigma_spatial = funcs

    score = train_model(X_train, X_train, X_val, X_val, train_fn,
                        val_fn, num_epochs, batch_size, True)

    print("Best score", score)
    if "sig" in spatial and "tps" not in spatial:
        log_sigma = sigma_spatial.b.get_value()
        print("Estimation of spatial standard deviation: ",
              log_sigma, np.exp(log_sigma))
    if "mu" in spatial:
        #mu = mu_spatial.b.get_value()
        mu_spatial = mu_spatial + 0
        mu = mu_spatial.eval()
        print("Estimation of spatial mean: ",
              mu)

    z_trains = []
    mean_trans = []
    for batch in iterate_minibatches(X_train, X_train, 100, shuffle=False):
        inputs, targets = batch
        z_trains += [encode_fn(inputs)]
        if spatial != 'None':
            mean_trans += [param_fn(inputs)]
    z_train = np.concatenate(z_trains, axis=0)
    if spatial != 'None': 
        mean_trans = np.concatenate(mean_trans, axis=0)
        print("sd mean transf:", np.std(mean_trans, axis=0))
        cov_trans = np.cov(mean_trans.T)
        print("Cor transf", np.corrcoef(mean_trans.T))
        mean_trans = np.mean(mean_trans, axis=0)
        print("mean transf:", mean_trans)

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

    n = 5

    x = np.linspace(1 / n, 1 - 1 / n, n)
    y = np.linspace(1 / n, 1 - 1 / n, n)

    pred_mat = []
    pred_mat_id = []
    pred_mat_sig = []

    if spatial  != 'None':
        post_identity = mean_trans.reshape(1, n_spatial).astype('float32')
        identity = np.zeros((1, n_spatial), dtype="float32")
        if n_spatial == 6:
            identity = np.array([[1, 0, 0, 0, 1, 0]], dtype='float32')
        elif n_spatial == 5:
            identity = np.array([[0, 1, 1, 0, 0]], dtype='float32')


    for k, i in enumerate(x):
        preds_line = []
        preds_line_id = []
        preds_line_sig = []
        for m, j in enumerate(y):
            from scipy.stats import norm
            if n_latent == 2:
                true_i = norm.ppf(i)
                true_j = norm.ppf(j)
                data = np.array([true_i, true_j], dtype='float32')
            else:
                data = norm.rvs(size=n_latent).astype('float32')

            if "sig" in spatial:
                noise = np.random.standard_normal((1, n_spatial)).astype('float32')
                noise = noise * np.exp(log_sigma)
                mean_params = np.zeros((1, n_spatial))
                if n_spatial == 6:
                    mean_params = np.array([[1, 0, 0, 0, 1, 0]], dtype='float32')

                if "mu" in spatial:
                    mean_params = mu  # + np.array([[1, 0, 0, 0, 1, 0]], dtype='float32')
                #  np.array([[1, 0, 0, 0, 1, 0]], dtype='float32') identity
                # spat_params = np.array([[1, 0, 0, 0, 1, 0]], dtype='float32') + np.exp(log_sigma) * noise
                # spat_params = np.array([[-0.3, 1, 1, 0, 0]], dtype='float32') + np.exp(log_sigma) * noise
                # spat_params = np.array([[0, 1, 1, 0, 0]], dtype='float32') + np.exp(log_sigma) * noise
                # spat_params = np.random.multivariate_normal(np.array([1, 0, 0, 0, 1, 0]), cov_trans, (1,))
                # spat_params = np.random.multivariate_normal(np.array([-0.3, 1, 1, 0, 0]), cov_trans, (1,))
                # spat_params = mean_trans + np.sqrt(np.diagonal(cov_trans)) * noise
                # spat_params = np.random.multivariate_normal(mean_trans, cov_trans, (1, ))
                # spat_params = mean_params + noise * np.exp(log_sigma)

                spat_params = mean_params + noise
                spat_params = spat_params.astype('float32')
                # spat_params =np.array([[1, 0, 0, 0, 1, 0]], dtype='float32')
            data = data.reshape(1, n_latent)
            if spatial == 'None':
                pred = decode_fn(data)[0][0]
            else:
                pred = decode_fn(data, post_identity)[0][0]
                pred_id = decode_fn(data, identity)[0][0]
                preds_line_id += [pred_id]
            preds_line += [pred]
            if "sig" in spatial:
                pred_sig = decode_fn(data, spat_params)[0][0]
                preds_line_sig += [pred_sig]
        pred_mat += [np.concatenate(preds_line, axis=1)]
        if "sig" in spatial:
            pred_mat_sig += [np.concatenate(preds_line_sig, axis=1)]
        if spatial  != 'None':
            pred_mat_id += [np.concatenate(preds_line_id, axis=1)]
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

    if "sig" in spatial:
        manifold = np.concatenate(pred_mat_sig, axis=0)
        manifold = manifold[np.newaxis, :]
        line_sample = pred_mat_sig[0][np.newaxis, :]
        line_sample = np.concatenate(3 * [line_sample], axis=0)
        line_sample = line_sample.transpose(1, 2, 0)
        img = np.concatenate([manifold, manifold, manifold], axis=0)
        img = img.transpose(1, 2, 0)
        imsave('output/manifold_sig_' + id_ + '.jpg', img)
        imsave('output/new_sample_sig_' + id_ + '.jpg', line_sample)

    if spatial  != 'None':
        manifold = np.concatenate(pred_mat_id, axis=0)
        manifold = manifold[np.newaxis, :]
        line_sample = pred_mat_id[0][np.newaxis, :]
        line_sample = np.concatenate(3 * [line_sample], axis=0)
        line_sample = line_sample.transpose(1, 2, 0)
        img = np.concatenate([manifold, manifold, manifold], axis=0)
        img = img.transpose(1, 2, 0)
        imsave('output/manifold_ident_' + id_ + '.jpg', img)
        imsave('output/new_sample_ident_' + id_ + '.jpg', line_sample)

    # NEAREST NEIGHBORS METRICS 
    # Dist between images
    # train - test
    if save_distances:
        print("Nearest neighbors metrics computations ...")
        print("- in the image space -")


        nn = NearestNeighbors(n_neighbors=10, algorithm="brute")
        nn.fit(X_val.reshape(X_val.shape[0], -1))
        dist, _ = nn.kneighbors(X_train.reshape(X_train.shape[0], -1))
        dst_test_train = np.mean(dist)
        nn = NearestNeighbors(n_neighbors=10, algorithm="brute")
        nn.fit(X_train.reshape(X_train.shape[0], -1))
        dist, _ = nn.kneighbors(X_val.reshape(X_val.shape[0], -1))
        dst_test_train = (dst_test_train + np.mean(dist)) / 2
        print("Train-test dists:", dst_test_train, np.mean(dist))

        # gen - train
        n_gen = 10000
        z_gen = np.random.standard_normal((n_gen, n_latent)).astype("float32")
        if spatial == 'None':
            x_gen = decode_fn(z_gen)
        else:
            x_gen = decode_fn(z_gen, np.repeat(identity, n_gen, axis=0))
        dist, _ = nn.kneighbors(x_gen.reshape(X_val.shape[0], -1))
        dst_gen_train = np.mean(dist)
        nn = NearestNeighbors(n_neighbors=10, algorithm="brute")
        nn.fit(x_gen.reshape(x_gen.shape[0], -1))
        dist, _ = nn.kneighbors(X_train.reshape(X_train.shape[0], -1))
        print("Train-Gen dists:", dst_gen_train, np.mean(dist))
        dst_gen_train = (dst_gen_train + np.mean(dist)) / 2

        print("- in the embedded space -")
        z_val = encode_fn(X_val)
        nn = NearestNeighbors(n_neighbors=10, algorithm="brute")
        nn.fit(z_val)
        dist, _ = nn.kneighbors(z_train)
        dst_test_train_emb = np.mean(dist)
        nn = NearestNeighbors(n_neighbors=10, algorithm="brute")
        nn.fit(z_train)
        dist, _ = nn.kneighbors(z_val)
        dst_test_train_emb = (dst_test_train_emb + np.mean(dist)) / 2

        dist, _ = nn.kneighbors(z_gen)
        dst_gen_train_emb = np.mean(dist)
        nn = NearestNeighbors(n_neighbors=10, algorithm="brute")
        nn.fit(z_gen)
        dist, _ = nn.kneighbors(z_train)
        dst_gen_train_emb = (dst_gen_train_emb + np.mean(dist)) / 2

        print("Saving dist in embedded space ...")
        perm = np.random.permutation(X_train.shape[0])[:n_gen]
        plt.scatter(z_train[perm, 0], z_train[perm, 1], c=y_train[perm], s=5)
        plt.savefig("output/dist_train_" + id_ + ".png")
        plt.close()
        plt.scatter(z_val[:, 0], z_val[:, 1], c=y_val, s=5)
        plt.savefig("output/dist_val_" + id_ + ".png")
        plt.close()

        pca = PCA(n_components=2)
        z_train_pca = pca.fit_transform(z_train)
        plt.scatter(z_train_pca[perm, 0], z_train_pca[perm, 1], c=y_train[perm], s=5)
        plt.savefig("output/dist_train_pca_" + id_ + ".png")
        plt.close()
        z_val_pca = pca.fit_transform(z_val)
        plt.scatter(z_val_pca[:, 0], z_val_pca[:, 1], c=y_val, s=5)
        plt.savefig("output/dist_val_pca_" + id_ + ".png")
        plt.close()



    if save_encoding:
        print("Saving encoding ...")
        # encode_train = np.zeros((X_train.shape[0], n_latent))
        # y_enc_train = np.zeros(y_train.shape)
        # for i, batch in enumerate(iterate_minibatches(X_train, y_train, 100, shuffle=False)):
        #     inputs, targets = batch
        #     encode_train[(i * 100): ((i + 1) * 100), :] = encode_fn(inputs)
        #     y_enc_train[(i * 100): ((i + 1) * 100)] = targets

        # encode_val = np.zeros((X_val.shape[0], n_latent))
        # y_enc_val = np.zeros(y_val.shape)
        # for i, batch in enumerate(iterate_minibatches(X_val, y_val, 100, shuffle=False)):
        #     inputs, targets = batch
        #     encode_val[(i * 100):((i + 1) * 100), :] = encode_fn(inputs)
        #     y_enc_val[(i * 100): ((i + 1) * 100)] = targets
        encode_train = encode_fn(X_train)
        encode_val = encode_fn(X_val)
        y_enc_train = y_train
        y_enc_val = y_val
        print(encode_train.shape, encode_val.shape)

        import pandas as pd
        encode_train_df = pd.DataFrame(encode_train)
        encode_val_df = pd.DataFrame(encode_val)
        encode_train_df["y"] = y_enc_train
        encode_val_df["y"] = y_enc_val
        encode_train_df.to_csv("encoded_mnist_train_" + id_ + ".csv")
        encode_val_df.to_csv("encoded_mnist_val_" + id_ + ".csv")

    with open('file.txt', 'a') as file:
        txt = "\n n_ep: " + str(n_epoch)
        txt = txt + " b_s: " + str(batch_size)
        txt = txt + " id_" + id_
        txt = txt + " n_lat: " + str(n_latent)
        txt = txt + " n_hid: " + str(n_hidden)
        txt = txt + " n_layers: " + str(n_layers) + "\n"
        txt = txt + " score: " + "{0:.2f}".format(score)
        if save_distances:
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

    # Use run_exp to run new experiments
    # data, model(trough a module), n_latent is the dimension of
    # the general purpose latent code, n_spatial is the dimension
    # of the spatial latent code. n_hidden is the number of hidden 
    # units in a dense layer, n_layer is the number of layers in the 
    # decoder and the encoder.

    # Num of latent variable 32, 48, 64

    run_exp(data, tvae, n_epoch=num_epochs, batch_size=batch_size,
            id_='tvae-32', n_latent=26, n_hidden=256, n_layers=2,
            spatial="tvae", n_spatial=6)
    run_exp(data, tvae, n_epoch=num_epochs, batch_size=batch_size,
            id_='tvae-48', n_latent=42, n_hidden=256, n_layers=2,
            spatial="tvae", n_spatial=6)
    run_exp(data, tvae, n_epoch=num_epochs, batch_size=batch_size,
            id_='tvae-64', n_latent=58, n_hidden=256, n_layers=2,
            spatial="tvae", n_spatial=6)

    run_exp(data, tps_vae, n_epoch=num_epochs, batch_size=batch_size,
            id_='tps-tvae-32', n_latent=14, n_hidden=256, n_layers=2,
            spatial="tps_vae", n_spatial=18)
    run_exp(data, tps_vae, n_epoch=num_epochs, batch_size=batch_size,
            id_='tps-tvae-48', n_latent=30, n_hidden=256, n_layers=2,
            spatial="tps_vae", n_spatial=18)
    run_exp(data, tps_vae, n_epoch=num_epochs, batch_size=batch_size,
            id_='tps-tvae-64', n_latent=46, n_hidden=256, n_layers=2,
            spatial="tps_vae", n_spatial=18)

    # run_exp(data, conv_tvae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_='conv_tvae_3', n_latent=10, n_hidden=256, n_layers=2,
    #          spatial="tvae", n_spatial=6, save_distances=False)

    # run_exp(data, conv_vae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_='conv_vae', n_latent=20, n_hidden=256, n_layers=2,
    #          spatial="None", n_spatial=0, save_distances=False)

    # For disentanglement:
    # run_exp(data, vae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_='vae_8', n_latent=8, n_hidden=512, n_layers=1, save_encoding=True)
    # run_exp(data, vae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_='vae_20', n_latent=20, n_hidden=128, n_layers=1, save_encoding=True, save_distances=False)

    # run_exp(data, tvae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_='tvae_8', n_latent=8, n_hidden=256, n_layers=2,
    #         spatial="tvae", n_spatial=6, save_encoding=True, save_distances=False)
    # run_exp(data, tvae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_='tvae_20', n_latent=20, n_hidden=256, n_layers=2,
            # spatial="tvae", n_spatial=6, save_encoding=True, save_distances=False)

    # TVAE with estimation of prior sigma(diagonal) on u (without estimating mu)
    # run_exp(data, tvae_aff_sig, n_epoch=num_epochs, batch_size=batch_size,
    #     id_="50", n_latent=14, n_hidden=256, n_layers=2,
    #     spatial="tvae_aff_sig", n_spatial=6)

    # TVAE with estimation of prior mu, Sigma(diagonal) on u
    # run_exp(data, tvae_aff_mu_sig, n_epoch=num_epochs, batch_size=batch_size,
    #     id_="51", n_latent=14, n_hidden=256, n_layers=2,
    #     spatial="tvae_aff_mu_sig", n_spatial=6)

    # run_exp(data, tvae_aff_mu_sig_repar, n_epoch=num_epochs, batch_size=batch_size,
    #     id_="52", n_latent=14, n_hidden=256, n_layers=2,
    #     spatial="tvae_aff_mu_sig_repar", n_spatial=5)


    # TVAE with sigma estimation: tps_vae_nn_prior

    # run_exp(data, tps_vae_nn_prior, n_epoch=num_epochs, batch_size=batch_size,
    #     id_=-7, n_latent=2, n_hidden=256, n_layers=2,
    #     spatial="tps_vae_nn_prior_sig", n_spatial=5, n_hidden_prior=5)
    # run_exp(data, tps_vae_nn_prior, n_epoch=num_epochs, batch_size=batch_size,
    #     id_=-8, n_latent=2, n_hidden=256, n_layers=2,
    #     spatial="tps_vae_nn_prior_sig", n_spatial=5, n_hidden_prior=10)
    # run_exp(data, tps_vae_nn_prior, n_epoch=num_epochs, batch_size=batch_size,
    #     id_=-9, n_latent=2, n_hidden=256, n_layers=2,
    #     spatial="tps_vae_nn_prior_sig", n_spatial=5, n_hidden_prior=20)
    # run_exp(data, tps_vae_nn_prior, n_epoch=num_epochs, batch_size=batch_size,
    #     id_=-7, n_latent=2, n_hidden=256, n_layers=2,
    #     spatial="tps_vae_nn_prior_sig", n_spatial=5, n_hidden_prior=40)
    # run_exp(data, tvae_mu_sig, n_epoch=num_epochs, batch_size=batch_size,
    #     id_=-6, n_latent=15, n_hidden=256, n_layers=2,
    #     spatial="tvae_mu_sig", n_spatial=5)

    # run_exp(data, tvae_mu_sig, n_epoch=num_epochs, batch_size=batch_size,
    # id_=-6.5, n_latent=2, n_hidden=128, n_layers=2,
    # spatial="tvae_mu_sig", n_spatial=5)

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
    # run_exp(data, tps_vae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_=38, n_latent=2, n_hidden=256, n_layers=2,
    #         spatial="tps_vae", n_spatial=18)
    # run_exp(data, tps_vae, n_epoch=num_epochs, batch_size=batch_size,
    #         id_=39, n_latent=12, n_hidden=256, n_layers=2,
    #         spatial="tps_vae", n_spatial=18)
