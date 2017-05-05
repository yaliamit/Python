import sys
sys.path.append("../..")

from data import load_mnist
import numpy as np
import time
from PIL import Image

n_latent = 2
id_ = str(1)

print("Loading data...")

(X_train, y_train), (X_val, y_val) = load_mnist("../../data")

prod_shape = np.prod(X_train.shape[1:])
X_train = X_train.reshape(-1, prod_shape)
X_val = X_val.reshape(-1, prod_shape)

mean_train = np.mean(X_train, axis=0)
mean_val = np.mean(X_val, axis=0)
std_train = 1  # np.std(X_train, axis=0)
std_val = 1  # np.std(X_val, axis=0) /!\ many pixels have zero std 


X_train_scaled = (X_train - mean_train) / std_train
X_val_scaled = (X_val - mean_val) / std_val

from sklearn.decomposition import PCA


print("Fitting model ...")
model = PCA(n_components=n_latent)
model.fit(X_train_scaled)

from sklearn.metrics import log_loss

reconstructed_train = model.inverse_transform(model.transform(X_train_scaled))
reconstructed_val = model.inverse_transform(model.transform(X_val_scaled))

reconstructed_train = reconstructed_train * std_train + mean_train
reconstructed_val = reconstructed_val * std_train + mean_val

eps = 1e-6
reconstructed_train = np.clip(reconstructed_train, eps, 1 - eps)
reconstructed_val = np.clip(reconstructed_val, eps, 1 - eps)

train_score = X_train*np.log(reconstructed_train) + (1 - X_train) * np.log(1 - reconstructed_train)
val_score = X_val*np.log(reconstructed_val) + (1 - X_val) * np.log(1 - reconstructed_val)
train_score = train_score.sum() / X_train.shape[0]
val_score = val_score.sum() / X_val.shape[0]


print("train score", train_score, 'val_score', val_score)

z_train = model.transform(X_train_scaled)

print("Z1 ", np.mean(z_train[:, 0]), np.std(z_train[:, 0]))
print("Z2 ", np.mean(z_train[:, 1]), np.std(z_train[:, 1]))
# print(np.corrcoef(np.transpose(z_train)))

print("Saving samples ...")

n = 10
decoded_imgs = model.inverse_transform(model.transform(X_val_scaled[:n]))
decoded_imgs = decoded_imgs * std_val + mean_val
decoded_imgs = np.clip(decoded_imgs, 0, 1)
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

x = np.linspace(1 / n, 1 - 1 / n, n)
y = np.linspace(1 / n, 1 - 1 / n, n)


x = np.linspace(z_train_m_0 - 3 * z_train_sd_0, z_train_m_0 + 3 * z_train_sd_0, n)
y = np.linspace(z_train_m_1 - 3 * z_train_sd_1, z_train_m_1 + 3 * z_train_sd_1, n)


pred_mat = []
for k, i in enumerate(x):
    preds_line = []
    for m, j in enumerate(y):
        from scipy.stats import norm
        if n_latent == 2:
            true_i = norm.ppf(i)
            true_j = norm.ppf(j)
            data = np.array([i, j], dtype='float32')
        else:
            data = norm.rvs(size=n_latent).astype('float32')
        data = data.reshape(1, n_latent)
        pred = model.inverse_transform(data)[0] + mean_train
        pred = np.clip(pred, 0, 1)
        pred = pred.reshape(28, 28)
        preds_line += [pred]
    pred_mat += [np.concatenate(preds_line, axis=0)]
manifold = np.concatenate(pred_mat, axis=1)

manifold = manifold[np.newaxis, :]
print(manifold.shape)

img = np.concatenate([manifold, manifold, manifold], axis=0)
img = img.transpose(1, 2, 0)
imsave('output/manifold_'+id_+'.jpg', img)

save_encode = False
if save_encode:
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
