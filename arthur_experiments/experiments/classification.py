import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import sys
import pandas as pd

sys.path.append("..")
from data import load_mnist
if __name__ == '__main__':
    print('Loading dataset ...')
    (x_train, y_train), (x_test, y_test) = load_mnist("../../data")

    n_samples = x_train.shape[0]
    perm = np.random.permutation(n_samples)

    # x_train = x_train[perm[:1000]]
    # y_train = y_train[perm[:1000]]
    x_train = x_train.reshape(-1, 28**2)
    x_test = x_test.reshape(-1, 28**2)
    skip = False
    if not skip:
        print('Raw MNIST ...')
        print('Fitting ...')
        clf = LinearSVC(C=1)
        clf.fit(x_train, y_train)
        print('Train-test score: ', clf.score(x_train, y_train), clf.score(x_test, y_test))
        # np.random.seed(2017)
        # scores = cross_val_score(clf, x_train, y_train,
        #                          n_jobs=3, cv=9, verbose=1)
        # print(scores.mean(), scores.std())
        # 91.19 %  sig = 7.4e-3
        # 91.86 %
    skip = False

    skip = False
    if not skip:
        mean_train = np.mean(x_train, axis=0)
        mean_test = np.mean(x_test, axis=0)

        x_train_scaled = x_train - mean_train
        x_test_scaled = x_test - mean_test

        print("Fitting PCA ...")
        pca = PCA(n_components=8)
        pca.fit(x_train_scaled)
        z_train = pca.transform(x_train_scaled)
        z_test = pca.transform(x_test_scaled)
        print('Fitting ...')
        clf = LinearSVC(C=1)
        clf.fit(z_train, y_train)
        print('Train-test score: ', clf.score(z_train, y_train), clf.score(z_test, y_test))
        # np.random.seed(2017)
        # scores = cross_val_score(clf, z_train, y_train,
        #                          n_jobs=3, cv=9, verbose=2)
        # print(scores.mean(), scores.std())
        # 75.09 / 1.32
    skip = False

    skip = False
    if not skip:
        mean_train = np.mean(x_train, axis=0)
        mean_test = np.mean(x_test, axis=0)

        x_train_scaled = x_train - mean_train
        x_test_scaled = x_test - mean_test

        print("Fitting PCA ...")
        pca = PCA(n_components=20)
        pca.fit(x_train_scaled)
        z_train = pca.transform(x_train_scaled)
        z_test = pca.transform(x_test_scaled)
        print('Fitting ...')
        clf = LinearSVC(C=1.)
        clf.fit(z_train, y_train)
        print('Train-test score: ', clf.score(z_train, y_train), clf.score(z_test, y_test))
        # np.random.seed(2017)
        # scores = cross_val_score(clf, z_train, y_train,
        #                          n_jobs=3, cv=9, verbose=2)
        # print(scores.mean(), scores.std())
        # 85.62 1.20
    skip = False


    skip = False
    if not skip:
        print('VAE 8-dimensional latent code ...')
        print('Loading code ...')
        df_train = pd.read_csv('encoded_mnist_train_vae_8.csv')
        df_test = pd.read_csv('encoded_mnist_val_vae_8.csv')
        print((df_train['y'] == y_train).all())
        print((df_test['y'] == y_test).all())
        print(df_train.shape, df_test.shape)
        features = [str(i) for i in range(8)]
        x_train = df_train[features].values
        x_test = df_test[features].values
        # x_train = x_train[perm[:1000]]
        # y_train = y_train[perm[:1000]]

        print('Fitting ...')
        clf = LinearSVC(C=1.)
        clf.fit(x_train, y_train)
        print('Train-test score: ', clf.score(x_train, y_train), clf.score(x_test, y_test))
        # np.random.seed(2017)
        # scores = cross_val_score(clf, x_train, y_train,
        #                          n_jobs=3, cv=9, verbose=1)
        # print(scores.mean(), scores.std())
        # 87.70 % 9.3e-3
        # 88%
    skip = False

    skip = False
    if not skip:
        print('TVAE 8-dimensional latent code ...')
        print('Loading code ...')
        df_train = pd.read_csv('encoded_mnist_train_tvae_8.csv')
        df_test = pd.read_csv('encoded_mnist_val_tvae_8.csv')
        print((df_train['y'] == y_train).all())
        print((df_test['y'] == y_test).all())
        print(df_train.shape, df_test.shape)
        features = [str(i) for i in range(8)]
        x_train = df_train[features].values
        x_test = df_test[features].values
        # x_train = x_train[perm[:1000]]
        # y_train = y_train[perm[:1000]]

        print('Fitting ...')
        clf = LinearSVC(C=.01)
        clf.fit(x_train, y_train)
        print('Train-test score: ', clf.score(x_train, y_train), clf.score(x_test, y_test))
        # np.random.seed(2017)
        # scores = cross_val_score(clf, x_train, y_train,
        #                          n_jobs=3, cv=9, verbose=1)
        # print(scores.mean(), scores.std())
        # 89.30 6.8e-3
        # 88.39%
    skip = False

    skip = False
    if not skip:
        print('DTVAE 8-dimensional latent code ...')
        print('Loading code ...')
        df_train = pd.read_csv('encoded_mnist_train_dtvae_8.csv')
        df_test = pd.read_csv('encoded_mnist_val_dtvae_8.csv')
        print((df_train['y'] == y_train).all())
        print((df_test['y'] == y_test).all())
        print(df_train.shape, df_test.shape)
        features = [str(i) for i in range(8)]
        x_train = df_train[features].values
        x_test = df_test[features].values

        print('Fitting ...')
        clf = LinearSVC(C=1)
        clf.fit(x_train, y_train)
        print('Train-test score: ', clf.score(x_train, y_train), clf.score(x_test, y_test))
        # np.random.seed(2017)
        # scores = cross_val_score(clf, x_train, y_train,
        #                          n_jobs=3, cv=9, verbose=1)
        # print(scores.mean(), scores.std())
        # 93.56 6.4e-3
        # 93.87 %
    skip = False


    skip = False
    if not skip:
        print('VAE 20-dimensional latent code ...')
        print('Loading code ...')
        df_train = pd.read_csv('encoded_mnist_train_vae_20.csv')
        df_test = pd.read_csv('encoded_mnist_val_vae_20.csv')
        print((df_train['y'] == y_train).all())
        print((df_test['y'] == y_test).all())
        print(df_train.shape, df_test.shape)
        features = [str(i) for i in range(20)]
        x_train = df_train[features].values
        x_test = df_test[features].values
        # x_train = x_train[perm[:1000]]
        # y_train = y_train[perm[:1000]]

        print('Fitting ...')
        clf = LinearSVC(C=10)
        clf.fit(x_train, y_train)
        print('Train-test score: ', clf.score(x_train, y_train), clf.score(x_test, y_test))
        # np.random.seed(2017)
        # scores = cross_val_score(clf, x_train, y_train,
        #                          n_jobs=3, cv=9, verbose=1)
        # print(scores.mean(), scores.std())
        # 85.77 % 1.22 e-2
        # 86.3%
    skip = False

    skip = False
    if not skip:
        print('TVAE 20-dimensional latent code ...')
        print('Loading code ...')
        df_train = pd.read_csv('encoded_mnist_train_tvae_20.csv')
        df_test = pd.read_csv('encoded_mnist_val_tvae_20.csv')
        print((df_train['y'] == y_train).all())
        print((df_test['y'] == y_test).all())
        print(df_train.shape, df_test.shape)
        features = [str(i) for i in range(20)]
        x_train = df_train[features].values
        x_test = df_test[features].values
        # x_train = x_train[perm[:1000]]
        # y_train = y_train[perm[:1000]]

        print('Fitting ...')
        clf = LinearSVC(C=.01)
        clf.fit(x_train, y_train)
        print('Train-test score: ', clf.score(x_train, y_train), clf.score(x_test, y_test))
        # np.random.seed(2017)
        # scores = cross_val_score(clf, x_train, y_train,
        #                          n_jobs=3, cv=9, verbose=1)
        # print(scores.mean(), scores.std())
        # 90.80 % 6.6e-3
        # 90.51% 
    skip = False

    skip = False
    if not skip:
        print('DTVAE 20-dimensional latent code ...')
        print('Loading code ...')
        df_train = pd.read_csv('encoded_mnist_train_dtvae_20.csv')
        df_test = pd.read_csv('encoded_mnist_val_dtvae_20.csv')
        print((df_train['y'] == y_train).all())
        print((df_test['y'] == y_test).all())
        print(df_train.shape, df_test.shape)
        features = [str(i) for i in range(20)]
        x_train = df_train[features].values
        x_test = df_test[features].values
        # x_train = x_train[perm[:1000]]
        # y_train = y_train[perm[:1000]]

        print('Fitting ...')
        clf = LinearSVC(C=1)
        clf.fit(x_train, y_train)
        print('Train-test score: ', clf.score(x_train, y_train), clf.score(x_test, y_test))
        # np.random.seed(2017)
        # scores = cross_val_score(clf, x_train, y_train,
        #                          n_jobs=3, cv=9, verbose=1)
        # print(scores.mean(), scores.std())
        # 93.56 6.05e-3
        # 94.17
    skip = False

    skip = False
    if not skip:
        print('TPS-TVAE 8-dimensional latent code ...')
        print('Loading code ...')
        df_train = pd.read_csv('encoded_mnist_train_tps_vae_8_class.csv')
        df_test = pd.read_csv('encoded_mnist_val_tps_vae_8_class.csv')
        print((df_train['y'] == y_train).all())
        print((df_test['y'] == y_test).all())
        print(df_train.shape, df_test.shape)
        features = [str(i) for i in range(8)]
        x_train = df_train[features].values
        x_test = df_test[features].values
        # x_train = x_train[perm[:1000]]
        # y_train = y_train[perm[:1000]]

        print('Fitting ...')
        clf = LinearSVC(C=1)
        clf.fit(x_train, y_train)
        print('Train-test score: ', clf.score(x_train, y_train), clf.score(x_test, y_test))
        # np.random.seed(2017)
        # scores = cross_val_score(clf, x_train, y_train,
        #                          n_jobs=3, cv=9, verbose=2)
        # print(scores.mean(), scores.std())
        # 88.42 0.58
    skip = False

    skip = False
    if not skip:
        print('TPS-TVAE 20-dimensional latent code ...')
        print('Loading code ...')
        df_train = pd.read_csv('encoded_mnist_train_tps_vae_20_class.csv')
        df_test = pd.read_csv('encoded_mnist_val_tps_vae_20_class.csv')
        print((df_train['y'] == y_train).all())
        print((df_test['y'] == y_test).all())
        print(df_train.shape, df_test.shape)
        features = [str(i) for i in range(20)]
        x_train = df_train[features].values
        x_test = df_test[features].values
        # x_train = x_train[perm[:1000]]
        # y_train = y_train[perm[:1000]]

        print('Fitting ...')
        clf = LinearSVC(C=1)
        clf.fit(x_train, y_train)
        print('Train-test score: ', clf.score(x_train, y_train), clf.score(x_test, y_test))
        # np.random.seed(2017)
        # scores = cross_val_score(clf, x_train, y_train,
        #                          n_jobs=3, cv=9, verbose=2)
        # print(scores.mean(), scores.std())
        # 94.65 0.46
    skip = False

