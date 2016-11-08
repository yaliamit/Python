from __future__ import division, print_function, absolute_import
import unittest, itertools

from nose.tools import assert_true
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_raises)
from scipy import stats
from amitgroup.stats import bernoullimm

from sklearn.datasets.samples_generator import make_spd_matrix

rng = np.random.RandomState(0)


def test_sample_bernoulli():
    """
    Test sample generation from mixture.sample_bernoulli
    """

    n_features, n_samples = 2, 300
    axis = 1
    mu = np.clip(rng.rand(n_features),.01,.99)
    

    samples = bernoullimm.sample_bernoulli(
        mu, n_samples=n_samples)

    assert_true(np.allclose(samples.mean(axis), mu, atol=1.3))
    

def _naive_lpbpdf(X, mu):
    # slow and naive implementation of lpbpdf
    ref = np.empty((len(X), len(mu)))
    for i, m in enumerate(mu):
        ref[:, i] = stats.bernoulli.logpmf(X, m).sum(axis=1)
    return ref

def test_lpbpdf():
    """
    test a slow and naive implementation of lmvnpdf and
    compare it to the vectorized version (mixture.lmvnpdf) to test
    for correctness
    """
    n_features, n_components, n_samples = 2, 3, 10
    mu = np.clip(rng.rand(n_components,n_features),.01,.99)
    log_inv_mu = np.log(1-mu)
    log_mu_odds = np.log(mu) - log_inv_mu
    log_inv_mu_sums = log_inv_mu.sum(-1)

    X = (rng.rand(n_samples, n_features) > .5).astype(np.uint8)

    ref = _naive_lpbpdf(X, mu)
    lpr = bernoullimm.log_product_of_bernoullis_mixture_likelihood(X, log_mu_odds, log_inv_mu_sums)
    assert_array_almost_equal(lpr, ref)



def test_BernoulliMM_attributes():
    n_components, n_features = 10, 4

    b = bernoullimm.BernoulliMM(n_components, random_state=rng)
    weights = rng.rand(n_components)
    weights = weights / weights.sum()
    means = np.clip(rng.rand(n_components, n_features),
                    .01,.99)
    log_inv_means = np.log(1-means)
    log_odds = np.log(means) - log_inv_means
    log_inv_mean_sums = log_inv_means.sum(-1)


    assert_true(b.n_components == n_components)
    

    b.weights_ = weights
    assert_array_almost_equal(b.weights_, weights)
    b.means_ = means
    assert_array_almost_equal(b.means_, means)
    b.log_inv_means_ = log_inv_means
    assert_array_almost_equal(b.log_inv_means_, log_inv_means)
    b.log_odds_ = log_odds
    assert_array_almost_equal(b.log_odds_, log_odds)
    b.log_inv_mean_sums_ = log_inv_mean_sums
    assert_array_almost_equal(b.log_inv_mean_sums_, log_inv_mean_sums)


class BernoulliMMTester():
    do_test_eval = True

    def _setUp(self):
        self.n_components = 10
        self.n_features = 400
        self.weights = rng.rand(self.n_components)
        self.weights = self.weights / self.weights.sum()
        self.means = np.clip(rng.rand(self.n_components, 
                                      self.n_features),
                             .01,.99)
        self.threshold = -0.5
        self.I = np.eye(self.n_features)

    def test_eval(self):

        b = self.model(n_components=self.n_components,
                       random_state=rng)
        # Make sure the means are far apart so responsibilities.argmax()
        # picks the actual component used to generate the observations.
        b.means_ = self.means
        b.log_odds_, b.log_inv_mean_sums_ = bernoullimm._compute_log_odds_inv_means_sums(self.means)
        b.weights_ = self.weights

        bernoulliidx = np.repeat(np.arange(self.n_components), 5)
        n_samples = len(bernoulliidx)
        X = (rng.randn(n_samples, self.n_features) <= b.means_[bernoulliidx]).astype(np.uint8)

        ll, responsibilities = b.eval(X)

        self.assertEqual(len(ll), n_samples)
        self.assertEqual(responsibilities.shape,
                         (n_samples, self.n_components))
        assert_array_almost_equal(responsibilities.sum(axis=1),
                                  np.ones(n_samples))
        assert_array_equal(responsibilities.argmax(axis=1), bernoulliidx)

    def test_sample(self, n=100):
        b = self.model(n_components=self.n_components,
                       random_state=rng)
        # Make sure the means are far apart so responsibilities.argmax()
        # picks the actual component used to generate the observations.
        b.means_ = self.means
        b.log_odds_, b.log_inv_mean_sums_ = bernoullimm._compute_log_odds_inv_means_sums(self.means)

        b.weights_ = self.weights

        samples = b.sample(n)
        self.assertEqual(samples.shape, (n, self.n_features))

    def test_train(self, params='wm'):
        b = bernoullimm.BernoulliMM(n_components=self.n_components)
        b.weights_ = self.weights
        b.means_ = self.means
        b.log_odds_, b.log_inv_mean_sums_ = bernoullimm._compute_log_odds_inv_means_sums(self.means)


        # Create a training set by sampling from the predefined distribution.
        X = b.sample(n_samples=100)
        b = self.model(n_components=self.n_components,
                       
                       random_state=rng, 
                       n_iter=1, init_params=params)
        b.fit(X)

        # Do one training iteration at a time so we can keep track of
        # the log likelihood to make sure that it increases after each
        # iteration.
        trainll = []
        for iter in range(5):
            b.params = params
            b.init_params = ''
            b.fit(X)
            trainll.append(self.score(b, X))
        b.n_iter = 10
        b.init_params = ''
        b.params = params
        b.fit(X)  # finish fitting

        delta_min = np.diff(trainll).min()
        self.assertTrue(
            delta_min > self.threshold,
            "The min nll increase is %f which is lower than the admissible"
            " threshold of %f. The likelihoods are %s."
            % (delta_min, self.threshold,  trainll))

    def test_train_degenerate(self, params='wm'):
        """ Train on degenerate data with 0 in some dimensions
        """
        # Create a training set by sampling from the predefined distribution.
        X = (rng.rand(100, self.n_features) > .5).astype(np.uint8)
        X.T[1:] = 0
        b = self.model(n_components=2, 
                       random_state=rng, n_iter=5,
                       init_params=params)
        b.fit(X)
        trainll = b.score(X)
        self.assertTrue(np.sum(np.abs(trainll / 100 / X.shape[1])) < 5)

    def test_train_1d(self, params='wm'):
        """ Train on 1-D data
        """
        # Create a training set by sampling from the predefined distribution.
        X = (rng.rand(100, 1) > .5).astype(np.uint8)
        #X.T[1:] = 0
        b = self.model(n_components=2, 
                       random_state=rng, n_iter=5,
                       init_params=params)
        b.fit(X)
        trainll = b.score(X)
        self.assertTrue(np.sum(np.abs(trainll / 100)) < 2)

    def score(self, b, X):
        return b.score(X).sum()


class TestBernoulliMMW(unittest.TestCase, BernoulliMMTester):
    model = bernoullimm.BernoulliMM
    setUp = BernoulliMMTester._setUp




def test_multiple_init():
    """Test that multiple inits performs at least as well as a single one"""
    X = (rng.rand(30, 5) > .5).astype(np.uint8)

    b = bernoullimm.BernoulliMM(n_components=2, 
                    random_state=rng, n_iter=5)
    train1 = b.fit(X).score(X).sum()
    b.n_init = 5
    out_b = b.fit(X)
    print(out_b.means_)
    train2 = out_b.score(X).sum()

    print("train2 = {0}, train1 = {1}".format(train2,train1))
    assert_true(train2 >= train1 - 1.e-2)


def test_n_parameters():
    """Test that the right number of parameters is estimated"""
    n_samples, n_dim, n_components = 7, 5, 2
    X = (rng.rand(n_samples, n_dim)>.5).astype(np.uint8)
    n_params = n_dim * n_components + n_components -1

    b = bernoullimm.BernoulliMM(n_components=n_components, 
                                random_state=rng, n_iter=1)
    b.fit(X)
    assert_true(b._n_parameters() == n_params)


# def test_aic():
#     """ Test the aic and bic criteria"""
#     n_samples, n_dim, n_components = 50, 3, 2
#     X = (rng.randn(n_samples, n_dim) > .5).astype(np.uint8)
#     SGH = 0.5 * (X.var() + np.log(2 * np.pi))  # standard gaussian entropy


#     b = bernoullimm.BernoulliMM(n_components=n_components, 
#                                 random_state=rng)
#     b.fit(X)
#     aic = 2 * n_samples * SGH * n_dim + 2 * g._n_parameters()
#     bic = (2 * n_samples * SGH * n_dim +
#                np.log(n_samples) * g._n_parameters())
#     bound = n_dim * 3. / np.sqrt(n_samples)
#     assert_true(np.abs(g.aic(X) - aic) / n_samples < bound)
#     assert_true(np.abs(g.bic(X) - bic) / n_samples < bound)


if __name__ == '__main__':
    import nose
    nose.runmodule()
