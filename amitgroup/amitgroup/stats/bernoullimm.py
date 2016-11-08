from __future__ import division, print_function, absolute_import
from sklearn.utils.extmath import logsumexp
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator
import numpy as np
import random, collections
import scipy.sparse


# Author: Gustav Larsson
#         Mark Stoehr <stoehr.mark@gmail.com>
#



EPS = np.finfo(float).eps

def log_product_of_bernoullis_mixture_likelihood(X, log_odds, log_inv_mean_sums):
    """Log likelihood function for the product of Bernoullis
    mixture distribution
    """
    return np.dot(X,log_odds.T) + log_inv_mean_sums


def sample_bernoulli(mean, n_samples=1,
                    random_state=None,
                    data_type=np.uint8):
    """Generate random samples from a Bernoulli distribution.

    Parameters
    ----------
    mean : array_like, shape (n_features,)
        Mean of the distribution.

    n_samples : int, optional
        Number of samples to generate. Defaults to 1.

    Returns
    -------
    X : array, shape (n_features, n_samples)
        Randomly generated sample
    """
    rng = check_random_state(random_state)
    n_dim = len(mean)
    rand = rng.rand(n_dim, n_samples)
    if n_samples == 1:
        rand.shape = (n_dim,)

    return (rand.T > mean).T.astype(data_type)




class BernoulliMM(BaseEstimator):
    """
    Bernoulli Mixture model with an EM solver.

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance

    min_prob : float, optional
        Floor for the minimum probability

    thresh : float, optional
        Convergence threshold.

    n_iter : float, optional
        Number of EM iterations to perform

    n_init : int, optional
        Number of random initializations to perform with
        the best kept.

    params : string, optional
        Controls which parameters are updated during training.
        If 'w' is in the string then the weights are updated,
        and if 'm' is in the string then the means are updated.
        The default is 'wm'.

    init_params : string, optional
        Controls which parameters are updated during initialization.
        If 'w' is in the string then the weights are updated,
        and if 'm' is in the string then the means are updated.
        The default is 'wm'.

    float_type : numpy type, optional
        What float type to use for the parameter arrays.

    Attributes
    ----------
    `weights_` : array, shape (`n_components`,)
        Stores the mixing weights for each component

    `means_` : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    `converged_` : bool
        True when convergence was reached in fit(), False otherwise.

    data_mat : ndarray
        Binary data array. Can be of any shape, as long as the first axis separates samples.
        Values in the data must be either 0 or 1 for the algorithm to work. We will refer to the size of this array as ``(N, A, B, ...)``, where `N` is the number of samples, and ``(A, B, ...)`` the shape of a data sample.
    init_type : string
        Specifies the algorithm initialization.
         * `unif_rand` : Unified random.
         * `specific` : TODO: Add explanation of this.

    Attributes
    ----------

    r

    num_data : int
        Number of data entries.
    data_length : int
        Length of data flattened.
    iterations : int
        Number of iterations from the EM. Will be set after calling :py:func:`run_EM`.
    templates : ndarray
        Mixture components templates. Array of shape ``(num_mix, A, B, ...)``, where ``(A, B, ...)`` is the shape of a single data entry.
    work_templates : ndarray
        Flattened mixture components templates. Array of shape ``(num_mix, data_length)``.
    weights : ndarray
        The probabilities of drawing from a certain mixture component. Array of length ``num_mix``.
    affinities : ndarray
        The contribution of each original data point to each mixture component. Array of shape ``(num_data, num_mix)``.
    init_seed : integer
        Seed for randomness.

    Examples
    --------
    Create a mixture model with 2 mixture componenents.

    >>> import amitgroup as ag
    >>> import numpy as np
    >>> data = np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1]])
    >>> mixture = ag.stats.BernoulliMixture(2, data)

    Run the algorithm until specified tolerance.

    >>> mixture.run_EM(1e-3)

    Display the mixture templates and the corresponding weights.

    >>> mixture.templates # doctest: +SKIP
    array([[ 0.95      ,  0.95      ,  0.50010438],
           [ 0.05      ,  0.05      ,  0.95      ]])
    >>> mixture.weights # doctest: +SKIP
    array([ 0.66671347,  0.33328653])

    Display the latent variable, describing what combination of mixture components
    a certain data frame came from:

    >>> mixture.affinities # doctest: +SKIP
    array([[  9.99861515e-01,   1.38484719e-04],
           [  2.90861524e-03,   9.97091385e-01],
           [  9.97376426e-01,   2.62357439e-03]])

    """
    def __init__(self, n_components=1,
                 random_state=None, thresh=1e-6, min_prob=1e-2, min_num=30,
                 n_iter=100,tol=1e-6, n_init=1, params='wm', init_params='wm',blocksize=0,
                 float_type=np.float64,
                 binary_type=np.uint8, verbose=False):
        # TODO: opt_type='expected'
        self.n_components = n_components
        self.thresh = thresh
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params
        self.float_type = float_type
        self.binary_type = binary_type
        self.min_prob = min_prob
        self.min_num = 30
        self.verbose=verbose
        # blocksize controls whether we do the likelihood computation in blocks to prevent memory blowup
        self.blocksize = blocksize
        if self.n_init < 1:
            raise ValueError('BernoulliMM estimation requires at least one run')

        self.weights_ = np.ones(self.n_components,dtype=self.float_type)/ self.n_components
        self.converged_ = False



        # # If we're reconstructing a trained Bernoulli mixture model, then we might
        # # intiailize this class without a data matrix
        # if data_mat is not None:
        #     self.num_data = data_mat.shape[0]
        #     self.data_shape = data_mat.shape[1:]
        #     # flatten data to just be binary vectors
        #     self.data_length = np.prod(data_mat.shape[1:])

        #     if isinstance(data_mat, np.matrix):
        #         pass# Let it be
        #         self.data_mat = data_mat
        #         self.not_data_mat = 1 - self.data_mat
        #         self.sparse = False
        #     if scipy.sparse.issparse(data_mat):
        #         pass# Let it be
        #         self.data_mat = data_mat
        #         self.sparse = True
        #     else:
        #         self.data_mat = np.asmatrix(data_mat.reshape(self.num_data, self.data_length).astype(np.uint8))
        #         self.not_data_mat = 1 - self.data_mat
        #         self.sparse = False

        #     # If we change this to a true bitmask, we should do ~data_mat

        #     self.iterations = 0
        #     # set the random seed
        #     self.seed = init_seed
        #     np.random.seed(self.seed)


        #     self.min_probability = 0.05

        #     # initializing weights
        #     self.weights = 1./num_mix * np.ones(num_mix, dtype=self.float_type)
        #     #self.opt_type=opt_type TODO: Not used yet.
        #     self.init_affinities_templates(init_type)

        # Data sizes:
        # data_mat : num_data * data_length
        # weights : num_mix
        # work_templates : num_mix * data_length
        # affinities : num_data * num_mix




    def eval(self, X):
        """Evaluate the model on data

        Compute the log probability of X under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of X.

        Parameters
        ----------
        X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob: array_like, shape (n_samples,)
            Log probabilities of each data point in X
        responsibilities: array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError('the shape of X  is not compatible with self')

        if self.blocksize > 0:
            logprob = np.zeros(X.shape[0],dtype=self.float_type)
            responsibilities = np.zeros((X.shape[0],self.n_components),dtype=self.float_type)
            block_id = 0
            if self.verbose:
                print("Running block multiplication")

            for block_id in range(0,X.shape[0],self.blocksize):
                blockend = min(X.shape[0],block_id+self.blocksize)
                lpr = (log_product_of_bernoullis_mixture_likelihood(X[block_id:blockend], self.log_odds_,
                                                            self.log_inv_mean_sums_)
               + np.log(self.weights_))
                logprob[block_id:blockend] = logsumexp(lpr, axis=1)
                responsibilities[block_id:blockend] = np.exp(lpr - (logprob[block_id:blockend])[:, np.newaxis])
        else:
            lpr = (log_product_of_bernoullis_mixture_likelihood(X, self.log_odds_,
                                                            self.log_inv_mean_sums_)
               + np.log(self.weights_))

            logprob = logsumexp(lpr, axis=1)
            responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def score(self, X):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        """
        logprob, _ = self.eval(X)
        return logprob

    def predict(self, X):
        """Predict label for data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = (n_samples,)
        """
        logprob, responsibilities = self.eval(X)
        return responsibilities.argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior probability of data under each Gaussian
        in the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        responsibilities : array-like, shape = (n_samples, n_components)
            Returns the probability of the sample for each Gaussian
            (state) in the model.
        """
        data_shape = X.shape[1:]
        # flatten data to just be binary vectors
        data_length = np.prod(data_shape)
        if len(data_shape) > 1:
            logprob, responsibilities = self.eval(X.reshape(X.shape[0],  data_length))
        else:
            logprob, responsibilities = self.eval(X)
        return responsibilities

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples
        """
        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)
        weight_cdf = np.cumsum(self.weights_)

        X = np.empty((n_samples, self.means_.shape[1]))
        rand = random_state.rand(n_samples)
        # decide which component to use for each sample
        comps = weight_cdf.searchsorted(rand)
        # for each component, generate all needed samples
        for comp in range(self.n_components):
            # occurrences of current component in X
            comp_in_X = (comp == comps)
            # number of those occurrences
            num_comp_in_X = comp_in_X.sum()
            if num_comp_in_X > 0:
                X[comp_in_X] = sample_bernoulli(
                    self.means_[comp],
                    num_comp_in_X, random_state=random_state,
                    data_type=self.binary_type).T
        return X

    def set_means_weights(self,means,weights):
        """
        Set the means and the weights of the model so that one can
        load a model from being saved.
        """

        self.means_ = means
        self.weights_ = weights
        self.log_odds_, self.log_inv_mean_sums_ = _compute_log_odds_inv_means_sums(self.means_)

    def fit(self, X):
        """
        Run the EM algorithm to specified convergence.

        Parameters
        ----------
        X : array_like, shape (n,) + d
            List of data points assumed that the dimensions are such that
            `np.prod(X.shape[1:])==n_features`
        """
        random_state = check_random_state(self.random_state)
        X = np.asarray(X, dtype=self.binary_type)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        data_shape = X.shape[1:]
        # flatten data to just be binary vectors
        data_length = np.prod(data_shape)
        if len(data_shape) > 1:
            X = X.reshape(X.shape[0],  data_length)

        if X.shape[0] < self.n_components:
            raise ValueError(
                'BernoulliMM estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        inv_X = 1- X
        max_log_prob = -np.infty

        # if debug_plot:
        #     plw = ag.plot.PlottingWindow(subplots=(1, self.num_mix), figsize=(self.num_mix*3, 3))

        for cur_init in range(self.n_init):
            if self.verbose:
                print("Current parameter initialization: {0}".format(cur_init))


            if 'm' in self.init_params or not hasattr(self,'means_'):
                if self.verbose:
                    print("Initializing means")
                indices = np.arange(X.shape[0])
                random_state.shuffle(indices)
                self.means_ = np.array(tuple(
                    np.clip(X[indices[i::self.n_components]].mean(0),
                            self.min_prob,
                            1-self.min_prob)
                    for i in range(self.n_components)))

                self.log_odds_, self.log_inv_mean_sums_ = _compute_log_odds_inv_means_sums(self.means_)

            if 'w' in self.init_params or not hasattr(self,'weights_'):
                if self.verbose:
                    print("Initializing weights")

                self.weights_ = np.tile(1.0 / self.n_components,
                                        self.n_components)





            log_likelihood = []
            self.iterations = 0
            self.converged_ = False
            for i in range(self.n_iter):
                # Expectation Step
                curr_log_likelihood, responsibilities = self.eval(X)
                log_likelihood.append(curr_log_likelihood.sum())
                if self.verbose:
                    print("Iteration {0}: loglikelihood {1}".format(i, log_likelihood[-1]))

                # check for convergence
                if i > 0 and abs(log_likelihood[-1] - log_likelihood[-2])/abs(log_likelihood[-2]) < \
                   self.thresh:
                    self.converged_ = True
                    break

                # ag.info("Iteration {0}: loglikelihood {1}".format(self.iterations, loglikelihood))
                # maximization step
                self._do_mstep(X,
                               responsibilities,
                               self.params,
                               self.min_prob)



            if self.n_iter:
                if log_likelihood[-1] > max_log_prob:
                    if self.verbose:
                        print("updated best params for {0}".format(self.score(X).sum()))
                    max_log_prob = log_likelihood[-1]
                    best_params = {'weights': self.weights_,
                                   'means' : self.means_}



        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")

        if len(data_shape) > 1:
            X = X.reshape(*( (X.shape[0],) + data_shape))

        if self.n_iter:
            self.means_ = best_params['means']
            self.log_odds_, self.log_inv_mean_sums_ = _compute_log_odds_inv_means_sums(self.means_)
            self.weights_ = best_params['weights']

        return self


    def _do_mstep(self, X, responsibilities, params, min_prob=1e-7):
        """ Perform the Mstep of the EM algorithm and return the class weights
        """
        weights = responsibilities.sum(axis=0)

        if self.blocksize > 0:
            weighted_X_sum=np.zeros((weights.shape[0],X.shape[1]),dtype=self.float_type)

            if self.verbose:
                print("Running block multiplication for mstep")

            for blockstart in range(0,X.shape[0],self.blocksize):
                blockend=min(X.shape[0],blockstart+self.blocksize)
                res = responsibilities[blockstart:blockend].T
                weighted_X_sum += np.dot(res,X[blockstart:blockend])

        else:
            weighted_X_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)
        if 'w' in params:
            self.weights_ = (weights / (weights.sum() + 10 * EPS) + EPS)

        if 'm' in params:
            self.means_ = np.clip(weighted_X_sum * inverse_weights,min_prob,1-min_prob)
            self.log_odds_, self.log_inv_mean_sums_ = _compute_log_odds_inv_means_sums(self.means_)


        return weights



    def _n_parameters(self):
        """Return the number of free parameters in the model"""
        ndim = self.means_.shape[1]
        mean_params = ndim * self.n_components
        return int(mean_params + self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        bic: float (the lower the better)
        """
        return (-2 * self.score(X).sum() +
                self._n_parameters() * np.log(X.shape[0]))


    def aic(self, X):
        """Akaike information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        aic: float (the lower the better)
        """
        return - 2 * self.score(X).sum() + 2 * self._n_parameters()



    def cluster_underlying_data(self,Z,X=None,
                                responsibilities=None):
        """In cases where binary codes for underlying data are clustered
        and we wish to recover clusters from the underlying data based
        on the clustering of binary codes

        Parameters
        ----------
        Z : array of shape(n_samples, k)
            List of k-dimensional data points.  Each row
            corresponds to a single data point.  X is the binary codes for these
            data, or the responsibilities are a posterior distribution over the classes
            that generated the data.


        X : array of shape(n_samples, n_dimensions), optional
            List of n_dimensions-dimensional data points.  Each row
            corresponds to a single data point. Should be defined
            if responsibilities is None.  Should be binary data

        responsibilities : array of shape(n_samples, n_components)
            Should be defined if X is None, posterior distribution over
            the n_components for each data point.

        Returns

        """
        # check that the number of data points matches the number
        # of data estimated
        if X is None:
            if responsibilities is None:
                raise RuntimeError("no binary data provided")
        else:
            responsibilities = self.predict_proba(X)

        responsibilities = responsibilities.T

        underlying_clusters = np.dot(responsibilities,
                                     Z.reshape(Z.shape[0],np.prod(Z.shape[1:]))) / np.lib.stride_tricks.as_strided(responsibilities.sum(1),
                                                                                                                   shape=(responsibilities.shape[0],
                                                                                                                          np.prod(Z.shape[1:])),
                                                                                                                          strides=(responsibilities.strides[0],0))
        return underlying_clusters





def _compute_log_odds_inv_means_sums(means):
    """Compute the log odds, and the sums over the log inverse means
    to enable fast likelihood computation
    """
    log_inv_means = np.log(1-means)
    return np.log(means) - log_inv_means, log_inv_means.sum(-1)
