from __future__ import division, print_function, absolute_import
import amitgroup as ag
import numpy as np
import random, collections
import scipy.sparse

BernoulliMixtureSimple = collections.namedtuple('BernoulliMixtureSimple',
                                                'log_templates log_invtemplates weights')


class BernoulliMixture(ag.util.Saveable):
    """
    Bernoulli Mixture model with an EM solver.

    Parameters
    ----------
    num_mix : int
        Number of mixture components. 
    data_mat : ndarray
        Binary data array. Can be of any shape, as long as the first axis separates samples.
        Values in the data must be either 0 or 1 for the algorithm to work. We will refer to the size of this array as ``(N, A, B, ...)``, where `N` is the number of samples, and ``(A, B, ...)`` the shape of a data sample.
    init_type : string
        Specifies the algorithm initialization.
         * `unif_rand` : Unified random. 
         * `specific` : TODO: Add explanation of this.
    
    Attributes 
    ----------
    num_mix : int
        Number of mixture components.
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
    def __init__(self,num_mix,data_mat,min_num=0, init_type='unif_rand',init_seed=0, float_type=np.float64, max_iter=50, n_init=1):
        # TODO: opt_type='expected'
        self.num_mix = num_mix
        self.float_type = float_type
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_type = init_type
        # If we're reconstructing a trained Bernoulli mixture model, then we might
        # intiailize this class without a data matrix
        self.min_num=min_num
        if data_mat is not None:
            self.num_data = data_mat.shape[0]
            self.data_shape = data_mat.shape[1:]
            # flatten data to just be binary vectors
            self.data_length = np.prod(data_mat.shape[1:])
    
            if isinstance(data_mat, np.matrix):
                pass# Let it be
                self.data_mat = data_mat
                self.not_data_mat = 1 - self.data_mat
                self.sparse = False
            if scipy.sparse.issparse(data_mat):
                pass# Let it be
                self.data_mat = data_mat
                self.sparse = True
            else:
                self.data_mat = np.asmatrix(data_mat.reshape(self.num_data, self.data_length).astype(np.uint8))
                self.not_data_mat = 1 - self.data_mat
                self.sparse = False

            # If we change this to a true bitmask, we should do ~data_mat

            self.iterations = 0
            # set the random seed
            self.seed = init_seed
            np.random.seed(self.seed)


            self.min_probability = 0.05 

        # Data sizes:
        # data_mat : num_data * data_length
        # weights : num_mix
        # work_templates : num_mix * data_length
        # affinities : num_data * num_mix


    # TODO: save_template never used!
    def run_EM(self, tol, min_probability=0.05, debug_plot=False):
        """ 
        Run the EM algorithm to specified convergence.
        
        Parameters
        ----------
        tol : float
            The tolerance gives the stopping condition for convergence. 
            If the loglikelihood decreased with less than ``tol``, then it will break the loop.
        min_probability : float
            Disallow probabilities to fall below this value, and extend below one minus this value.
        """
        #self._preload_log_templates()
            
        all_llhs = []
        all_templates = []
        all_weights = []
    
        for loop in range(self.n_init):
            # initializing weights
            self.weights = 1/self.num_mix * np.ones(self.num_mix, dtype=self.float_type)
            #self.opt_type=opt_type TODO: Not used yet.
            self.init_affinities_templates(self.init_type)

            self.min_probability = min_probability 
            loglikelihood = -np.inf
            # First E step plus likelihood computation
            self.M_step()
            new_loglikelihood = self._compute_loglikelihoods()

            if debug_plot:
                plw = ag.plot.PlottingWindow(subplots=(1, self.num_mix), figsize=(self.num_mix*3, 3))

            self.iterations = 0
            while np.isinf(loglikelihood) or np.fabs((new_loglikelihood - loglikelihood)/loglikelihood) > tol:
                if self.iterations >= self.max_iter:
                    break
                ag.info("Iteration {0}: loglikelihood {1}".format(self.iterations, loglikelihood))
                loglikelihood = new_loglikelihood
                # M-step
                self.M_step()
                # E-step
                new_loglikelihood = self._compute_loglikelihoods()
                
                self.iterations += 1

                if debug_plot and not self._plot(plw):
                    raise ag.AbortException 

            self.loglikelihood = loglikelihood
            self.set_templates()

            all_llhs.append(loglikelihood)
            all_templates.append(self.templates)
            all_weights.append(self.weights)

        best_i = np.argmax(all_llhs)
        self.templates = all_templates[best_i]
        self.weights = all_weights[best_i]
        
    def cluster_underlying_data(self,underlying_data):
        """
        U
        """
        # check that the number of data points matches the number
        # of data estimated
        assert underlying_data.shape[0] == self.num_data
        underlying_shape = underlying_data.shape[1:]
        underlying_clusters = np.dot(self.affinities.T,
                            underlying_data.reshape(self.num_data,np.prod(underlying_shape)))
        return underlying_clusters.reshape( (self.num_data,) + underlying_shape)

    def _plot(self, plw):
        if not plw.tick():
            return False 
        self.set_templates()
        for m in range(self.num_mix):
            # TODO: Fix this somehow
            if self.templates.ndim == 3:
                plw.imshow(self.templates[m], subplot=m)
            elif self.templates.ndim == 4:
                plw.imshow(self.templates[m].mean(axis=0), subplot=m)
            else:
                raise ValueError("debug_plot not supported for 5 or more dimensional data")
        return True
 
    def M_step(self):
        self.weights = np.asarray(np.mean(self.affinities,axis=0)).ravel()
        self.work_templates[:] = np.asarray(self.affinities.T * self.data_mat)
        self.work_templates /= self.num_data 
        self.work_templates /= self.weights.reshape((self.num_mix, 1))
        self.threshold_templates()
        self._preload_log_templates()

    def _preload_log_templates(self):
        self.log_templates = np.log(self.work_templates)
        self.log_invtemplates = np.log(1-self.work_templates)
        self.log_operand = np.log(self.work_templates / (1 - self.work_templates))
        self.log_sum_invtemplates = np.log(1-self.work_templates).sum(axis=1).reshape((1, -1))
        if self.sparse:
            a = self.log_invtemplates.T.sum(axis=0)
            self.log_invfill = np.tile(a, (self.num_data, 1))

    def get_bernoulli_mixture_named_tuple():
        return BernoulliMixtureSimple(log_templates=self.log_templates,
                                      log_invtemplates=self.log_invtemplates,
                                      weights=self.weights)
                                      
        
    def threshold_templates(self):
        self.work_templates = np.clip(self.work_templates, self.min_probability, 1-self.min_probability) 

    def init_affinities_templates(self,init_type):
        if init_type == 'unif_rand':
            random.seed(self.seed)
            idx = range(self.num_data)
            random.shuffle(idx)
            #self.affinities = np.zeros((self.num_data,
            #                            self.num_mix), dtype=self.float_type)
            self.work_templates = np.empty((self.num_mix,
                                       self.data_length), dtype=self.float_type)
            #self.work_templates = np.random.random((self.num_mix, self.data_length), dtype=self.float_type)
            self.affinities = np.random.random((self.num_data, self.num_mix))
            #self.affinities /= self.affinities.sum(axis=0)
            self.affinities /= np.sum(self.affinities,axis=1).reshape((self.num_data, 1))
            if 0:
                for mix_id in range(self.num_mix):
                    self.affinities[self.num_mix*np.arange(self.num_data/self.num_mix)+mix_id,mix_id] = 1.
                    aff = self.affinities[:,mix_id]
                    self.work_templates[mix_id] = np.squeeze(np.asarray(self.data_mat.T * aff.reshape((-1, 1)))) / aff.sum() 
                    self.threshold_templates()
        elif init_type == 'specific':
            random.seed(self.seed)
            idx = range(self.num_data)
            random.shuffle(idx)
            self.affinities = np.zeros((self.num_data,
                                        self.num_mix))
            self.work_templates = np.zeros((self.num_mix,
                                       self.data_length))
            for mix_id in range(self.num_mix):
                self.affinities[self.num_mix*np.arange(self.num_data/self.num_mix)[1]+mix_id,mix_id] = 1.
                self.work_templates[mix_id] = np.mean(self.data_mat[self.affinities[:,mix_id]==1],axis=0)
                self.threshold_templates()

        #self._preload_log_templates()

    def init_templates(self):
        self.work_templates = np.zeros((self.num_mix,
                                   self.data_length))
        self.templates = np.zeros((self.num_mix,
                                   self.data_length))

    def set_templates(self):
        if self.min_num > 0:
            ii=np.where(self.weights*self.num_data>self.min_num)[0]
            temp_weights=self.weights[ii]/np.sum(self.weights[ii])
            self.weights=temp_weights
            self.num_mix=ii.shape[0]
    #       self.templates = self.work_templates.reshape((self.num_mix,)+self.data_shape)
            self.work_templates=self.work_templates[ii,:]
        self.log_templates = np.log(self.work_templates)
        self.log_invtemplates = np.log(1-self.work_templates)
        self._compute_loglikelihoods()
        self.templates = self.work_templates.reshape((self.num_mix,)+self.data_shape)
        self.log_templates = np.log(self.templates)
        self.log_invtemplates = np.log(1-self.templates)


    def set_weights(self,new_weights):
        np.testing.assert_approx_equal(np.sum(new_weights),1.)
        assert(new_weights.shape==(self.num_mix,))
        self.weights = new_weights
        
    def _compute_loglikelihoods(self):
        template_logscores = self.get_template_loglikelihoods()
        loglikelihoods = template_logscores + np.log(self.weights).reshape((1,self.num_mix))
        max_vals = np.amax(loglikelihoods,axis=1)
        self.mle = max_vals

        # adjust the marginals by a value to avoid numerical
        # problems
        # TODO: Use scipy.misc.logsumexp?
        logmarginals_adj = np.sum(np.exp(loglikelihoods - max_vals.reshape((self.num_data, 1))),axis=1)
        loglikelihood = np.sum(np.exp(logmarginals_adj)) + np.sum(max_vals)
        self.affinities = np.exp(loglikelihoods-(logmarginals_adj+max_vals).reshape((self.num_data, 1)))
        self.affinities /= np.sum(self.affinities,axis=1).reshape((self.num_data, 1))
        return loglikelihood

    def compute_loglikelihood(self,datamat):
        #num_data = datamat.shape[0]
        #np.tile(self.log_templates 
        pass

    def get_template_loglikelihoods(self):
        """Calculates the log likelihood using the current templates"""
        return self.data_mat * self.log_operand.T + self.log_sum_invtemplates

    def remix(self, data):
        """
        Gives the mixture model an alternative set of input, and computes the mixture components on it.

        It is for instance appropriate to give the originals images to this function, if the mixture component originally was calculated using features of the originals.
    
        Parameters
        ----------
        data : ndarray
            The data that should be combined with the affinities (mixture weights) of the mixture model. Takes an array of any number of any shape ``(N, A', B', ...)``, where `N` is the number of samples. Notice that we have marked the other dimensions with a dash, since the size of this array does not have to equal the size of the data array used to train the model in the first place. 

        Returns
        -------
        remixed : ndarray
            The `data` averaged according to the mixture components. Will have the shape ``(num_mix, A', B', ...)``.
        """
        aff = np.asarray(self.affinities)
        return np.asarray([np.average(data, axis=0, weights=aff[:,m]) for m in range(self.num_mix)])

    def remix_iterable(self, iterable):
        """
        Similar to :func:`remix`, except takes any iterable object. This is useful if your remixing data is bigger than what your memory can hold at once, in which case
        you can create a generator object that yields each sample. 
    
        Parameters
        ----------
        iterable : iterable 
            Any iterable data structure, for instance a generator. 

        Returns
        -------
        remixed : ndarray
            The `iterable` averaged according to the mixture components.
        """
        output = None
        N = 0
        for i, edges in enumerate(iterable): 
            if output is None:
                output = np.zeros((self.num_mix,) + edges.shape) 

            for k in range(self.num_mix):
                output[k] += edges * self.affinities[i,k]
            N += 1

        output /= N * self.weights.reshape((-1,) + (1,)*(output.ndim-1))
        return output
        
    def which_component(self, sample_index):
        """
        Takes a sample index and returns which component it ended up mixing with.
        """
        return np.argmax(self.affinities[sample_index])

    def mixture_components(self):
        """
        Returns a list of which mixture component each data entry is associate with the most. 

        Returns
        -------
        components: list 
            A list of length `num_data`  where `components[i]` indicates which mixture index the `i`-th data entry belongs the most to (results should be degenerate).
        """
        return np.argmax(np.asarray(self.affinities), axis=1)

    def indices_lists(self):
        """
        Returns a list of index arrays for each mixture model.

        Returns
        -------
        lists : list
            Returns a python list of length `self.num_mix` of ndarrays. The ndarray `lists[i]` contains the indices of the data entries that are associated with the mixture component `i`.
        """
        # Probably an easier and faster way to do this
        indices = self.mixture_components()
        return [np.where(indices == i)[0] for i in range(self.num_mix)]
      
    if 0:
        def save(self, filename, save_affinities=False):
            """
            Save mixture components to a numpy npz file.
            
            Parameters
            ----------
            filename : str
                Path to filename
            save_affinities : bool
                Save ``affinities`` or not. This is an option since this will proportional to input data size, which
                can be much larger than simply the mixture templates. 
            """
            entries = dict(templates=self.templates, weights=self.weights)
            if save_affinities:
                entries['affinities'] = self.affinities
            np.savez(filename, **entries) 

    def save_to_dict(self, save_affinities=False):
        entries = dict(templates=self.templates, weights=self.weights, num_mix=self.num_mix, num_data=self.num_data, data_length=self.data_length, data_shape=self.data_shape, sparse=self.sparse)
        if save_affinities:
            entries['affinities'] = self.affinities
        return entries

    @classmethod
    def load_from_dict(cls, d):
        num_mix = d['num_mix']
        obj = cls(num_mix, None) # Has no data
        obj.templates = d['templates']
        obj.weights = d['weights']
        obj.affinities = d.get('affinities')
        obj.num_data = d['num_data']
        obj.data_length = d['data_length']
        obj.data_shape = d['data_shape']
        obj.sparse = d['sparse']
        if obj.templates.size > 0:
            obj.work_templates = obj.templates.reshape((obj.num_mix, obj.data_length))
            obj._preload_log_templates()

        #obj.log_templates = np.log(obj.templates)
        #obj.log_invtemplates = np.log(1-obj.templates)
        #self.templates = self.work_templates.reshape((self.num_mix,)+self.data_shape)
        #self.log_templates = np.log(self.templates)
        #self.log_invtemplates = np.log(1-self.templates)

        #obj.set_templates()
        #obj._preload_log_templates()
        return obj
    
