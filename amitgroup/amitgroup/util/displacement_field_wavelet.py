from __future__ import absolute_import
from __future__ import division

import amitgroup as ag
import numpy as np
from copy import deepcopy
from .displacement_field import DisplacementField
from .interp2d import interp2d

from amitgroup.util import wavelet

class DisplacementFieldWavelet(DisplacementField):
    """
    Displacement field using wavelets.
    
    This class requires the package `PyWavelets <http://www.pybytes.com/pywavelets/>`_.
    
    Refer to :class:`DisplacementField` for interface documentation.

    Parameters
    ----------
    shape : tuple
        Size of the displacement field.
    wavelet : string
        Specify wavelet type, ``'db1'`` (D2) to ``'db20'`` (D40).
    penalty : float
        Coefficient signifying the size of the prior. Higher means less deformation.
        This is only needed if the derivative is needed.
    rho : float
        A high value penalizes the prior for higher coarse-to-fine coefficients more.
        This is only needed if the derivative is needed.
    """
    def __init__(self, shape, wavelet='db2', rho=2.0, penalty=1.0, means=None, variances=None, level_capacity=None):
        #super(DisplacementFieldWavelet, self).__init__(shape)
        assert means is None or means.ndim == 3
        assert variances is None or variances.ndim == 3
        
        self.wavelet = wavelet 
        self.mode = 'per'
        self.shape = shape
        self.prepare_shape()
        self.rho = rho 
        self.penalty = penalty
        #biggest = self.scriptNs[-1]        
        self.u = None
        self.full_size_means = means
        self.full_size_variances = variances

        self.reset(level_capacity)


    def reset(self, level_capacity):
        if level_capacity is None:
            self.level_capacity = self.levels
        else:
            self.level_capacity = level_capacity
        N = 1 << self.level_capacity 
        self.ushape = (2, N, N)

        # We divide the penalty, since the raw penalty is the ratio
        # of the variance between the coefficients and the loglikelihood.
        # It is more natural to want the variance between how much the 
        # the coefficients can create a deformation in space instead, which
        # implies an adjustment of 2**self.levels for the s.d. We take
        # the square of this since we're dealing with the variance. 
        # Notice: Penalty is only applicable if means and variances are not set manually
        if self.penalty:
            # Removed for now, since the penalty is pretty arbitrary anyway
            self.penalty_adjusted = self.penalty# / 4**self.levels 

        if self.full_size_means is not None:
            self.mu = self.full_size_means[:,:N,:N]
        else:
            self.mu = np.zeros(self.ushape)

        if self.full_size_variances is not None:
            self.lmbks = 1/self.full_size_variances[:,:N,:N]
        else:
            self._init_default_lmbks()

        self._init_u()

    @classmethod
    def shape_for_size(cls, size, level_capacity=np.inf):
        # TODO: Is this function used? It won't play nice if level_capacity has its default value.
        N = 1 << level_capacity
        return (2, N, N)

    def _init_u(self):
        new_u = np.copy(self.mu)
        if self.u is not None:
            # Resizes the coefficients, and fills with self.mu
            A, B = min(new_u.shape[1], self.u.shape[1]), min(new_u.shape[2], self.u.shape[2])
            new_u[:,:A,:B] = self.u[:,:A,:B]

        self.u = new_u

    @classmethod
    def make_lambdas(cls, shape, levels=None, eta=1.0, rho=1.0):
        if levels is None:
            levels = int(np.log2(max(shape)))
        N = 1 << levels
        lambdas = np.zeros((N, N))
        for level in range(levels, -1, -1):
            S = 1 << level 
            lambdas[:S,:S] = eta * 2.0**(rho * level)
        return lambdas
        

    def _init_default_lmbks(self):
        self.lmbks = np.zeros(self.ushape)
        for level in range(self.levels, -1, -1):
            N = 2**level
            self.lmbks[:,:N,:N] = self.penalty_adjusted * 2.0**(self.rho * level)
    
        #print self.lmbks

    def set_flat_u(self, flat_u, level):
        """
        Sets `u` from a flattened array of a subset of `u`.
        
        The size of the subset is determined by level. The rest of `u` is filled with zeros.
        """
        assert level <= self.level_capacity, "Please increase coefficient capacity for this level"
        # First reset
        # TODO: This might not be needed either
        self.u.fill(0.0)
        #shape = self.coef_shape(level)
        N = 1 << level
        # TODO: Should not need 2*N*N
        self.u.shape, flat_u.shape, N
        self.u[:,:N,:N] = flat_u[:2*N*N].reshape((2, N, N))

    def prepare_shape(self):
        side = max(self.shape)
        self.levels = int(np.log2(side))
        self.levelshape = tuple(map(int, map(np.log2, self.shape)))
        #self.scriptNs = map(len, pywt.wavedec(np.zeros(side), self.wavelet, level=self.levels, mode=self.mode))

    def deform_x(self, x0, x1, last_level=np.inf):
        last_level = min(last_level, self.level_capacity)
        Ux0, Ux1 = self.invtransform(x0, x1, last_level)
        return x0+Ux0, x1+Ux1

    def deform_map(self, x, y, last_level=np.inf):
        last_level = min(last_level, self.level_capacity)
        return self.invtransform(x, y, last_level) 

    def transform(self, f, level):
        """
        Forward transform of the wavelet.
        """ 
        new = np.empty(self.ushape)
        S = 1 << level
        # TODO: Slicing should not be necessary
        new[0,:S,:S] = ag.util.wavelet.wavedec2(f[0], self.wavelet, level, shape=self.shape)
        new[1,:S,:S] = ag.util.wavelet.wavedec2(f[1], self.wavelet, level, shape=self.shape)
        return new 

    # TODO: last_level not used
    def invtransform(self, x, y, last_level=np.inf):
        """See :func:`DisplacementField.deform_map`"""
        Ux = ag.util.wavelet.waverec2(self.u[0], self.wavelet, shape=self.shape)
        Uy = ag.util.wavelet.waverec2(self.u[1], self.wavelet, shape=self.shape)
        return Ux, Uy 

    def deform(self, F, levels=np.inf):
        """See :func:`DisplacementField.deform`"""
        im = np.zeros(F.shape)

        x0, x1 = self.meshgrid()
        z0, z1 = self.deform_x(x0, x1, levels)
        im = interp2d(z0, z1, F)
        return im
    
    def abridged_u(self, levels=None):
        #return self.u[:,:self.flat_limit(last_level)]
        S = 1 << levels 
        return self.u[:,:S,:S].copy()

    def coef_shape(self, last_level=None):
        return (self.ushape[0], self.flat_limit(last_level))

    def logprior(self, last_level=None):
        N = None if last_level is None else 1 << last_level
        return -(self.lmbks * (self.u - self.mu)**2)[:,:N,:N].sum() / 2

    def logprior_derivative(self, last_level=None):
        N = None if last_level is None else 1 << last_level
        ret = (-self.lmbks * (self.u - self.mu))[:,:N,:N]
        return ret

    def sum_of_coefficients(self, last_level=None):
        # Return only lmbks[0], because otherwise we'll double-count every
        # value (since they are the same)
        return self.lmbks[0,:self.flat_limit(last_level)].sum()

    def number_of_coefficients(self, levels=None):
        return self.ushape[1]

    def copy(self):
        return deepcopy(self) 

    def flat_limit(self, last_level=None):
        # TODO: Come up with better name, and maybe place
        return None if last_level is None else _flat_start(last_level+1, 0, self.levelshape)

    def randomize(self, sigma=0.01, rho=2.5, start_level=1, levels=3):
        """
        Randomly sets the coefficients up to a certain level by sampling a Gaussian. 
        
        Parameters
        ----------  
        sigma : float
            Standard deviation of the Gaussian. The `sigma` is adjusted to a normalized image
            scale and not the scale of coefficient values (nor pixels). This means that setting `sigma` to 1, the standard
            deviation is the same size as the image, which is a lot. A more appropriate value is
            thus 0.01.
        rho : float
            A value higher than 1, will cause more dampening for higher coefficients, which will
            result in a smoother deformation.
        levels: int
            Number of levels that should be randomized. The levels above will be set to zero. For a funny-mirror-type deformation, this should be limited to about 3.

        Examples
        --------
        >>> import amitgroup as ag
        >>> import matplotlib.pylab as plt

        Generate 9 randomly altered faces.
        
        >>> face = ag.io.load_example('faces')[0]
        >>> imdef = ag.util.DisplacementFieldWavelet(face.shape, 'db8')
        >>> ag.plot.images([imdef.randomize(0.1).deform(face) for i in range(9)])
        >>> plt.show()
        """
        # Reset all values first
        self.u.fill(0.0)
    
        end_level = min(self.levels+1, start_level+levels)
        for q in range(2):
            for level in range(end_level, start_level-1, -1):
                N = 1 << level

                # First of all, a coefficient of 1, will be shift the image 1/2**self.levels, 
                # so first we have to adjust for that.
                # Secondly, higher coefficient should be adjusted by roughly 2**-s, to account
                # for the different amplitudes of a wavelet basis (energy-conserving reasons).
                # Finally, we might want to dampen higher coefficients even further, to create
                # a smoother image. This is done by rho.
                adjust = 2.0**(self.levels - rho * max(level-1, 0))

                self.u[:,:N,:N] = np.random.normal(0.0, sigma, (2, N, N)) * adjust
        return self

    def ilevels(self):
        for level in range(self.levels+1):
            alphas = 1 if level == 0 else 3
            yield level, (alphas,)+_levels2shape(self.levelshape, level)

    def print_lmbks(self, last_level=np.inf):
        for level, (alphas, N, M) in self.ilevels():
            if level == last_level:
                break

    def print_u(self, last_level=np.inf):
        for level, (alphas, N, M) in self.ilevels():
            if level == last_level:
                break

    # TODO: The name 'u' for the coefficients is congruent with the book, 
    #  but a bit confusing for other people. Change.
    def ulevel(self, level):
        alphas = 1 if level == 0 else 3
        size = _levels2shape(self.levelshape, level)
        #TODO: return self.u[:,level,:alphas,:size[0],:size[1]]

    def lmbk_level(self, level):
        alphas = 1 if level == 0 else 3
        size = _levels2shape(self.levelshape, level)
        return self.lmbks[:,level,:alphas,:size[0],:size[1]]
