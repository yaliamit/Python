from __future__ import division

import math
import numpy as np
import amitgroup as ag
import amitgroup.util

def _powerof2(v):
    # Does not handle 0, but that's not valid for image deformations anyway
    return (v & (v-1)) == 0 

def _cost(u, imdef, F, I, delF, x, y, level, llh_variances):
    """Calculate the cost."""
    imdef.set_flat_u(u, level)

    dx = 1/np.prod(F.shape)
    z0, z1 = imdef.deform_x(x, y, level) 
    
    # Interpolate F at deformation
    Fzs = ag.util.interp2d(z0, z1, F)

    # Cost
    terms = Fzs - I
    llhs = terms**2
    #loglikelihood = -(terms**2).sum() / 2.0 #  * dx
    llhs /= llh_variances
    loglikelihood = -llhs.sum()/2.0
    logprior = imdef.logprior(level)

    cost = -logprior - loglikelihood
    return cost

def _cost_deriv(u, imdef, F, I, delF, x, y, level, llh_variances):
    """Calculate the derivative of the cost."""
    imdef.set_flat_u(u, level)

    dx = 1/np.prod(F.shape)
    z0, z1 = imdef.deform_x(x, y, level) 

    # Interpolate F at zs
    Fzs = ag.util.interp2d(z0, z1, F)

    # Interpolate delF at zs 
    delFzs = ag.util.interp2d(z0, z1, delF, fill_value=0.0)

    terms = Fzs - I

    W = np.empty((2,) + terms.shape)     
    for q in range(2):
        W[q] = delFzs[q] * terms
        W[q] /= llh_variances
    #W *= dx
    vqks = imdef.transform(W, level)
    N = 2**level
    ret = (-imdef.logprior_derivative() + vqks)[:,:N,:N].flatten()
    return ret
    
def image_deformation(F, I, last_level=3, penalty=1.0, rho=2.0, wavelet='db2', tol=1e-5, \
                      maxiter=5, start_level=1, means=None, variances=None, llh_variances=1.0, debug_plot=False):
    """
    Deforms an a prototype image `F` into a data image `I` using a Daubechies wavelet basis and maximum a posteriori. 

    Parameters
    ----------
    F : ndarray
        Prototype image. Array of shape ``(W, H)`` with normalized intensitites. Both `W` and `H` must be powers of two.
    I : ndarray
        Data image that the prototype will try to deform into. Array of shape ``(W, H)``. 
    last_level : int, optional
        Coefficient depth limit. If `None`, then naturally bounded by image size. 
        A higher level will allow for finer deformations, but incur a computational overhead.
    penalty : float, optional
        Determines the weight of the prior as opposed to the likelihood. (arbitrarily proportional to the ratio of the inverse variance of the gaussian deformations of the prior and the likelihood). Reduce this value if you want more deformations.
    rho : float, optional
        Determines the penalty of more granular coefficients. Increase to smoothen. Must be strictly positive.
    wavelet : str
        Wavelet type. See :class:`DisplacementFieldWavelet` for more information.
    maxiter : int, optional
        Maximum number of iterations per level.
    tol : float, optional
        Cost change must be less than `tol` before succesful termination at each coarse-to-fine level.
    first_level : int, optional
        First coarse-to-fine coefficient level.
    means : ndarray or None, optional
        Manually specify the means of the prior coefficients. If this and `variances` are set, then `penalty` and `rho` are unused. Must be of size ``(2, C)``, where `C` is the number of coefficients for the wavelet.
    variances : ndarray or None, optional
        Analagous to `means`, for specifying variances of the prior coefficients. Size should be the same as for `means`.
    llh_variances : ndarray or scalar, optional
        Specify log-likelihood variances per-pixel as a ``(W, H)`` array, or overall as a scalar. Default is 1.0, which can be seen as the variance being absorbed by the variance of the prior.
    debug_plot : bool, optional
        Output deformation progress live using :class:`PlottingWindow`. 
    
    Returns
    -------
    imdef : DisplacementFieldWavelet
        The deformation in the form of a :class:`DisplacementField`. 
    info : dict
        Dictionary with info:
         * `cost`: The final cost value.

    Examples
    --------
    Deform an image into a prototype image:

    >>> import amitgroup as ag
    >>> import numpy as np
    >>> import matplotlib.pylab as plt

    Load two example faces and perform the deformation:

    >>> F, I = ag.io.load_example('two-faces')
    >>> imdef, info = ag.stats.image_deformation(F, I, penalty=0.1)
    >>> Fdef = imdef.deform(F)

    Output the results:

    >>> ag.plot.deformation(F, I, imdef)
    >>> plt.show()

    """
    assert rho > 0, "Parameter rho must be strictly positive"
    assert len(F.shape) == 2 and len(I.shape) == 2, "Images must be 2D ndarrays"
    assert _powerof2(F.shape[0]) and _powerof2(I.shape[1]), "Image sides must be powers of 2"
    assert F.shape == I.shape, "Images must have the same shape"

    #from scipy.optimize import fmin_bfgs
    from amitgroup.stats.fmin_bfgs_tol import fmin_bfgs_tol as fmin_bfgs

    level_capacity = last_level
    delF = np.asarray(np.gradient(F, 1/F.shape[0], 1/F.shape[1]))
    dx = 1/np.prod(F.shape)

    settings = dict(
        penalty=penalty, 
        rho=rho, 
        wavelet=wavelet, 
        level_capacity=level_capacity, 
        means=means,
        variances=variances,
    ) 
    
    imdef = ag.util.DisplacementFieldWavelet(F.shape, **settings)
    x, y = imdef.meshgrid()

    if debug_plot:
        plw = ag.plot.PlottingWindow(figsize=(8, 4), subplots=(1,2))
        def cb(uk):
            if not plw.tick(1):
                raise ag.AbortException() 
            plw.imshow(imdef.deform(F), subplot=0)
            plw.imshow(I, subplot=1)
    else:
        cb = None 


    min_cost = np.inf
    for level in range(start_level, last_level+1):
        ag.info("Running coarse-to-fine level", level)
        u = imdef.abridged_u(level)
        args = (imdef, F, I, delF, x, y, level, llh_variances)
        try:
            new_u, cost, min_deriv, Bopt, func_calls, grad_calls, warnflag = \
                fmin_bfgs(_cost, u, _cost_deriv, args=args, callback=cb, tol=tol, maxiter=maxiter, full_output=True, disp=False)
        except ag.AbortException:
            return None, {}
        
        if cost < min_cost:
            # If the algorithm makes mistakes and returns a really high cost, don't use it.
            min_cost = cost
            imdef.u[:,:u.shape[1],:u.shape[2]] = new_u.reshape(u.shape)

    #if debug_plot:
    #    plw.mainloop()

    return imdef, {'cost': min_cost}

