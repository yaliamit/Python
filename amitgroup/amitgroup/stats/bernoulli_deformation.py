from __future__ import division, print_function, absolute_import
import numpy as np
import amitgroup as ag
import amitgroup.util
import amitgroup.features
import math
import sys

def _cost(u, imdef, F, X, neg_X, delFjs, x, y, level, all_js):
    """Calculate the cost."""
    imdef.set_flat_u(u, level)

    # Calculate deformed xs
    z0, z1 = imdef.deform_x(x, y, level)

    # Interpolate F at zs
    Fjzs = ag.util.interp2d(z0, z1, F)

    # 3. Cost

    # log-prior
    logprior = imdef.logprior()

    # log-likelihood
    loglikelihood = (X * np.log(Fjzs) + neg_X * np.log(1-Fjzs)).sum()

    # cost
    return -logprior - loglikelihood

def _cost_deriv(u, imdef, F, X, neg_X, delFjs, x, y, level, all_js):
    """Calculate the derivative of the cost."""
    imdef.set_flat_u(u, level)

    # Calculate deformed xs
    z0, z1 = imdef.deform_x(x, y, level)

    # Interpolate F at zs
    Fjzs = ag.util.interp2d(z0, z1, F)
    neg_Fjzs = 1 - Fjzs

    # Interpolate delF at zs 
    delFjzs = ag.util.interp2d(z0, z1, delFjs, fill_value=0.0)

    s = -(X/Fjzs - neg_X/neg_Fjzs)
    W = np.empty((2,) + x.shape) # Change to empty
    for q in range(2):
        grad = delFjzs[q]
        W[q] = (s * grad).sum(axis=0) 

    vqks = imdef.transform(W, level)
    N = 2**level
    return (-imdef.logprior_derivative() + vqks)[:,:N,:N].flatten()

if 0:
    def _cost_num_deriv(u, imdef, F, X, neg_X, delFjs, x, y, level, all_js):
        """Numerical derivative of the cost. Can be used for comparison."""
        imdef.set_flat_u(u)

        orig_u = np.copy(imdef.u)
        
        deriv = np.zeros(orig_u.shape)
        limit = imdef.flat_limit(level) 
        dt = 0.00001
        for q in range(2):
            for i in range(limit):
                u = np.copy(orig_u)
                u[q,i] -= dt
                cost0 = _cost(u, imdef, F, X, delFjs, x, y, level, all_js)
                u = np.copy(orig_u)
                u[q,i] += dt
                cost1 = _cost(u, imdef, F, X, delFjs, x, y, level, all_js)
                deriv[q,i] = (cost1-cost0)/(2*dt)

        # Compare
        deriv = deriv[:,:2**(level),2**(level)].flatten()
        return deriv

def bernoulli_deformation(F, I, last_level=None, penalty=1.0, tol=0.001, rho=2.0, wavelet='db2', maxiter=50, start_level=1, means=None, variances=None, debug_plot=False):
    assert F.ndim == 3, "F must have 3 axes"
    assert F.shape == I.shape, "F and I shape mismatch {0} {1}".format(F.shape, I.shape)
    assert F.shape[0] == 8, "Currently only supports 8 features (not {0})".format(F.shape[0])

    #from scipy.optimize import fmin_bfgs
    from amitgroup.stats.fmin_bfgs_tol import fmin_bfgs_tol as fmin_bfgs

    # This, or an assert
    X = I.astype(float)
    all_js = range(8)

    # We don't need more capacity for coefficients than the last_level
    level_capacity = last_level

    delFjs = []
    for j in all_js:
        delF = np.gradient(F[j], 1/F[j].shape[0], 1/F[j].shape[1])
        # Normalize since the image covers the square around [0, 1].
        delFjs.append(delF)
    delFjs = np.rollaxis(np.asarray(delFjs), 1)

    settings = dict(
        penalty=penalty, 
        wavelet=wavelet, 
        rho=rho, 
        level_capacity=level_capacity, 
        means=means, 
        variances=variances,
    )
    
    imdef = ag.util.DisplacementFieldWavelet(F.shape[1:], **settings)

    x, y = imdef.meshgrid()

    if debug_plot:
        plw = ag.plot.PlottingWindow(figsize=(8, 8), subplots=(4,4))
        def cb(uk):
            if not plw.tick(1):
                raise ag.AbortException() 
            for j in range(8):
                plw.imshow(imdef.deform(F[j]), subplot=j*2)
                plw.imshow(I[j], subplot=j*2+1)
    else:
        cb = None 

    min_cost = np.inf
    for level in range(start_level, last_level+1): 
        ag.info("Running coarse-to-fine level", level)
        
        imdef.reset(level)
        
        u = imdef.u
        args = (imdef, F, X, 1-X, delFjs, x, y, level, all_js)

        try:
            new_u, cost, min_deriv, Bopt, func_calls, grad_calls, warnflag = \
                fmin_bfgs(_cost, u, _cost_deriv, args=args, callback=cb, tol=tol, maxiter=maxiter, full_output=True, disp=False)
        except ag.AbortException:
            return None, {}
    
        if cost < min_cost:
            # If the algorithm makes mistakes and returns a really high cost, don't use it.
            min_cost = cost
            #imdef.u[:,:u.shape[1],:u.shape[2]] = new_u.reshape(u.shape)
            imdef.u = new_u.reshape(imdef.u.shape)

    #if debug_plot:
    #    plw.mainloop()

    return imdef, {'cost': min_cost}
    
