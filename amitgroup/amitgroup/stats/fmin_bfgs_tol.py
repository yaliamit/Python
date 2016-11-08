from __future__ import division, print_function, absolute_import
# This is an almost exact copy of the fmin_bfgs in scipy.optimize, except that the stopping
# condition has been changed from checking the gtol norm, to the difference in cost.
# For use when fprime might not converge to a 0 norm at convergence.

from scipy.optimize.linesearch import line_search_wolfe1, line_search_wolfe2
import numpy
from numpy import Inf, sqrt, asarray, isinf, fabs

__all__ = ['fmin_bfgs_tol']

_epsilon = sqrt(numpy.finfo(float).eps)

def wrap_function(function, args):
    ncalls = [0]
    def function_wrapper(x):
        ncalls[0] += 1
        return function(x, *args)
    return ncalls, function_wrapper

def vecnorm(x, ord=2):
    if ord == Inf:
        return numpy.amax(abs(x))
    elif ord == -Inf:
        return numpy.amin(abs(x))
    else:
        return numpy.sum(abs(x)**ord, axis=0)**(1.0 / ord)

def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in Scipy. Level 4 is the first
        # level in user code.
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)

_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev' : 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.'}


class Result(dict):
    """ Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess : ndarray
        Values of objective function, Jacobian and Hessian (if available).
    nfev, njev, nhev: int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit: int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, self.keys())) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.iteritems()])
        else:
            return self.__class__.__name__ + "()"

def fmin_bfgs_tol(f, x0, fprime=None, args=(), tol=1e-5, norm=Inf,
              epsilon=_epsilon, maxiter=None, full_output=0, disp=1,
              retall=0, callback=None):
    """
    Minimize a function using the BFGS algorithm.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    fprime : callable f'(x,*args), optional
        Gradient of f.
    args : tuple, optional
        Extra arguments passed to f and fprime.
    tol : float, optional
        Cost change must be less than `tol` before succesful termination.
    norm : float, optional
        Order of norm (Inf is max, -Inf is min)
    epsilon : int or ndarray, optional
        If fprime is approximated, use this value for the step size.
    callback : callable, optional
        An optional user-supplied function to call after each
        iteration.  Called as callback(xk), where xk is the
        current parameter vector.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e. f(xopt) == fopt.
    fopt : float
        Minimum value.
    gopt : ndarray
        Value of gradient at minimum, f'(xopt), which should be near 0.
    Bopt : ndarray
        Value of 1/f''(xopt), i.e. the inverse hessian matrix.
    func_calls : int
        Number of function_calls made.
    grad_calls : int
        Number of gradient calls made.
    warnflag : integer
        1 : Maximum number of iterations exceeded.
        2 : Gradient and/or function calls not changing.
    allvecs  :  list
        Results at each iteration.  Only returned if retall is True.

    Other Parameters
    ----------------
    maxiter : int
        Maximum number of iterations to perform.
    full_output : bool
        If True,return fopt, func_calls, grad_calls, and warnflag
        in addition to xopt.
    disp : bool
        Print convergence message if True.
    retall : bool
        Return a list of results at each iteration if True.

    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
        functions. See the 'BFGS' `method` in particular.

    Notes
    -----
    Optimize the function, f, whose gradient is given by fprime
    using the quasi-Newton method of Broyden, Fletcher, Goldfarb,
    and Shanno (BFGS)

    References
    ----------
    Wright, and Nocedal 'Numerical Optimization', 1999, pg. 198.

    """
    opts = {'tol': tol,
            'norm': norm,
            'eps': epsilon,
            'disp': disp,
            'maxiter': maxiter,
            'return_all': retall}

    res = _minimize_bfgs(f, x0, args, fprime, callback=callback, **opts)

    if full_output:
        retlist = res['x'], res['fun'], res['jac'], res['hess'], \
                res['nfev'], res['njev'], res['status']
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']

def _minimize_bfgs(fun, x0, args=(), jac=None, callback=None,
                   tol=1e-5, norm=Inf, eps=_epsilon, maxiter=None,
                   disp=False, return_all=False,
                   **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options for the BFGS algorithm are:
        disp : bool
            Set to True to print convergence messages.
        maxiter : int
            Maximum number of iterations to perform.
        tol : float
            Cost change must be less than `tol` before succesful termination.
        norm : float
            Order of norm (Inf is max, -Inf is min).
        eps : float or ndarray
            If `jac` is approximated, use this value for the step size.

    This function is called by the `minimize` function with `method=BFGS`.
    It is not supposed to be called directly.
    """
    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0)*200
    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)
    gfk = myfprime(x0)
    k = 0
    N = len(x0)
    I = numpy.eye(N, dtype=int)
    Hk = I
    old_fval = f(x0)
    old_old_fval = old_fval + 5000
    xk = x0
    if retall:
        allvecs = [x0]
    sk = [2*0.1]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    while (fabs(old_fval - old_old_fval) > tol) and (k < maxiter):
        pk = -numpy.dot(Hk, gfk)
        alpha_k, fc, gc, old_fval2, old_old_fval2, gfkp1 = \
           line_search_wolfe1(f, myfprime, xk, pk, gfk,
                              old_fval, old_old_fval)
        if alpha_k is not None:
            old_fval = old_fval2
            old_old_fval = old_old_fval2
        else:
            # line search failed: try different one.
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     line_search_wolfe2(f, myfprime, xk, pk, gfk,
                                        old_fval, old_old_fval)
            if alpha_k is None:
                # This line search also failed to find a better solution.
                warnflag = 2
                break
        xkp1 = xk + alpha_k * pk
        if retall:
            allvecs.append(xkp1)
        sk = xkp1 - xk
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1
        if callback is not None:
            callback(xk)
        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        #if (gnorm <= gtol):
        #    break

        if not numpy.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        try:  #this was handled in numeric, let it remaines for more safety
            rhok = 1.0 / (numpy.dot(yk, sk))
        except ZeroDivisionError:
            rhok = 1000.0
            print("Divide-by-zero encountered: rhok assumed large")
        if isinf(rhok):  #this is patch for numpy
            rhok = 1000.0
            print("Divide-by-zero encountered: rhok assumed large")
        A1 = I - sk[:, numpy.newaxis] * yk[numpy.newaxis, :] * rhok
        A2 = I - yk[:, numpy.newaxis] * sk[numpy.newaxis, :] * rhok
        Hk = numpy.dot(A1, numpy.dot(Hk, A2)) + rhok * sk[:, numpy.newaxis] \
                * sk[numpy.newaxis, :]

    fval = old_fval
    if warnflag == 2:
        msg = _status_message['pr_loss']
        if disp:
            print("Warning:", msg)
            print("         Current function value:",fval)
            print("         Iterations:", k)
            print("         Function evaluations:", func_calls[0])
            print("         Gradient evaluations:", grad_calls[0])

    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
        if disp:
            print("Warning:", msg)
            print("         Current function value:", fval)
            print("         Iterations:", k)
            print("         Function evaluations:", func_calls[0])
            print("         Gradient evaluations:", grad_calls[0])
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value:", fval)
            print("         Iterations:", k)
            print("         Function evaluations:", func_calls[0])
            print("         Gradient evaluations:", grad_calls[0])

    result = Result(fun=fval, jac=gfk, hess=Hk, nfev=func_calls[0],
                    njev=grad_calls[0], status=warnflag,
                    success=(warnflag == 0), message=msg, x=xk)
    if retall:
        result['allvecs'] = allvecs
    return result
