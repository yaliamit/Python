import numpy as np
import scipy.signal

def convolve2d(signal, kernel, mode='full'):
    """
    Version of ``scipy.signal.convolve2d`` that allows `signal` to be of a dimension greater than 2, processing each 2D slice separately.

    Parameters
    ----------
    signal : ndarray
        Input array to be convolved with dimension 2 or greater. If dimension is greater than 2, each 2D slice will be convolved separately.
    kernel : ndarray
        2D array of kernel.
    mode: str, optional
        A string indicating the size of the output:

        ``valid`` : the output consists only of those elements that do not
           rely on the zero-padding.

        ``same`` : the output is the same size as ``in1`` centered
           with respect to the 'full' output.

        ``full`` : the output is the full discrete linear cross-correlation
           of the inputs. (Default)

    Returns
    -------
    out : ndarray
        An array of the same dimensions as `signal`, containing a subset of the discrete linear
        convolution of all 2D slices in `signal` with `kernel`.
    """
    import itertools
    ret = np.empty_like(signal) 
    # This is confirmed much faster than arrange a kernel with padded 1, and using
    # scipy.signal.convolve.
    for indices in itertools.product(*[range(i) for i in signal.shape[:-2]]):
        ret[indices] = scipy.signal.convolve2d(signal[indices], kernel, mode) 
    return ret 


def inflate2d(signal, kernel):
    """
    Let binary features spread to neighbors according to a kernel.
    
    Pararmeters:
    signal : ndarray
        Input array of any dimension 2 or greater. If dimension is greater than 2, each 2D slice will be convolved separately.
    kernel : ndarray
        2D array of kernel.
    """  
    return np.clip(convolve2d(signal, kernel, mode='same'), 0, 1)

# New name
dilate2d = inflate2d
