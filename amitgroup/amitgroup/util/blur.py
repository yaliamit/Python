
import scipy.signal
import numpy as np
import amitgroup as ag
import math

def _gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions. """
    isize = int(math.ceil(size))
    if not sizey:
        sizey = size
    isizey = int(math.ceil(sizey))
    x, y = np.mgrid[-isize:isize+1, -isizey:isizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def _blur_and_shrink(im, n, ny=None):
    g = _gauss_kern(n, sizey=ny)
    im = scipy.signal.convolve2d(im, g, mode='valid')
    return im

def blur_image(im, n, ny=None, maintain_size=True):
    """ 
    Blurs the image by convolving with a gaussian kernel of typical
    size `n`. The optional keyword argument `ny` allows for a different
    size in the `y` direction.
    
    You can also use scipy.ndimage.filters.gaussian_filter.

    Parameters
    ----------
    im : ndarray
        2D or 3D array with an image. If the array has 3 dimensions, then the last channel is assumed to be a color channel, and the blurring is done separately for each channel.
    n : int
        Kernel size.
    ny : int
        Kernel size in y, if specified. 
    maintain_size : bool
        If True, the size of the image will be maintained. This is done by first padding the image with the edge values, before convolving.

    Examples
    --------
    >>> import amitgroup as ag
    >>> import matplotlib.pylab as plt

    Blur an image of a face:

    >>> face = ag.io.load_example('faces')[0]
    >>> face2 = ag.util.blur_image(face, 5)
    >>> ag.plot.images([face, face2]) 
    >>> plt.show()
    """

    if im.ndim == 3:
        assert maintain_size, "Not implemented yet"
        for ch in range(im.shape[2]):
            im[...,ch] = blur_image(im[...,ch], n, ny)
        return im

    else:
        if maintain_size:
            if ny is None:
                ny = n 
            i_n = int(math.ceil(n))
            i_ny = int(math.ceil(ny))
            x, y = np.mgrid[-i_n:im.shape[0]+i_n, -i_ny:im.shape[1]+i_ny].astype(float)
            bigger = ag.util.interp2d(x, y, im.astype(float), startx=(0, 0), dx=(1, 1))
            return _blur_and_shrink(bigger, n, ny) 
        else:
            return _blur_and_shrink(im, n, ny)
