
from __future__ import absolute_import
import numpy as np
import scipy.signal
import amitgroup as ag
import matplotlib.colors as col
from amitgroup.features.features import (array_bedges,
                                         array_byte_bedges,
                                         array_bedges2,
                                         array_bspread,
                                         array_intensities,
                                         change_saturation_c
                                         )


# def change_saturation(xx,fac):
#
#     xx=np.transpose(xx,(1,2,0))
#
#     yy=change_saturation_c(np.floatX(xx),np.floatX(fac))
#     yy=np.transpose(yy,(2,0,1))
#
#     return(yy)

def change_saturation(xx,fac):

    xx=np.transpose(xx,(0,2,3,1))
    yy=np.zeros(np.shape(xx))
    for i,x in enumerate(xx):
        y=col.rgb_to_hsv(x)
        y[:,1,:]=np.minimum(y[:,1,:]*fac[i],1)
        yy[i]=col.hsv_to_rgb(y)

    #yy=change_saturation_c(np.floatX(xx),np.floatX(fac))
    yy=np.transpose(yy,(0,3,1,2))

    return(yy)

# Builds a kernel along the edge direction
def _along_kernel(direction, radius):
    d = direction % 4
    kern = None
    if d == 2:  # S/N
        kern = np.zeros((radius * 2 + 1,) * 2, dtype=np.uint8)
        kern[radius, :] = 1
    elif d == 0:  # E/W
        kern = np.zeros((radius * 2 + 1,) * 2, dtype=np.uint8)
        kern[:, radius] = 1
    elif d == 3:  # SE/NW
        kern = np.eye(radius * 2 + 1, dtype=np.uint8)[::-1]
    elif d == 1:  # NE/SW
        kern = np.eye(radius * 2 + 1, dtype=np.uint8)

    return kern


def bspread(X, spread='box', radius=1, first_axis=False):
    """
    Spread binary edges.

    Parameters
    ----------
    X : ndarray  (3D or 4D)
        Binary edges to spread. Shape should be ``(rows, cols, A)`` or ``(N,
        rows, cols, A)``, where `A` is the number of edge features.
    first_axis: bool
         If True, the images will be assumed to be ``(A, rows, cols)`` or ``(N,
         A, rows, cols)``.
    spread : 'box', 'orthogonal', None
        If set to `'box'` and `radius` is set to 1, then an edge will appear if
        any of the 8 neighboring pixels detected an edge. This is equivalent to
        inflating the edges area with 1 pixel. The size of the box is dictated
        by `radius`.
        If `'orthogonal'`, then the features will be extended by `radius`
        perpendicular to the direction of the edge feature (i.e. along the
        gradient).
    radius : int
        Controls the extent of the inflation, see above.
    """
    single = X.ndim == 3
    if single:
        X = X.reshape((1,) + X.shape)
    if not first_axis:
        X = np.rollaxis(X, 3, start=1)

    Xnew = array_bspread(X, spread, radius)

    if not first_axis:
        Xnew = np.rollaxis(Xnew, 1, start=4)

    if single:
        Xnew = Xnew.reshape(Xnew.shape[1:])

    return Xnew


def bedges(images, k=6, spread='box', radius=1, minimum_contrast=0.0, color_gradient_thresh=0,
        contrast_insensitive=False, first_axis=False, max_edges=None,
        preserve_size=True, pre_blurring=None):
    """
    Extracts binary edge features for each pixel according to [AmitBook]_.

    The function returns 8 different binary features, representing directed
    edges. Let us define a south-going edge as when it starts at high intensity
    and drops when going south (this would make south edges the lower edge of
    an object, if background is low intensity and the object is high
    intensity). By this defintion, the order of the returned edges is S, SE, E,
    NE, N, NW, W, SW.

    Parameters
    ----------
    images : ndarray
        Input an image of shape ``(rows, cols)`` or a list of images as an
        array of shape ``(N, rows, cols)``, where ``N`` is the number of
        images, and ``rows`` and ``cols`` the size of each image.
    k : int
        There are 6 contrast differences that are checked. The value `k`
        specifies how many of them must be fulfilled for an edge to be present.
        The default is all of them (`k` = 6) and gives more conservative edges.
    spread : 'box', 'orthogonal', None
        If set to `'box'` and `radius` is set to 1, then an edge will appear if
        any of the 8 neighboring pixels detected an edge. This is equivalent to
        inflating the edges area with 1 pixel. The size of the box is dictated
        by `radius`.
        If `'orthogonal'`, then the features will be extended by `radius`
        perpendicular to the direction of the edge feature (i.e. along the
        gradient).
    radius : int
        Controls the extent of the inflation, see above.
    minimum_contrast : double
        Requires the gradient to have an absolute value greater than this, for
        an edge to be detected. Set to a non-zero value to reduce edges firing
        in low contrast areas.
    contrast_insensitive : bool
        If this is set to True, then the direction of the gradient does not
        matter and only 4 edge features will be returned.
    first_axis: bool
         If True, the images will be returned with the features on the first
         axis as ``(A, rows, cols)`` instead of ``(rows, cols, A)``, where `A`
         is either 4 or 8. If mutliple input entries, then the output will be
         ``(N, A, rows, cols)``.
    max_edges : int or None
        Maximum number of edges that can assigned at a single pixel. The ones
        assigned will be the ones with the higest contrast.
    preserve_size : bool
        If True, the returned feature vector has the same size as the input
        vector, but it will have an empty border of size 2 around it.

    Returns
    -------
    edges : ndarray
        An array of shape ``(rows, cols, A)`` if entered as a single image, or
        ``(N, rows, cols, A)`` of multiple. Each pixel in the original image
        becomes a binary vector of size 8, one bit for each cardinal and
        diagonal direction. Note that if `first_axis` is True, this shape will
        change.
    """
    single = images.ndim == 2
    if single:
        images = images[np.newaxis]

    if images.ndim == 4:
        assert images.shape[3] == 1, "bedges does not handle multiple color channels"
        images = images[...,0]

    # TODO: Temporary stuff
    if pre_blurring is not None and pre_blurring != 0.0:
        images = images.copy()
        for i in range(images.shape[0]):
            images[i] = ag.util.blur_image(images[i], pre_blurring)

    if images.dtype == np.float64:
        if max_edges is not None:
            features = array_bedges2(images, k, minimum_contrast,
                                     contrast_insensitive, max_edges)
        else:
            features = array_bedges(images, k, minimum_contrast,
                                    contrast_insensitive)
    elif images.dtype == np.uint8:
        features = array_byte_bedges(images, k, int(minimum_contrast * 255),
                                     contrast_insensitive).view(np.uint8)

    else:
        raise ValueError("Input image must be float64 or uint8")

    # Spread the feature
    features = bspread(features, radius=radius, spread=spread, first_axis=True)

    # Skip the 2-pixel border that is not valid
    if not preserve_size:
        features = features[:,:,2:-2,2:-2]

    if not first_axis:
        features = np.rollaxis(features, axis=1, start=features.ndim)

    if single:
        features = features[0]

    return features


def bedges_from_image(im, k=6, spread='box', radius=1, minimum_contrast=0.0,
        contrast_insensitive=False, first_axis=False, return_original=False):
    """
    This wrapper for :func:`bedges`, will take an image file, load it and
    compute binary edges for each color channel separately, and then finally OR
    the result.

    Parameters
    ----------
    im : str / ndarray
        This can be either a string with a filename to an image file, or an
        ndarray of three dimensions, where the third dimension is the color
        channels.

    Returns
    -------
    edges : ndarray
        An array of shape ``(8, rows, cols)`` if entered as a single image, or
        ``(N, 8, rows, cols)`` of multiple. Each pixel in the original image
        becomes a binary vector of size 8, one bit for each cardinal and
        diagonal direction.
    return_original : bool
        If True, then the original image is returned as well as the edges.
    image : ndarray
        An array of shape ``(rows, cols, D)``, where `D` is the number of color
        channels, probably 3 or 4. This is only returned if `return_original`
        is set to True.

    The rest of the argument are the same as :func:`bedges`.
    """
    if isinstance(im, str) or isinstance(im, file):
        from PIL import Image
        im = np.array(Image.open(im))
        # TODO: This needs more work. We should probably make bedges work with
        # any type and then just leave it at that.
        if im.dtype == np.uint8:
            im = im.astype(np.float64) / 255.0

    # Run bedges on each channel, and then OR it.
    dimensions = im.shape[-1]

    # This will use all color channels, including alpha, if there is one
    edges = [bedges(im[...,i],
                    k=k,
                    spread=spread,
                    radius=radius,
                    minimum_contrast=minimum_contrast,
                    contrast_insensitive=contrast_insensitive,
                    first_axis=first_axis)
                for i in range(dimensions)]

    final = reduce(np.bitwise_or, edges)

    if return_original:
        return final, im
    else:
        return final
