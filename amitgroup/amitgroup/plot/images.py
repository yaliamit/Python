
import amitgroup as ag
import numpy as np
import math

def images(data, zero_to_one=True, show=True, subplots=None, caption=None):
    """
    Display images that range a grid. Especially designed for probability images, ranging from 0 to 1.

    Parameters
    ----------
    data : ndarray
        An array of images or a single image of shape. The values should range between 0 and 1 (at least, that is how they will be colorized).
    zero_to_one : bool
        If True, then 0.0 and below will be pitch black, and 1.0 and above will be chalk white.    
    show : bool
        Call `pylab.show()` inside the function.
    subplots : tuple or None
        Specify the shape of the subplots manually.
    caption : lambda(index, element)
        A function that takes the index and the element (``data[index]``) and should return a string.
    """
    import matplotlib.pylab as plt

    settings = {
        'interpolation': 'nearest',
        'cmap': plt.cm.gray,
    }

    if zero_to_one:
        settings['vmin'] = 0.0
        settings['vmax'] = 1.0

    if isinstance(data, np.ndarray) and data.ndim == 2:
        fig = plt.figure()
        plt.subplot(111).set_axis_off()
        plt.imshow(data, **settings)
    else:
        # TODO: Better find out pleasing aspect ratios
        N = len(data)
        if subplots is not None:
            sh = subplots
            assert len(data) <= np.prod(subplots) 
        elif N <= 3:
            sh = (1, len(data))
        elif N == 5 or N == 6:
            sh = (2, 3)
        elif N == 12:
            sh = (3, 4)
        else:
            perside = int(math.ceil(math.sqrt(len(data))))
            sh = (perside,)*2
        fig = plt.figure()
        for i, im in enumerate(data): 
            plt.subplot(sh[0], sh[1], 1+i).set_axis_off()
            plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            plt.imshow(im, **settings)
            if caption is not None:
                plt.title(caption(i, im))

    if show:
        plt.show()

def deformation(F, I, displacement_field, show_diff=False, show=True):
    """
    
    Plot how a :class:`DisplacementField` applies to a prototype image `F` to qualitatively compare it with `I`.

    Especially designed to plot the results of :func:`image_deformation`.

    Parameters
    ----------
    F : ndarray
        Prototype image.
    I : ndarray
        Data image. This is included for comparison.
    displacement_field : DisplacementField  
        A displacement field that will be applied to `F` and also plotted by itself.
    show_diff : bool
        If True, displays the difference map instead of the vector field. 
    show : bool
        Call `pylab.show()` inside the function.
    """
    import matplotlib.pylab as plt
    
    assert isinstance(displacement_field, ag.util.DisplacementField) 

    x, y = displacement_field.meshgrid()
    Ux, Uy = displacement_field.deform_map(x, y) 
    Fdef = displacement_field.deform(F)

    d = dict(interpolation='nearest', cmap=plt.cm.gray)
    plt.figure(figsize=(7,7))
    plt.subplot(221)
    plt.title("Prototype")
    plt.imshow(F, **d)
    plt.subplot(222)
    plt.title("Data image")
    plt.imshow(I, **d) 
    plt.subplot(223)
    plt.title("Deformed prototype")
    plt.imshow(Fdef, **d)
    plt.subplot(224)
    if show_diff:
        plt.title("Difference")
        plt.imshow(Fdef - I, interpolation='nearest')
        plt.colorbar()
    else:
        plt.title("Displacement field")
        plt.quiver(np.asarray(y), np.asarray(-x), np.asarray(Uy), np.asarray(-Ux), linewidth=2.0)
    if show:
        plt.show()
