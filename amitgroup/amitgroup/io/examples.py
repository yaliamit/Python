
import amitgroup as ag
import amitgroup.io
import numpy as np
import os

def datapath(name): 
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', name) 

def load_example(name):
    """
    Loads example data.

    The `amitgroup` package comes with some subsets of data sets, to use in examples and for quick testing.
    
    Parameters
    ----------
    name : str
        * `"faces"`: loads ``N`` faces into an array of shape ``(N, rows, cols)``.
        * `"two-faces"`: loads 2 faces into an array of shape ``(2, rows, cols)``.
        * `"oldfaithful"`: loads the classic Old Faithful eruption data as an array of shape ``(N, 2)``.
        * `"mnist"`: loads 10 MNIST nines into an array of shape ``(10, rows, cols)``.

    Examples
    --------
    >>> import amitgroup as ag
    >>> import matplotlib.pylab as plt

    Load faces:

    >>> faces = ag.io.load_example("faces")
    >>> ag.plot.images(faces[:4])
    >>> plt.show()

    Load MNIST:

    >>> digits = ag.io.load_example("mnist")
    >>> ag.plot.images(digits[:4])
    >>> plt.show()

    """
    if name == 'faces':
        return [x.img for x in ag.io.load_all_images(datapath('Images_0'))]
    if name == 'two-faces':
        data = np.load(datapath('twoface.npz'))
        return np.array([data['im1'], data['im2']])
    if name == 'oldfaithful':
        return np.genfromtxt(datapath('oldfaithful'))
    if name == 'mnist':
        return np.load(datapath('nines.npz'))['images']
    else:
        raise ValueError("Example data does not exist")
