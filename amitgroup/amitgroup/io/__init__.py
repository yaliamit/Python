from __future__ import absolute_import

from .mnist import load_mnist
from .casia import load_casia
from .norb import load_small_norb
from .cifar import load_cifar_10, load_cifar_100
from .examples import load_example

__all__ = ['load_mnist',
           'load_small_norb',
           'load_casia',
           'load_example',
           'load_cifar_10',
           'load_cifar_100']

try:
    import tables
    _pytables_ok = True
    del tables
except ImportError:
    _pytables_ok = False

if _pytables_ok:
    from .hdf5io import load, save
else:
    def _f(*args, **kwargs):
        raise ImportError("You need PyTables for this function")
    load = save = _f
