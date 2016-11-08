from __future__ import absolute_import

# Load the following modules by default
from .core import (set_verbose,
                   set_parallel,
                   get_parallel_pool,
                   apply_function_at_pool,
                   info,
                   warning,
                   AbortException,
                   bytesize,
                   humanize_bytesize,
                   memsize,
                   span,
                   apply_once,
                   apply_once_over_axes,
                   Timer)


# Lazy load these?
from amitgroup import io
from amitgroup import features
from amitgroup import stats
from amitgroup import util
from amitgroup import plot
from . import image

VERSION = (0, 9, 1)
ISRELEASE = False
__version__ = '{0}.{1}.{2}'.format(*VERSION)
if not ISRELEASE:
    __version__ += '.dev'

# Modules
__all__ = ['io', 'features', 'stats', 'util', 'plot', 'image']

# Core functions
__all__ += ['set_verbose',
            'info',
            'warning',
            'AbortException',
            'bytesize',
            'humanize_bytesize',
            'memsize',
            'span',
            'apply_once_over_axes',
            'Timer']


def test(verbose=False):
    from amitgroup import tests
    import unittest
    unittest.main()
