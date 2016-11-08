#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import os.path

from Cython.Distutils import build_ext

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
"""


def cython_extension(modpath, mp=False):
    extra_compile_args = ["-O3"]
    extra_link_args = []
    if mp:
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-fopenmp')
    filepath = os.path.join(*modpath.split('.')) + ".pyx"
    return Extension(modpath, [filepath],
                     extra_compile_args=extra_compile_args,
                     extra_link_args=extra_link_args)

setup(
    name='amitgroup',
    cmdclass={'build_ext': build_ext},
    version='0.9.1',
    url="https://github.com/amitgroup/amitgroup",
    description="Code for Yali Amit's Research Group.",
    maintainer='Gustav Larsson',
    maintainer_email='gustav.m.larsson@gmail.com',
    packages=[
        'amitgroup',
        'amitgroup.features',
        'amitgroup.io',
        'amitgroup.stats',
        'amitgroup.util',
        'amitgroup.util.wavelet',
        'amitgroup.plot',
    ],
    ext_modules=[
        cython_extension("amitgroup.features.features"),
        cython_extension("amitgroup.features.spread_patches"),
        cython_extension("amitgroup.features.code_parts"),
        cython_extension("amitgroup.util.interp2d"),
        cython_extension("amitgroup.util.nn2d"),
        cython_extension("amitgroup.plot.resample"),
    ],
    include_dirs=[np.get_include()],
    license='BSD',
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
)
