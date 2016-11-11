Documentation
=============

This document explains how to generate the documentations. If you are an average user, you do not need to do this and can just go to:

 * https://amitgroup.github.com/amitgroup/

Requirements
------------

 * [Sphinx](http://sphinx.pocoo.org) (0.2>=, 0.2pre is OK)
 * [Numpydoc](http://pypi.python.org/pypi/numpydoc)
 * And the requirements for amitgroup.

If you use Sphinx 1.1.3, then the Cython code will not be properly documented.

Building
--------

While in `amitgroup/`:
    
    python setup.py build_ext --inplace
    cd doc
    make html 

This will generate documentation in `build/html`.

