.. _reference:

API Reference
=============

.. currentmodule:: amitgroup

.. _features:

Core package (:mod:`amitgroup`)
-------------------------------

.. autosummary::
   :toctree: generated/

   AbortException
   Timer
   apply_once
   bytesize
   humanize_bytesize
   span

Printing
~~~~~~~~

.. autosummary::
   :toctree: generated/

   info
   memsize
   set_verbose
   warning

Feature extraction (:mod:`amitgroup.features`)
----------------------------------------------
.. currentmodule:: amitgroup.features

Classes
~~~~~~~

.. autosummary:: 
   :toctree: generated/

   BinaryDescriptor
   EdgeDescriptor
   PartsDescriptor

Functions
~~~~~~~~~

.. autosummary:: 
   :toctree: generated/

   bedges
   spread_patches
   code_parts

.. _io:

Input/Output (:mod:`amitgroup.io`)
----------------------------------
.. currentmodule:: amitgroup.io

Main functions
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   load
   save

Load data sets
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   load_casia
   load_example
   load_mnist
   load_small_norb

Statistics (:mod:`amitgroup.stats`)
-----------------------------------
.. currentmodule:: amitgroup.stats

.. autosummary::
   :toctree: generated/

   BernoulliMixture
   bernoulli_deformation
   image_deformation

Utilities (:mod:`amitgroup.util`)
---------------------------------
.. currentmodule:: amitgroup.util

.. autosummary::
   :toctree: generated/

   DisplacementField
   DisplacementFieldWavelet
   Saveable
   blur_image
   interp2d
   pad
   pad_repeat_border
   pad_repeat_border_corner
   pad_to_size

Image handling (:mod:`amitgroup.image`)
---------------------------------------
.. currentmodule:: amitgroup.image

.. autosummary::
   :toctree: generated/

    asgray
    bounding_box
    bounding_box_as_binary_map
    crop
    crop_to_bounding_box
    extract_patches
    integrate
    load
    offset
    resize_by_factor
    save

Wavelets (:mod:`amitgroup.util.wavelet`)
----------------------------------------
.. currentmodule:: amitgroup.util.wavelet

Read more about wavelets in the chapter :ref:`wavelet`.

Main functions
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
  
   daubechies_factory
   wavedec
   waverec
   wavedec2
   waverec2

Helper functions
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
  
   contiguous_to_structured
   smart_deflatten
   smart_flatten
   structured_to_contiguous

Plotting (:mod:`amitgroup.plot`)
--------------------------------
.. currentmodule:: amitgroup.plot

The plotting module adds some convenience functions for using matplotlib_ and pygame_.

Classes
~~~~~~~

.. autosummary::
   :toctree: generated/

   ColorImageGrid
   ImageGrid
   PlottingWindow

Functions
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   deformation
   images

.. _matplotlib: http://matplotlib.sourceforge.net
.. _pygame: http://www.pygame.org/
