from __future__ import division, print_function, absolute_import
import numpy as np

class DisplacementField(object):
    """
    Displacement field. Abstract base class for representing a deformation of a 2D mesh grid. 
    """
    def __init__(self, shape):
        self.shape = shape
        self.prepare_shape()

    def prepare_shape(self, shape):
        """
        Prepare shape. Subclasses can for instance prepare the 
        maximum number of coefficients appropriate for the shape.
        """
        pass

    def deform_map(self, x, y):
        """
        Creates a deformation array according the image deformation. 

        Parameters
        ----------
        x, y : ndarray
            Arrays of `x` and `y` values. Generate these by ``numpy.mgrid``. Array of shape ``(L, L)``.

        Returns
        -------
        Ux : ndarray
            Deformation along the `x` axis. Array of shape ``(L, L)``. 
        Uy : ndarray
            Same as above, along `y` axis. 
        """
        raise NotImplemented("Can't use DisplacementField directly")
        
    def deform(self, F):
        """
        Deforms the image F according to this displacement field.

        Parameters
        ----------
        F : ndarray
            2D array of data.
        
        Returns
        -------
        Fdef : ndarray
            2D array of the same size as `F`, representing a deformed version of `F`. 
        """
        raise NotImplemented("Can't use DisplacementField directly")


    def meshgrid(self):
        """
        Returns a mesh of `x` and `y` values appropriate to use with this displacement field.
    
        Returns
        -------
        x, y : ndarray, ndarray
            A tuple of two arrays, each with the same shape as the canonical shape of this displacement field.

        See also
        --------
        :func:`meshgrid_for_shape`
        """
        return self.meshgrid_for_shape(self.shape)

    @classmethod
    def meshgrid_for_shape(cls, shape):
        """
        Returns a mesh of `x` and `y` values appropriate to use with a displacement field of given shape. 

        Parameters
        ----------
        shape : tuple
            Tuple of length 2 specifying the size of the mesh.
        
        Returns
        -------
        x, y : ndarray
            A tuple of two arrays, each with the shape of the `shape` parameter.

        Examples
        --------
        >>> import amitgroup as ag

        Generate a mesh grid of appropriate size:

        >>> x, y = ag.util.DisplacementField.meshgrid_for_shape((4, 4))
        >>> x
        array([[ 0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.25,  0.25,  0.25,  0.25],
               [ 0.5 ,  0.5 ,  0.5 ,  0.5 ],
               [ 0.75,  0.75,  0.75,  0.75]])

        """
        dx = 1 / shape[0]
        dy = 1 / shape[1]
        return np.mgrid[0:1.0-dx:shape[0]*1j, 0:1.0-dy:shape[1]*1j]
 
    def __repr__(self):
        return "{0}(shape={1})".format(self.__class__.__name__, self.shape)
