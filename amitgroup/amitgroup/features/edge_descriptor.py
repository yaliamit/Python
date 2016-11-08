
import amitgroup as ag
import amitgroup.features
from .binary_descriptor import BinaryDescriptor

@BinaryDescriptor.register('edges')
class EdgeDescriptor(BinaryDescriptor):
    """
    Binary descriptor for simple oriented edges.

    The parameters are similar to :func:`amitgroup.features.bedges`.
    
    Parameters
    ----------
    polarity_sensitive : bool
        If True, the polarity of the edges will matter. Otherwise, edges of opposite direction will be collapsed into one feature.
    k : int
        See :func:`amitgroup.features.bedges`.
    radius : int
        Radius of edge spreading. See :func:`amitgroup.features.bedges`.
    min_contrast : float 
        See :func:`amitgroup.features.bedges`.
    """
    def __init__(self, polarity_sensitive=True, k=5, radius=1, min_contrast=0.1):
        self.settings = {}
        # Change this 
        self.settings['contrast_insensitive'] = not polarity_sensitive 
        self.settings['k'] = k 
        self.settings['radius'] = radius 
        self.settings['minimum_contrast'] = min_contrast 

    def extract_features(self, img):
        #return ag.features.bedges_from_image(img, **self.settings)
        return ag.features.bedges(img, **self.settings)

    def save_to_dict(self):
        return self.settings

    @classmethod
    def load_from_dict(cls, d):
        return cls(d)
