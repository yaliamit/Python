

@BinaryDescriptor.register('hog')
from binary_descriptor import BinaryDescriptor
    def __init__(self, patch_size, num_parts, settings={}):
    
        self.settings = {}
        self.settings['cells_per_block'] = (3, 3)
        self.settings['pixels_per_cell'] = (9, 9)
        self.settings['orientations'] = 9 
        self.settings['normalise'] = True 
        
        self.settings['binarize_threshold'] = 0.05

    def extract_features(self, image, settings={}):
        from skimage import feature
        hog = feature.hog(image, 
                          orientations=self.settings['orientations'],
                          pixels_per_cell=self.settings['pixels_per_cell'],
                          cells_per_block=self.settings['cells_per_block'],
                          normalise=self.settings['normalise'])

        # Let's binarize the features somehow
        hog = (hog > self.settings['binarize_threshold']).astype(np.uint8)

    
        # This is just to unravel it 
        sx, sy = image.shape
        bx, by = self.settings['cells_per_block']
        cx, cy = self.settings['pixels_per_cell']
        n_cellsx = int(np.floor(sx // cx))  # number of cells in x
        n_cellsy = int(np.floor(sy // cy))  # number of cells in y
        n_blocksx = (n_cellsx - bx) + 1 
        n_blocksy = (n_cellsy - by) + 1
        shape = (n_blocksy, n_blocksx, by, bx, orientations)
    
        # We have to binarize the 
        # Now, let's reshape it to keep spatial locations, but flatten the rest
        
        return hog.reshape((n_blocksy, n_blocksx, -1))

    def save_to_dict(self):
        return self.settings

    @classmethod
    def load_from_dict(cls, d):
        return cls(d)
