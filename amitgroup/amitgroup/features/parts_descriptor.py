import random
import amitgroup as ag
import numpy as np
from .binary_descriptor import BinaryDescriptor

# TODO: Move this to util (now, it already has a load_image)
def load_image(path):
    from PIL import Image
    im = np.array(Image.open(path))
    return im.astype(np.float64)/255.0

@BinaryDescriptor.register('parts')
class PartsDescriptor(BinaryDescriptor):
    """
    Parts descriptor based on a mixture model of image patches.

    Edges are extracted using :func:`amitgroup.features.bedges`. Then, a mixture model of patches of 
    these edges is trained. The procedure is described in [Bernstein05]_.

    In the following, a `patch` refers to any random sample of an image, while a `part`
    refers to one of the trained mixture components.

    Parameters
    ----------
    patch_size : tuple
        Size of the patches in pixels, for instance ``(5, 5)``.
    num_parts : int
        Number of target mixture components. You could end up with fewer components, 
        if certain criteria are used.
    patch_frame : int
        Size of the frame that is excluded from the edge threshold.
    edges_threshold : int
        Number of minimum spread edges inside the patch frame, or the patch will be discarded.
    samples_per_image : int
        Numer of random samples per image when training.
    discard_threshold : 
        If this is `None`, no parts will be discarded.
        If this is set to a numeric value, then all parts will be standardized by their own
        background distribution, and this value will be used as a lower bound on that value,
        otherwise a part will be discarded. 
    min_probability : float
        Minimum probability of each Bernoulli probability. 
    bedges_settings : dict
        Dictionary of settings passed to :func:`amitgroup.features.bedges`.
    """
    def __init__(self, patch_size, num_parts, patch_frame=1, edges_threshold=10, samples_per_image=100, \
                 discard_threshold=None, min_probability=0.05, bedges_settings={}):
        self.patch_size = patch_size
        self.num_parts = num_parts 

        self.parts = None
        self.visparts = None

        self.settings = {}
        self.settings['patch_frame'] = patch_frame
        self.settings['threshold'] = edges_threshold 
        self.settings['threaded'] = False 
        self.settings['samples_per_image'] = samples_per_image 
        self.settings['min_probability'] = min_probability 

        # Or maybe just do defaults?
        # self.settings['bedges'] = {}
        self.settings['bedges'] = dict(k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=True)
        self.settings['bedges'].update(bedges_settings)

    def train_from_images(self, images):
        """
        Given a list of filenames, it will draw random patches from these images,
        and then train the parts model using those patches.

        Settings (such as patch size) are specified when constructing the class.

        Parameters
        ----------
        images : list or ndarray
            This can be:
                * A list of filenames.
                * A list of images (ndarray).
                * An ndarray of images of the same size. 
        """
        # TODO: Allow filenames to be a list of actual image objects too?
        raw_patches, raw_originals = self._random_patches_from_images(images)
        if len(raw_patches) == 0:
            raise Exception("No patches found, maybe your thresholds are too strict?")
        mixture = ag.stats.BernoulliMixture(self.num_parts, raw_patches, init_seed=0)
        # Also store these in "settings"
        mixture.run_EM(1e-8, min_probability=self.settings['min_probability'])
        ag.info("Done.")

        # Reject weak parts
        scores = np.empty(self.num_parts) 
        for i in range(self.num_parts):
            part = mixture.templates[i]
            sh = part.shape
            p = part.reshape((sh[0]*sh[1], sh[2]))
            
            #import ipdb; ipdb.set_trace()
            pec = p.mean(axis=0)
        
            N = np.sum(p * np.log(p/pec) + (1-p)*np.log((1-p)/(1-pec)))
            D = np.sqrt(np.sum(np.log(p/(1-p))**2 * p * (1-p)))

            scores[i] = N/D 

        # Only keep with a certain score
        visparts = mixture.remix(raw_originals)
        
        self.parts = mixture.templates[scores > 1]
        self.visparts = visparts[scores > 1]
        self.num_parts = self.parts.shape[0]
        
        # Update num_parts
        
        # Store the stuff in the instance
        #self.parts = mixture.templates
        #self.visparts = mixture.remix(raw_originals)

        self._preprocess_logs()

    def extract_features(self, image, settings={}):
        """
        Extracts and returns features as a binary array.
        
        Parameters
        ----------
        image : ndarray
            Image in the form of an numpy array. Both grayscale and colors is fine.
        settings : dict
            Additional settings that do not need retraining:
                * `"spread_radii"`: A tuple that specifies the radius of parts spreading in both axes.
        """
        if 1:
            edges = ag.features.bedges(image, **self._bedges_settings())
        else:
            # LEAVE-BEHIND: From multi-channel images
            edges = ag.features.bedges_from_image(image, **self._bedges_settings()) 
        return self._extract_parts(edges, settings=settings)

    def extract_features_many(self, images, settings={}):
        """
        Extract features from several images at once.
    
        This could be optimized by running this outer loop
        in Cython functions. The bedges function handles this, but
        not the code_parts.
        """
        ret = None
        for i, image in enumerate(images):
            parts = self.extract_features(image, settings=settings)
            if ret is None:
                ret = np.empty((len(images),) + parts.shape)
            ret[i] = parts
        return ret 

    #
    # Private functions
    #

    def _bedges_settings(self):
        return self.settings['bedges']

    def _random_patches_from_images(self, images):
        raw_patches = []
        raw_originals = [] 

        # TODO: Have an amitgroup / vision-research setting for "allow threading"
        if 0:
            if self.settings['threaded']:
                from multiprocessing import Pool
                p = Pool(8) # Should not be hardcoded
                mapfunc = p.map
            else:
                mapfunc = map

        ret = map(self._get_patches, images)

        for patches, originals in ret:
            raw_patches.extend(patches)
            raw_originals.extend(originals) 

        raw_patches = np.asarray(raw_patches)
        raw_originals = np.asarray(raw_originals)
        return raw_patches, raw_originals


    def _get_patches(self, image):
        samples_per_image = self.settings['samples_per_image']
        fr = self.settings['patch_frame']
        the_patches = []
        the_originals = []
        if isinstance(image, str):
            ag.info("Extracting patches from", filename)
            img = load_image(filename)
        else:
            ag.info("Extracting patches from image of size", image.shape)
            img = image
        #edges, img = ag.features.bedges_from_image(f, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=False, return_original=True, lastaxis=True)

        # Convert to grayscale if necessary
        if img.ndim == 3:
            img = img[...,:3].mean(axis=-1)

        # LEAVE-BEHIND
        if 1:
            edges = ag.features.bedges(img, **self.settings['bedges'])
        else:
            edges, img = ag.features.bedges_from_image(filename, return_original=True, **self.settings['bedges'])

        #s = self.settings['bedges'].copy()
        #if 'radius' in s:
        #    del s['radius']
        #edges_nospread = ag.features.bedges_from_image(filename, radius=0, **s)

        # How many patches could we extract?
        w, h = [edges.shape[i]-self.patch_size[i]+1 for i in range(2)]

        # TODO: Maybe shuffle an iterator of the indices?

        for sample in range(samples_per_image):
            for tries in range(20):
                x, y = random.randint(0, w-1), random.randint(0, h-1)
                selection = [slice(x, x+self.patch_size[0]), slice(y, y+self.patch_size[1])]
                # Return grayscale patch and edges patch
                edgepatch = edges[selection]
                #edgepatch_nospread = edges_nospread[selection]
                num = edgepatch[fr:-fr,fr:-fr].sum()
                if num >= self.settings['threshold']: 
                    the_patches.append(edgepatch)
                    #the_patches.append(edgepatch_nospread)
        
                    # The following is only for clearer visualization of the 
                    # patches. However, normalizing like this might be misleading
                    # in other ways.
                    vispatch = img[selection]
                    if 1:
                        pass
                    else:
                        vispatch = vispatch[...,:3].mean(axis=vispatch.ndim-1)
                    span = vispatch.min(), vispatch.max() 
                    if span[1] - span[0] > 0:
                        vispatch = (vispatch-span[0])/(span[1]-span[0])
                    the_originals.append(vispatch)
                    break

        return the_patches, the_originals


    def _preprocess_logs(self):
        """Pre-loads log values for easy extraction of parts from edge maps"""
        self._log_parts = np.log(self.parts)
        self._log_invparts = np.log(1-self.parts)
    
    def _extract_parts(self, edges, settings={}):

        partprobs = ag.features.code_parts(edges, self._log_parts, self._log_invparts, 
                                           self.settings['threshold'], self.settings['patch_frame'])
        parts = partprobs.argmax(axis=-1)

        # Zero-pad it to give back the same size. 
        # TODO: This might not be ideal in some situations, so we might want to reconsider this.
        parts = ag.util.zeropad(parts, (self._log_parts.shape[1]//2, self._log_parts.shape[2]//2))
        
        # Do spreading
        radii = settings['spread_radii']

        spread_parts = ag.features.spread_patches(parts, radii[0], radii[1], self.num_parts)
        return spread_parts 

    @classmethod
    def load_from_dict(cls, d):
        patch_size = d['patch_size']
        num_parts = d['num_parts']
        obj = cls(patch_size, num_parts)
        obj.parts = d['parts']
        obj.visparts = d['visparts']
        obj.settings = d['settings']
        obj._preprocess_logs()
        return obj

    def save_to_dict(self):
        return dict(num_parts=self.num_parts, patch_size=self.patch_size, parts=self.parts, visparts=self.visparts, settings=self.settings)
