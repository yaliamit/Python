# Randomly deforms an image.

import amitgroup as ag
import numpy as np

def main():
    np.random.seed(0)
    F = ag.io.load_example('two-faces')[0]
    images = [F]
    for i in range(8):
        imdef = ag.util.DisplacementFieldWavelet(F.shape, 'db4')
        imdef.randomize(0.01)
        Fdef = imdef.deform(F) 
        images.append(Fdef)
    
    ag.plot.images(images)

if __name__ == '__main__':
    main()
