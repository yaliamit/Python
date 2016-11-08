from __future__ import division
import numpy as np
import scipy.signal as sig
import scipy.stats
from itertools import product
import math
import amitgroup as ag
import amitgroup.util


# TODO: Under development
def hog(data, cell_size, block_size, num_bins=9):
    assert len(cell_size) == 2, "cell_size must be length 2"
    assert len(block_size) == 2, "block_size must be length 2"
    gradkernel = np.array([[-1, 0, 1]])

    mode = 'same'
    grad1 = sig.convolve(data, gradkernel, mode=mode)
    grad2 = sig.convolve(data, gradkernel.T, mode=mode)

    grad1[:,0] = grad1[:,-1] = grad2[0] = grad2[-1] = 0
    
    offset = cell_size

    angles = np.arctan2(grad2, grad1)
    # This converts the angle grid to a grid of bin indices
    discrete_angles = (angles*num_bins/(2*np.pi)).round().astype(int) % num_bins
    amplitudes = np.sqrt(grad1**2 + grad2**2)
    
    # Now, combine these into a histogram per pixel (with only one response)
    histograms = np.zeros(data.shape[:2] + (num_bins,))
    
    # TODO: This can be optimized.
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            histograms[x,y,discrete_angles[x,y]] = amplitudes[x,y]

    num_cells_per_block = np.prod(block_size)
    cells_shape = tuple([data.shape[i]//cell_size[i] for i in range(2)])
    blocks_shape = tuple([cells_shape[i]-block_size[i]+1 for i in range(2)])

    w, h = (cell_size[0]-1)/2, (cell_size[1]-1)/2
    
    features = np.zeros(blocks_shape + (num_bins, num_cells_per_block)) 
    
    block_pixel_size = tuple([block_size[i]*cell_size[i] for i in range(2)])
    mbx, mby = np.mgrid[0:block_pixel_size[0], 0:block_pixel_size[1]]
    block_gaussian = scipy.stats.norm.pdf(
        np.sqrt((mbx - (block_pixel_size[0]-1)/2)**2 + (mby - (block_pixel_size[1]-1)/2)**2), 
        scale=0.1*block_pixel_size[0])
    
    
    # First, create histograms for each pixel
    for bx in range(blocks_shape[0]):
        for by in range(blocks_shape[1]):
            b = bx, by
            selection = [slice(b[i]*cell_size[i], (b[i]+block_size[i])*cell_size[i]) for i in range(2)]
            
            # Extract block and add gaussian
            block_hist = histograms[selection] * block_gaussian.reshape(block_gaussian.shape + (1,))
            
            # And subsample it, so that each cell becomes a pixel
            #im = scipy.misc.imresize(histograms[selection], block_size)
            im = np.zeros((num_bins, num_cells_per_block))
            for px in range(block_hist.shape[0]):
                for py in range(block_hist.shape[1]):
                    # TODO: This is a mess. Handle corner cases in a better and faster way.
                    cx = (px-w)//cell_size[0]
                    cy = (py-h)//cell_size[1]
                    if px <= w:
                        x1 = 1.0
                    elif px >= block_size[0]*cell_size[0] - 1 - w:
                        x1 = 0.0
                    else:
                        x1 = (px - w) % cell_size[0]
                    x0 = 1.0 - x1
                        
                    if py <= h:
                        y1 = 1.0
                    elif py >= block_size[1]*cell_size[1] - 1 - h:
                        y1 = 0.0
                    else:
                        y1 = (py - h) % cell_size[1]
                    y0 = 1.0 - y1
                    
                    v = x0 * y0
                    if v > 0:
                        im[:, cx + cy*block_size[0]] += v * block_hist[px,py]
                        
                    v = x1 * y0
                    if v > 0:
                        im[:, cx+1 + cy*block_size[0]] += v * block_hist[px,py]
                    
                    v = x0 * y1
                    if v > 0:
                        im[:, cx + (cy+1)*block_size[0]] += v * block_hist[px,py]
                        
                    v = x1 * y1
                    if v > 0:
                        im[:, cx+1 + (cy+1)*block_size[0]] += v * block_hist[px,py]
            
            # Now, normalize this
            im /= np.sqrt(np.dot(im.flatten(), im.flatten()) + 0.01)
    
            features[bx, by] = im
            
    return features
