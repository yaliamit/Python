from __future__ import division, print_function, absolute_import
import numpy as np
import scipy.stats
import math
import amitgroup as ag

def visualize_hog(hog_features, cell_size, show=True, polarity_sensitive=True, phase=0.0, direction=1.0):
    import matplotlib.pylab as plt

    # We'll construct an image and then show it
    img = np.zeros(tuple([hog_features.shape[i] * cell_size[i] for i in range(2)]))

    num_bins = hog_features.shape[2] 

    x, y = np.mgrid[-0.5:0.5:cell_size[0]*1j, -0.5:0.5:cell_size[1]*1j]
    #circle = (x**2 + y**2) <= 0.28
    box = ag.util.zeropad(np.ones((cell_size[0]-2, cell_size[1]-2)), 1)

    arrows = np.empty((num_bins,) + cell_size)
    
    # Generate the lines that will indicate directions
    for d in range(num_bins):
        # v is perpendicular to the gradient (thus visualizing an edge)
        if polarity_sensitive:
            eff_num_bins = num_bins
        else:
            eff_num_bins = num_bins*2
    
        angle = direction * d*2*math.pi/eff_num_bins + phase
        v = np.array([math.cos(math.pi/2 + angle), -math.sin(math.pi/2 + angle)])
        # project our location onto this line and run that through a guassian (basically, drawing a nice line)
        arrows[d] = scipy.stats.norm.pdf(v[0] * x + v[1] * y, scale=0.07)

    #arrows[:] *= circle
    arrows *= box

    # We're only visualizing the max in each cell
    vis_features = hog_features.max(axis=-1)

    for x in range(hog_features.shape[0]):
        for y in range(hog_features.shape[1]):
            for angle in range(num_bins):
                img[x*cell_size[0]:(x+1)*cell_size[0],y*cell_size[1]:(y+1)*cell_size[1]] += arrows[angle] * vis_features[x, y, angle]
    
    plt.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    
    if show:
        plt.show()

def visualize_hog_color(hog_features, cell_size, show=True, polarity_sensitive=True, phase=0.0, direction=1.0):
    import matplotlib.pylab as plt

    # We'll construct an image and then show it
    img = np.zeros(tuple([hog_features.shape[i] * cell_size[i] for i in range(2)]) + (3,), dtype=np.float64)

    num_bins = hog_features.shape[2] 

    x, y = np.mgrid[-0.5:0.5:cell_size[0]*1j, -0.5:0.5:cell_size[1]*1j]
    circle = (x**2 + y**2) <= 0.28
    box = ag.util.zeropad(np.ones((cell_size[0]-2, cell_size[1]-2)), 1)

    arrows = np.empty((num_bins,) + cell_size)
    
    # Generate the lines that will indicate directions
    for d in range(num_bins):
        # v is perpendicular to the gradient (thus visualizing an edge)
        if polarity_sensitive:
            eff_num_bins = num_bins
        else:
            eff_num_bins = num_bins*2
    
        angle = direction * d*2*math.pi/eff_num_bins + phase
        v = np.array([math.cos(math.pi/2 + angle), -math.sin(math.pi/2 + angle)])
        # project our location onto this line and run that through a guassian (basically, drawing a nice line)
        arrows[d] = scipy.stats.norm.pdf(v[0] * x + v[1] * y, scale=0.07)

    arrows /= arrows.max()

    #arrows[:] *= circle
    arrows *= box

    # We're only visualizing the max in each cell
    vis_features = hog_features.max(axis=-1)

    arrows = np.expand_dims(arrows, -1)

    for x in range(hog_features.shape[0]):
        for y in range(hog_features.shape[1]):
            for angle in range(num_bins):
                mag = vis_features[x, y, angle]
                #mag = np.max(vis_features[x, y, angle]*2 - 1, 0)
                #ch = 1+angle%2#0#[0, 1][mag > 0.5]
                ch = [(0.5, 0.0, 1.0),
                      (1.0, 0.0, 0.0),
                      (0.5, 1.0, 0.0),
                      (0.0, 1.0, 1.0)][(-angle)%4]
                        
        #, [1, 2], [0, 2], [0, 1, 2]][angle]
                selection = (slice(x*cell_size[0], (x+1)*cell_size[0]),
                             slice(y*cell_size[1], (y+1)*cell_size[1]))
                
                img[selection] = np.maximum(img[selection], arrows[angle] * mag * ch)

    plt.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    
    if show:
        plt.show()
