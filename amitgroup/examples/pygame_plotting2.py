
import amitgroup as ag
import numpy as np

x, y = np.mgrid[0:1:100j, 0:1:100j]
plw = ag.plot.PlottingWindow()
t = 0.0
while plw.tick(60):
    im = np.sin(x + t) * np.cos(10*y + 10*t)
    plw.imshow(im, limits=(-1, 1))
    t += 1/60.
