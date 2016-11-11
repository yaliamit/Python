
import amitgroup as ag
import matplotlib.pylab as plt
import numpy as np

data = ag.io.load_example('two-faces')[1]

feat = ag.features.hog(data, (3, 3), (3, 3), num_bins=9)

plt.subplot(121)
plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
plt.subplot(122)
ag.plot.visualize_hog(feat, (21, 21), show=False)
plt.show()
