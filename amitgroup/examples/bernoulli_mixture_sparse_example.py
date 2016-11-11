import amitgroup as ag
import numpy as np

ag.set_verbose(True)
np.random.seed(0)

# generate synthetic data
num_templates = 3
template_size = (20,20)
templates = np.zeros((num_templates,
                      template_size[0],
                      template_size[1]))
for T in templates:
    x = np.random.randint(template_size[0]-4)
    y = np.random.randint(template_size[1]-4)
    T[x:x+4,
      y:y+4] = .95
    T = np.maximum(.05,T)

for T in templates:
    T = np.maximum(.05,T)


num_data = 100
XS = np.zeros((num_data,template_size[0],template_size[1]))

for X in XS:
    S = np.random.rand(template_size[0],template_size[1])
    X[S < templates[np.random.randint(num_templates)]] = 1.


if 1:
    import scipy.sparse
    X_flat = XS.reshape((XS.shape[0], -1))
    X_flat = np.matrix(X_flat)
    X_flat = scipy.sparse.csc_matrix(X_flat)
    bm = ag.stats.BernoulliMixture(3,X_flat)
    bm.run_EM(1e-8, debug_plot=False)
    print 'weights', bm.weights
    res = bm.remix(XS)
else:
    bm = ag.stats.BernoulliMixture(3,XS)
    bm.run_EM(1e-8, debug_plot=False)
    res = bm.templates


ag.plot.images(res)
