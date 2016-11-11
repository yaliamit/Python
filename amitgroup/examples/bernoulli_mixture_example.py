import amitgroup as ag
import numpy as np

#ag.set_verbose(True)
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


bm = ag.stats.BernoulliMixture(3,XS)
bm.run_EM(1e-8, debug_plot=False)

ag.plot.images(bm.templates)
