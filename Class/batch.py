__author__ = 'amit'

from manage_OUTPUT import plot_OUTPUT as po
import pylab as py
import make_net
import numpy as np

#x=np.arange(-10,10,.1)

#y=make_net.rect_sym(x,scale_in=.1)
po('_RESULTS/_BP_CIFAR100/_fncrc/_baseline/_sgd/OUT5_SM-br','SM')

#py.plot(x,y)
py.show()
print("done")