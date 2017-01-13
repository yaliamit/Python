import numpy as np
import pylab as py
import sys
import manage_OUTPUT
from scipy import linalg
parms={}
parms=manage_OUTPUT.process_args(sys.argv,parms)

PAR=[]
PARW=None
PARR=None
num_nets=parms['stop']-parms['start']
for j in range(parms['start'],parms['stop']):
    PAR=np.load(parms['net']+'_'+str(j)+'.npy')
    if (PARW==None):
        PARW=PAR[parms['layer_number']]
        PARR=PAR[parms['layer_number']+1]
    else:
        PARW+=PAR[parms['layer_number']]
        PARR+=PAR[parms['layer_number']+1]
    # svw=linalg.svd(PARW/j)
    # svr=linalg.svd(PARR/j)
    # py.figure(2)
    # py.plot(svw[1])
    # py.plot(svr[1])
    # py.show()


PARW=PARW/num_nets
PARR=PARR/num_nets

py.ion()
svw=linalg.svd(PARW)
svr=linalg.svd(PARR)
dx=parms['dimx']
dy=parms['dimy']
if (dy>0):
    py.figure(1)
    for j in range(30):
        py.subplot(1,2,1)
        py.imshow(svw[0][:,j].reshape(dx,dy))
        py.subplot(1,2,2)
        py.imshow(svr[0][:,j].reshape(dx,dy))
else:
    py.subplot(1,2,1)
    py.imshow(PARW)
    py.subplot(1,2,2)
    py.imshow(PARR)
py.figure(2)
py.plot(svw[1])
py.plot(svr[1])
py.show()
print("done")