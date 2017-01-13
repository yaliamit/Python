import numpy as np
import pylab as py
import sys
import manage_OUTPUT
parms={}
parms=manage_OUTPUT.process_args(sys.argv,parms)

PAR=[]
PARW=None
PARR=None
num_nets=parms['stop']-parms['start']
for j in range(parms['start'],parms['stop']):
    PAR[j]=np.load(parms['net']+'_'+str(j)+'.npy')
    if (PARW==None):
        PARW=PAR[j][0]
        PARR=PAR[j][1]
    else:
        PARW+=PAR[j][0]
        PARR+=PAR[j][1]

PARW=PARW/num_nets
PARR=PARR/num_nets

