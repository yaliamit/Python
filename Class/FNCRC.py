import os
import sys
import numpy as np

for j in range(10):

    seed=np.random.randint(0,200000)
    sd=' seed='+str(seed)+' '
    ss='python runber.py net=_pars/fncrc global_prob=[1.,-1.]'+sd+'fncrc_OUT'+str(j)
    os.system(ss)
    ss='python runber.py net=_pars/fncrc global_prob=[1.,0.]'+sd+'fncrc_RR_OUT'+str(j)
    os.system(ss)
    ss='python runber.py net=_pars/fncrc global_prob=[1.,1.]'+sd+'fncrc_R_OUT'+str(j)
    os.system(ss)
    ss='python runber.py net=_pars/fncrc use_existing=True start=1 mult=1 mod_net=modf_net global_prob=[1.,-1.]'+sd+'fncrc_L_OUT'+str(j)
    os.system(ss)

