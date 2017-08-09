import os
import sys
import numpy as np

for j in np.arange(0,10,1):

    seed=np.random.randint(0,200000)
    sd=' seed='+str(seed)+' '
    ss='python runber.py net=_pars/fncrc use_existing=True eta_init=.001 eta_current=.001 batch_size=5000 start=1 mult=1 mod_net=modf_net update=adam num_epochs=200 force_global_prob=[1.,0.]'+sd+'fncrc_RR_2c_c_OUT'+str(j)
    os.system(ss)
    ss='python runber.py net=_pars/fncrc use_existing=True eta_init=.001 eta_current=.001 batch_size=5000 start=1 mult=1 mod_net=modf_net update=adam num_epochs=200 force_global_prob=[1.,-1.]'+sd+'fncrc_2c_c_OUT'+str(j)
    os.system(ss)
    ss='python runber.py net=_pars/fncrc use_existing=True eta_init=.001 eta_current=.001 batch_size=5000 start=1 mult=1 mod_net=modf_net update=adam num_epochs=200 force_global_prob=[1.,1.]'+sd+'fncrc_R_2c_c_OUT'+str(j)
    os.system(ss)
    #ss='python runber.py net=_pars/fncrc use_existing=True start=1 mult=1 mod_net=modf_net num_epochs=200 use_existing=True start=1 mult=1 mod_net=modf_net global_prob=[1.,-1.]'+sd+'fncrc_L_2c_s_OUT'+str(j)
    #os.system(ss)

