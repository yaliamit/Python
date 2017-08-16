import os
import sys
import numpy as np

pfile='/fncrc'
nt=str(5000)
update='adam'
mod_net=''
ne=str(500)
for j in np.arange(0,1,1):

    seed=np.random.randint(0,200000)
    sd=' seed='+str(seed)+' '
    ss='python runber.py start_class.py net=_pars'+pfile+' hinge=1. num_train='+nt+' use_existing=True eta_init=.001 ' \
        'eta_current=.001 batch_size=5000 start=1 mult=1 mod_net=modf_net update='+update+' num_epochs='+ne+ \
        ' force_global_prob=[1.,-1.]'+sd+'f_OUT'+str(j)
    os.system(ss)
    ss='python runber.py start_class.py net=_pars'+pfile+' hinge=1. num_train='+nt+' use_existing=True eta_init=.001 ' \
       'eta_current=.001 batch_size=5000 start=1 mult=1 mod_net=modf_net update='+update+' num_epochs='+ne+\
       ' force_global_prob=[1.,1.]'+sd+'f_R_OUT'+str(j)
    os.system(ss)
    ss='python runber.py start_class.py net=_pars'+pfile+' hinge=1. num_train='+nt+' use_existing=True eta_init=.001 ' \
       'eta_current=.001 batch_size=5000 start=1 mult=1 mod_net=modf_net update='+update+' num_epochs='+ne+\
       ' force_global_prob=[1.,0.]'+sd+'f_RR_OUT'+str(j)
    os.system(ss)
    #ss='python runber.py start_class.py net=_pars'+pfile+' hinge=1. num_train='+nt+' use_existing=True eta_init=.001 ' \
    #   'eta_current=.001 batch_size=5000 NOT_TRAINABLE=[newdens1,] mod_net=modf_net start=1 mult=1 update+'+update+' num_epochs='+ne+ \
    #   ' force_global_prob=[1.,-1.]'+sd+'f_RRR_OUT'+str(j)
    os.system(ss)


