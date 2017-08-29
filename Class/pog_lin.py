import os
import sys
import numpy as np

pfile='/pogR_lin'
nt=str(50000)
update='adam'
eta=str(.001)
ne=str(200)
use_existing='True'
for j in np.arange(0,10,1):

    seed=np.random.randint(0,200000)
    sd=' seed='+str(seed)+' '
    ss='python runber.py start_class.py net=_pars' + pfile +' hinge=1. num_train='+nt+' eta_init='+eta+ \
        ' eta_current='+eta+' batch_size=100 num_epochs='+ne+' update='+update+ \
        ' force_global_prob=[1.,-1.]'+sd+'pog_OUT'+str(j)
    os.system(ss)
    ss='python runber.py start_class.py net=_pars'+pfile+' hinge=1. num_train='+nt+' use_existing='+use_existing+' eta_init='+eta+ \
       ' eta_current='+eta+' batch_size=100 start=1 mult=1 mod_net=mod_pog update='+update+' num_epochs='+ne+\
       ' force_global_prob=[1.,-1.] eta_schedule=[100.,.0005]'+sd+'apog_OUT'+str(j)
    os.system(ss)
    ss='python runber.py start_class.py net=_pars'+pfile+' hinge=1. num_train='+nt+' use_existing='+use_existing+' eta_init='+eta+ \
       ' eta_current='+eta+' batch_size=5000 start=1 mult=1 mod_net=mod_pog_direct update='+update+' num_epochs='+ne+\
       ' force_global_prob=[1.,-1] eta_schedule=[100.,.0005]'+sd+'dapog_OUT'+str(j)
    os.system(ss)
    #ss='python runber.py start_class.py net=_pars'+pfile+' hinge=1. num_train='+nt+' use_existing=True eta_init=.001 ' \
    #   'eta_current=.001 batch_size=5000 NOT_TRAINABLE=[newdens1,] mod_net=modf_net start=1 mult=1 update+'+update+' num_epochs='+ne+ \
    #   ' force_global_prob=[1.,-1.]'+sd+'f_RRR_OUT'+str(j)
    #os.system(ss)


