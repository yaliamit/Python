import os
import sys
import numpy as np

pfile='/fncrc'
nt=str(5000)
update='adam'
hinge='False'
mod_net='Null'
eta=str(.001)
ne=str(500)
use_existing='False'
for j in np.arange(3,10,1):

    seed=np.random.randint(0,200000)
    sd=' seed='+str(seed)+' '
    ss='python runber.py start_class.py net=_pars' + pfile +' hinge=' + hinge+' num_train='+nt+' use_existing='+use_existing+' eta_init='+eta+ \
        ' eta_current='+eta+' batch_size=5000 start=1 mult=1 mod_net='+mod_net+' output_net=junk update='+update+' num_epochs='+ne+ \
        ' force_global_prob=[1.,-1.]'+sd+'f_OUT'+str(j)
    os.system(ss)
    ss='python runber.py start_class.py net=_pars'+pfile+' hinge='+hinge+' num_train='+nt+' use_existing='+use_existing+' eta_init='+eta+ \
       ' eta_current='+eta+' batch_size=5000 start=1 mult=1 mod_net='+mod_net+' output_net=junk update='+update+' num_epochs='+ne+\
       ' force_global_prob=[1.,1.]'+sd+'f_R_OUT'+str(j)
    os.system(ss)
    ss='python runber.py start_class.py net=_pars'+pfile+' hinge=' + hinge+' num_train='+nt+' use_existing='+use_existing+' eta_init='+eta+ \
       ' eta_current='+eta+' batch_size=5000 start=1 mult=1 mod_net='+mod_net+' output_net=junk update='+update+' num_epochs='+ne+\
       ' force_global_prob=[1.,0.]'+sd+'f_RR_OUT'+str(j)
    os.system(ss)
    #ss='python runber.py start_class.py net=_pars'+pfile+' hinge=1. num_train='+nt+' use_existing=True eta_init=.001 ' \
    #   'eta_current=.001 batch_size=5000 NOT_TRAINABLE=[newdens1,] mod_net=modf_net start=1 mult=1 update+'+update+' num_epochs='+ne+ \
    #   ' force_global_prob=[1.,-1.]'+sd+'f_RRR_OUT'+str(j)
    #os.system(ss)


