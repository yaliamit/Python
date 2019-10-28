import os
import socket

if ('marx' in socket.gethostname()):
    scr = 'mrunber.py'
else:
    scr = 'runber.py'

for cl in range(10):
    com = 'python _scripts/'+scr+' main_opt.py _pars/pars_tvae --type=tvae --n_mix=2 --sdim=18 --num_train=100 --num_hlayers=1 --nepoch=200 --transformation=tps --hdim_dec=64 --mb_size=100 --CONS --cl='+str(cl)+' OUT'
    print(com)
    os.system(com)
    print(cl)