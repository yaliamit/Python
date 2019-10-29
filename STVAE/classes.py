import os
import socket
import sys
l=len(sys.argv)
opt=''
if l>1:
    opt=sys.argv[1]
OPT=''
if (opt=='OPT'):
    OPT='--OPT'
if ('marx' in socket.gethostname()):
    scr = 'mrunber.py'
else:
    scr = 'runber.py'

for cl in range(10):
    com = 'python _scripts/'+scr+' main_opt.py _pars/pars_cl '+OPT+' --cl='+str(cl)+' OUT_'+opt+'_'+str(cl)
    print(com)
    os.system(com)
    print(cl)

com = 'python _scripts/'+scr+' main_opt.py _pars/pars_cl '+OPT+' --nti=500 --num_train=1000 --mb_size=500 --classify OUT_'+opt
print(com)
os.system(com)