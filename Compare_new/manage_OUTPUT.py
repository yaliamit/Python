import time
import os
import shutil
import sys

def process_args(args,parms):
    print(args)
    for s in args:
        if ('=' in s):
            aa=str.split(s,'=')
            name=str.strip(aa[0],' ')
            value=str.strip(aa[1],' ')
            try:
                v=int(value)
            except ValueError:
                v=value
            if (v=='True'):
                parms[name]=True
            elif (v=='False'):
                parms[name]=False
            else:
                parms[name]=v
    print(parms)
    return(parms)

def print_OUTPUT(NETPARS):

    dirs=os.listdir('AOUT')
    nn=str.split(NETPARS['output_net'],'/',)
    nnn=nn[0]
    if (len(nn)==2):
        nnn=nn[1]

    ss='OUTPUT_'+nnn
    t=0
    for dd in dirs:
        if ss in dd:
            t+=1
    sys.stdout.flush()
    time.sleep(10)
    shutil.copyfile('OUTPUT.txt','AOUT/OUTPUT_'+nnn+'_'+str(t)+'.txt')

def plot_OUTPUT():
    import commands
    import numpy as np
    import pylab as py
    bt=np.fromstring(commands.getoutput('grep Train OUTPUT.txt | grep acc | cut -d":" -f2'),sep='\n\t\t\t')
    bv=np.fromstring(commands.getoutput('grep Val OUTPUT.txt | grep acc | cut -d":" -f2'),sep='\n\t\t\t')
    print(len(bt))
    py.plot(bt,label='train')

    py.plot(bv,label='val')
    py.legend(loc=4)
    py.show()