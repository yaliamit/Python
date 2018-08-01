import time
import os
import shutil
import subprocess as commands
import sys

def process_args(args,parms):
    print(args)

    for ss in args:
        addname=None
        if (':' in ss):
            bb=str.split(ss,':')
            addname=str.strip(bb[0],' ')
            s=bb[1]
            if (addname in parms):
                pp=parms[addname]
                addname=None
            else:
                pp={}
        else:
            s=ss
            pp=parms
        if ('=' in s):
            aa=str.split(s,'=')
            name=str.strip(aa[0],' ')
            value=str.strip(aa[1],' ')
            if ('[' in value):
                aa=str.split(str.strip(value,' []\n'),',')
                a=[]
                try:
                    int(aa[0])
                    for aaa in aa:
                        a.append(int(aaa))
                    v=tuple(a)
                except ValueError:
                    try:
                        float(aa[0])
                        for aaa in aa:
                            a.append(float(aaa))
                        v=tuple(a)
                    except ValueError:
                        for aaa in aa:
                            a.append(aaa)
                        v=tuple(a)
            else:
                try:
                    v=int(value)
                except ValueError:
                    try:
                        v=float(value)
                    except:
                        v=value
            if (v=='True'):
                pp[name]=True
            elif (v=='False'):
                pp[name]=False
            else:
                pp[name]=v
        if (addname is not None):
            parms[addname]=pp
    print(parms)
    return(parms)

def print_OUTPUT(name='OUTPUT'):

    ss=str.split(name,'-')
    OO='AOUT'
    if (len(ss)>1):
        if (ss[1] != ''):
            OO='_'+ss[1]+'/AOUT'

    dirs=os.listdir(OO)
    #cc=commands.getoutput('grep XXX ' + name + '.txt')
    cc=commands.check_output('grep XXX ' + name + '.txt',shell=True)

    nn=str.split(cc,' ')
    nna=nn[1]
    nnb=str.split(nna,'/')
    if (len(nnb)==2):
        nnn=nnb[1]
    else:
        nnn=nnb[0]
    ss=nnn
    t=0
    for dd in dirs:
        if ss in dd:
            t+=1
    #sys.stdout.flush()
    #time.sleep(10)
    shutil.copyfile(name+'.txt',OO+'/'+name+'_'+str(t)+'.txt')


def  plot_OUT(s):

    import numpy as np

    import pylab as py
    py.ion()
    aaa=commands.check_output('grep ERR ' + s + '.txt | cut -d" " -f2-4')
    bt=np.fromstring(aaa,sep='\n\t\t\t')
    btt=bt.reshape((-1,3))
    py.plot(btt)


def plot_OUTPUT(name='OUTPUT',code='',first=None,last=None):
    #import commands
    import numpy as np
    import pylab as py
    py.ion()
    havetrain=False
    oo=commands.check_output('grep Posi ' + name + '.txt  | cut -d" " -f2,3', shell=True)
    bp=[]
    #bp=np.fromstring(oo,sep='\n\t\t\t')
    #bp=bp.reshape((-1,2))
    bt=np.fromstring(commands.check_output('grep Train ' + name + '.txt | grep acc | cut -d":" -f2',shell=True),sep='\n\t\t\t')
    loss=np.fromstring(commands.check_output('grep Train ' + name + '.txt | grep loss | cut -d":" -f2',shell=True),sep='\n\t\t\t')
    fig=py.figure(2)
    py.plot(loss)
    py.figure(1)
    bv=np.fromstring(commands.check_output('grep Val ' + name + '.txt | grep acc | cut -d":" -f2',shell=True),sep='\n\t\t\t')
    ss='grep aggegate ' + name + '.txt'
    if (len(commands.check_output(ss,shell=True))):
        ss='grep aggegate ' + name + '.txt | cut -d"," -f4 | cut -d")" -f1'
        atest=np.fromstring(commands.check_output(ss,shell=True),sep='\n\t\t\t')
        if (type(atest) is np.ndarray and len(atest)>0 ):
            atest=atest[-1]
        ss='grep Post-train ' + name + '.txt | grep acc | cut -d":" -f2'
        atrain=np.fromstring(commands.check_output(ss,shell=True),sep='\n\t\t\t')
        if (type(atrain) is np.ndarray and len(atrain)>0):
            havetrain=True
            atrain=atrain[-1]
        if (havetrain):
            print(atest,atrain)
    if (first is not None and last is not None):
        bt=bt[first:last]
        bv=bv[first:last]
        if (bp!=[]):
            bp=bp[first:last]
        print(bv[-1],bt[-1])
    else:
        print(len(bt),bv[-1],bt[-1])
        if (havetrain>0):
            py.plot(len(bt)-2, atest, 'go', markersize=4)
            py.plot(len(bt)-2, atrain, 'bo', markersize=4)
    py.plot(bt,label='train '+code)
    py.plot(bv,label='val '+code)
    if (bp!=[]):
        py.plot(bp,label='Pos')
    py.legend(loc=4)

    py.show()