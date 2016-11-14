import numpy as np
import commands
import pylab as py

def many():
    ssp=commands.getoutput('grep -n trans $(ls -t -r OUTPUT.txt) | cut -d" " -f2 | nl')
    print(ssp)
    ss=commands.getoutput('grep rotations $(ls -t -r OUTPUT.txt) | cut -d" " -f5')
    bt=np.fromstring(ss,sep='\n')
    st=np.array(range(len(bt)))+1
    py.plot(st,bt)
    py.xticks(np.arange(1,len(bt)+1,1))
    py.grid(True)
    py.show()
    print(bt)

def one(name="OUTPUT.txt",name1=None):
    ssv=commands.getoutput('grep "Val acc" '+ name + ' | cut -d":" -f2 ')
    sst=commands.getoutput('grep "Train acc" '+ name + ' | cut -d":" -f2 ')
    ssl=commands.getoutput('grep "Train loss" '+ name + ' | cut -d":" -f2 ')
    sste=commands.getoutput('grep "Test acc" ' + name + ' | cut -d":" -f2 ')
    print 'sste',sste
    bv=np.fromstring(ssv,sep='\n')
    bt=np.fromstring(sst,sep='\n')
    bl=np.fromstring(ssl,sep='\n')
    bte=None
    bte1=None
    if sste != '':
        bte=np.fromstring(sste,sep='\n')
    if (name1 is not None):
        ssv1=commands.getoutput('grep "Val acc" '+ name1 + ' | cut -d":" -f2 ')
        sst1=commands.getoutput('grep "Train acc" '+ name1 + ' | cut -d":" -f2 ')
        ssl1=commands.getoutput('grep "Train loss" '+ name1 + ' | cut -d":" -f2 ')
        sste1=commands.getoutput('grep "Test acc" ' + name1 + ' | cut -d":" -f2 ')
        bv1=np.fromstring(ssv1,sep='\n')
        bt1=np.fromstring(sst1,sep='\n')
        bl1=np.fromstring(ssl1,sep='\n')
        st1=np.array(range(len(bv1)))+1
        if sste1 is not '':
          bte1=np.fromstring(sste1,sep='\n')

    st=np.array(range(len(bv)))+1


    py.figure(1)
    py.subplot(1,2,1)
    py.plot(st,bv)
    py.plot(st,bt)
    if (bte is not None and len(bte)):
        print bte
        py.plot(st,np.repeat(bte,len(st)),color='r')
    if (name1 is not None):
        py.plot(st1,bv1)
        py.plot(st1,bt1)
        if (bte1 is not None and len(bte1)):
            py.plot(st,np.repeat(bte1,len(st)),color='r')
    py.subplot(1,2,2)
    py.plot(st,bl)
    if (name1 is not None):
        py.plot(st1,bl1)

    #py.xticks(np.arange(1,len(bv)+1,1))
    #py.grid(True)
    py.show()
   # print(bt)


