import numpy as np
import commands
import pylab as py


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