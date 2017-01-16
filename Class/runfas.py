import commands
import time
import manage_OUTPUT
import os
import sys

# print 'Number of arguments:', len(sys.argv), 'arguments.'
# print 'Argument List:', str(sys.argv)
l=len(sys.argv)
outname=sys.argv[-1]
OUTNAME=outname+'.txt'
lOUTNAME=outname+'-cs.txt'
ss='python start_class.py '+' '.join(sys.argv[1:l-1]) + ' output='+outname+'>'+OUTNAME
if 'loc' in outname:
    os.system(ss)
else:
    f=open('runthingsCS.txt','w')
    f.write(ss+'\n')
    f.close()

    remcom='rm Desktop/Dropbox/Python/Class/'+outname+'.txt'
    os.system('ssh yaliamit@linux2.cs.uchicago.edu' + ' ' + remcom)

    commands.getoutput('rm' + outname +'-cs.txt')
    commands.getoutput('./GIT.sh')
    commands.getoutput('ssh yaliamit@linux2.cs.uchicago.edu \'cd Desktop/Dropbox/Python; git pull\'')

    os.system('ssh yaliamit@linux2.cs.uchicago.edu \'cd Desktop/Dropbox/Python/Class/; sbatch theano.sbatch \' & ')

    ss='start'
    while (ss != ''):
        time.sleep(30)
        ss=commands.getoutput('scp yaliamit@linux2.cs.uchicago.edu:Desktop/Dropbox/Python/Class/'+OUTNAME+' '+ lOUTNAME)
        print(ss)
    while (commands.getoutput('grep DONE ' + lOUTNAME)==''):
        time.sleep(30)
        commands.getoutput('scp yaliamit@linux2.cs.uchicago.edu:Desktop/Dropbox/Python/Class/'+OUTNAME+' ' + lOUTNAME)

    time.sleep(30)
    pnn=commands.getoutput('grep NNN ' + lOUTNAME)
    pnnn=str.split(pnn,':')
    netname=str.strip(pnnn[1],' ,\')')

    com='scp yaliamit@linux2.cs.uchicago.edu:Desktop/Dropbox/Python/Class/'+netname+'.*  _cs/Amodels/.'
    os.system(com)
    manage_OUTPUT.print_OUTPUT(outname+'-cs')

