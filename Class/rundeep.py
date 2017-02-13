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
lOUTNAME=outname+'-dp.txt'
ss='python start_class.py '+' '.join(sys.argv[1:l-1]) + ' output='+outname+'>'+OUTNAME
if 'loc' in outname:
    os.system(ss)
else:
    f=open('runthingsDP.txt','w')
    f.write(ss+'\n')
    f.close()
    os.system('chmod +x runthingsDP.txt')
    remcom='rm Desktop/Dropbox/Python/Class/'+outname+'.txt'
    os.system('ssh amit@aitken.uchicago.edu' + ' ' + remcom)

    commands.getoutput('rm' + outname +'-dp.txt')
    commands.getoutput('./GIT.sh')
    commands.getoutput('ssh amit@aitken.uchicago.edu \'cd Desktop/Dropbox/Python; git pull\'')

    os.system('ssh amit@aitken.uchicago.edu \' ssh deeplearner; cd /ga/amit/Desktop/Dropbox/Python/Class/; ./runthingsDP.txt \' & ')

    ss='start'
    while (ss != ''):
        time.sleep(30)
        ss=commands.getoutput('scp amit@aitken.uchicago.edu:Desktop/Dropbox/Python/Class/'+OUTNAME+' '+ lOUTNAME)
        print(ss)
    while (commands.getoutput('grep DONE ' + lOUTNAME)==''):
        time.sleep(30)
        commands.getoutput('scp amit@aitken.uchicago.edu:Desktop/Dropbox/Python/Class/'+OUTNAME+' ' + lOUTNAME)

    time.sleep(30)
    pnn=commands.getoutput('grep NNN ' + lOUTNAME)
    pnnn=str.split(pnn,':')
    netname=str.strip(pnnn[1],' ,\')')

    com='scp amit@aitken.uchicago.edu:Desktop/Dropbox/Python/Class/'+netname+'.*  _dp/Amodels/.'
    os.system(com)
    manage_OUTPUT.print_OUTPUT(outname+'-dp')

