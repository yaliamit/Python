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
lOUTNAME=outname+'-br.txt'
ss='python start_class.py '+' '.join(sys.argv[1:l-1]) + ' output='+outname+'>'+OUTNAME
if 'loc' in outname:
    os.system(ss)
else:
    f=open('runthingsBR.txt','w')
    f.write(ss+'\n')
    f.close()

    remcom='rm /ga/amit/Desktop/Dropbox/Python/Class/'+outname+'.txt'
    os.system('ssh amit@bernie.uchicago.edu' + ' ' + remcom)

    commands.getoutput('rm' + outname +'-br.txt')
    commands.getoutput('./GIT.sh')
    commands.getoutput('ssh amit@bernie.uchicago.edu \'cd /ga/amit/Desktop/Dropbox/Python; git pull\'')

    os.system('ssh amit@bernie.uchicago.edu \'cd /ga/amit/Desktop/Dropbox/Python/Class/; ./runthingsBR.txt \' & ')

    ss='start'
    while (ss != ''):
        time.sleep(30)
        ss=commands.getoutput('scp amit@bernie.uchicago.edu:/ga/amit/Desktop/Dropbox/Python/Class/'+OUTNAME+' '+ lOUTNAME)
        print(ss)
    while (commands.getoutput('grep DONE ' + lOUTNAME)==''):
        time.sleep(30)
        commands.getoutput('scp amit@bernie.uchicago.edu:/ga/amit/Desktop/Dropbox/Python/Class/'+OUTNAME+' ' + lOUTNAME)

    time.sleep(30)
    pnn=commands.getoutput('grep NNN ' + lOUTNAME)
    pnnn=str.split(pnn,':')
    netname=str.strip(pnnn[1],' ,\')')

    com='scp amit@bernie.uchicago.edu:/ga/amit/Desktop/Dropbox/Python/Class/'+netname+'.txt  _br/Amodels/.'
    os.system(com)
    manage_OUTPUT.print_OUTPUT(outname+'-br')

