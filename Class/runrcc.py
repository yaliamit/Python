import commands
import time
import manage_OUTPUT
import os
import sys

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

outname=sys.argv[1]
OUTNAME=outname+'.txt'
lOUTNAME=outname+'rcc.txt'
remcom='rm Desktop/Dropbox/Python/Class/'+outname+'.txt'
os.system('ssh yaliamit@midway2.rcc.uchicago.edu' + ' ' + remcom)

commands.getoutput('rm' + outname +'rcc.txt')
commands.getoutput('./GIT.sh')
commands.getoutput('ssh yaliamit@midway2.rcc.uchicago.edu \'cd Desktop/Dropbox/Python; git pull\'')

#os.system('ssh yaliamit@midway2.rcc.uchicago.edu \'cd Desktop/Dropbox/Python/Class/; sbatch theano.sbatch \' & ')

ss='start'
while (ss != ''):
    time.sleep(30)
    ss=commands.getoutput('scp yaliamit@midway2.rcc.uchicago.edu:Desktop/Dropbox/Python/Class/'+OUTNAME+' '+ lOUTNAME)
    print(ss)
while (commands.getoutput('grep DONE ' + outname+'rcc.txt')==''):
    time.sleep(30)
    commands.getoutput('scp yaliamit@midway2.rcc.uchicago.edu:Desktop/Dropbox/Python/Class/'+OUTNAME+' ' + lOUTNAME)

time.sleep(30)
pnn=commands.getoutput('grep NNN ' + lOUTNAME)
pnnn=str.split(pnn,':')
netname=str.strip(pnnn[1],' ,\')')

com='scp yaliamit@midway2.rcc.uchicago.edu:Desktop/Dropbox/Python/Class/'+netname+'.*  _rcc/Amodels/.'
os.system(com)
manage_OUTPUT.print_OUTPUT(outname+'rcc')

