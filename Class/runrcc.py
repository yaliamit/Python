import commands
import time
import manage_OUTPUT
import os

commands.getoutput('rm OUTPUTrcc.txt')
commands.getoutput('./GIT.sh')
commands.getoutput('ssh yaliamit@midway2.rcc.uchicago.edu \'cd Desktop/Dropbox/Python; git pull\'')
os.system('ssh yaliamit@midway2.rcc.uchicago.edu \'rm Desktop/Dropbox/Python/Class/OUTPUT.txt\'')
os.system('ssh yaliamit@midway2.rcc.uchicago.edu \'cd Desktop/Dropbox/Python/Class/; sbatch theano.sbatch \' & ')

ss='start'
while (ss != ''):
    time.sleep(30)
    ss=commands.getoutput('scp yaliamit@midway2.rcc.uchicago.edu:Desktop/Dropbox/Python/Class/OUTPUT.txt OUTPUTrcc.txt')
    print(ss)
while (commands.getoutput('grep DONE OUTPUTrcc.txt')==''):
    time.sleep(30)
    commands.getoutput('scp yaliamit@midway2.rcc.uchicago.edu:Desktop/Dropbox/Python/Class/OUTPUT.txt OUTPUTrcc.txt')
time.sleep(30)
pnn=commands.getoutput('grep NNN OUTPUTrcc.txt')
pnnn=str.split(pnn,':')
netname=str.strip(pnnn[1],' ,\')')

com='scp yaliamit@midway2.rcc.uchicago.edu:Desktop/Dropbox/Python/Class/'+netname+'.* _rcc/Amodels/.'
os.system(com)
manage_OUTPUT.print_OUTPUT('OUTPUTrcc')

