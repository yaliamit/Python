import commands
import time
import manage_OUTPUT
import os

commands.getoutput('rm OUTPUTrcc.txt')
commands.getoutput('./GIT.sh')
commands.getoutput('ssh yaliamit@midway2.rcc.uchicago.edu \'cd Desktop/Dropbox/Python; git pull\'')
os.system('ssh yaliamit@midway2.rcc.uchicago.edu \'cd Desktop/Dropbox/Python/Class/; sbatch theano.sbatch \' & ')
#
ss='start'
while (ss != ''):
    ss=commands.getoutput('scp yaliamit@midway2.rcc.uchicago.edu:Desktop/Dropbox/Python/Class/OUTPUT.txt OUTPUTrcc.txt')
    print(ss)
while (commands.getoutput('grep Test OUTPUTrcc.txt')==''):
    time.sleep(30)
    commands.getoutput('scp yaliamit@midway2.rcc.uchicago.edu:Desktop/Dropbox/Python/Class/OUTPUT.txt OUTPUTrcc.txt')

manage_OUTPUT.print_OUTPUT('OUTPUTrcc')

