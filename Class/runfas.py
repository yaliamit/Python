import commands
import time
import manage_OUTPUT
import os
commands.getoutput('rm OUTPUT.txt')
commands.getoutput('./GIT.sh')
commands.getoutput('ssh yaliamit@fasolt.cs.uchicago.edu \'cd Desktop/Dropbox/Python; git pull\'')
os.system('ssh yaliamit@fasolt.cs.uchicago.edu \'cd Desktop/Dropbox/Python/Class/; ./runthings.txt\' & ')

ss='start'
while (ss is not ''):
    ss=commands.getoutput('scp yaliamit@fasolt.cs.uchicago.edu:Desktop/Dropbox/Python/Class/OUTPUT.txt .')
    print(ss)
while (commands.getoutput('grep Test OUTPUT.txt')==''):
    time.sleep(30)
    commands.getoutput('scp yaliamit@fasolt.cs.uchicago.edu:Desktop/Dropbox/Python/Class/OUTPUT.txt .')

manage_OUTPUT.print_OUTPUT()

