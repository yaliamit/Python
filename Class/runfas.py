import commands
import time
import manage_OUTPUT

commands.getoutput('./GIT.sh')
commands.getoutput('ssh yaliamit@fasolt.cs.uchicago.edu \'cd Desktop/Dropbox/Python; git pull\'')
commands.getoutput('ssh yaliamit@fasolt.cs.uchicago.edu \'cd Desktop/Dropbox/Python/Class/; ./runthings.txt\'')

while (commands.getoutput('grep Test OUTPUT.txt')==''):
    time.sleep(30)
    commands.getoutput('scp yaliamit@fasolt.cs.uchicago.edu:Desktop/Dropbox/Python/Class/OUTPUT.txt .')

manage_OUTPUT.print_OUTPUT()

