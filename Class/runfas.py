import commands
import time


commands.getoutput('./GIT.sh')
commands.getoutput('scp runthings.txt yaliamit@fasolt.cs.uchicago.edu:/home/yaliamit/Desktop/Dropbox/Python/Class')
#commands.getoutput('ssh yaliamit@fasolt.cs.uchicago.edu \'cd /home/yaliamit/Desktop/Dropbox/Python/Class/; ./runthings.txt\'')

# while (commands.getoutput('grep Test OUTPUT.txt')==''):
#     time.sleep(30)
#     commands.getoutput('scp yaliamit@fasolt.cs.uchicago.edu:/home/yaliamit/Desktop/Dropbox/Python/Class/OUTPUT.txt .')
#
#
#
