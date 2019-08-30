import subprocess as commands
import time
import os
import sys

commands.check_output('cp sys.argv[-2]' + '.txt ' + dirname + '/.')

try:
    commands.check_output('ssh yaliamit@midway2.rcc.uchicago.edu "cd /home/yaliamit/Desktop/Dropbox/Python; git pull" ', shell=True)
except:
    print('pull failed')