import subprocess as commands
import time
import os
import sys
import numpy as np

# print 'Number of arguments:', len(sys.argv), 'arguments.'
# print 'Argument List:', str(sys.argv)
l=len(sys.argv)
outname=sys.argv[-1]
OUTNAME=outname+'.txt'
lOUTNAME='_OUTPUTS/'+outname+'-rcc.txt'
ss='python run_conv.py '+' '.join(sys.argv[1:l-1])+'>'+OUTNAME
if 'loc' in outname:
    os.system(ss)
else:
    f=open('_scripts/runthings','w')
    f.write(ss+'\n')
    f.close()

    remcom='rm Desktop/Dropbox/Python/TF/'+outname+'.txt'
    os.system('ssh yaliamit@midway2.rcc.uchicago.edu' + ' ' + remcom)

    try:
        commands.check_output('rm ' + lOUTNAME, shell=True)
    except:
        print('Failed rm')

    try:
        commands.check_output('../Class/GIT.sh',shell=True)
    except:
        print('GIT failed')
    try:
        commands.check_output('ssh yaliamit@midway2.rcc.uchicago.edu \'cd Desktop/Dropbox/Python; git pull \'',shell=True)
    except:
        print('pull failed')


    os.system('ssh yaliamit@midway2.rcc.uchicago.edu \'cd Desktop/Dropbox/Python/TF/; sbatch TF.sbatch \' & ')

    ss='start'
    ny='no'
    while (ny != ''):
        time.sleep(10)
        try:
            ss = commands.check_output('scp yaliamit@midway2.rcc.uchicago.edu:Desktop/Dropbox/Python/TF/' +
                                    OUTNAME + ' ' + lOUTNAME,shell=True)
            print(ss)
            ny=''
        except:
            ny='no'
            print('Initial copy fialed')
        print(ss)
    done = False
    while (not done):
        time.sleep(10)
        try:
            ss = commands.check_output('grep DONE ' + lOUTNAME, shell=True)
            print('Done grep', ss)
            if (ss != ''):
                done = True
        except:
            done = False
        try:
            commands.check_output('scp yaliamit@midway2.rcc.uchicago.edu:Desktop/Dropbox/Python/TF/' + OUTNAME + ' ' + lOUTNAME,
                            shell=True)
        except:
            print('copy failed')

    time.sleep(5)

