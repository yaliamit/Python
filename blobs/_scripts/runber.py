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
lOUTNAME='_OUTPUTS/'+outname+'-br.txt'

gpu_no=-1
try:
    bt=commands.check_output('ssh amit@marx.uchicago.edu "ssh amit@bernie.uchicago.edu \'nvidia-smi -q --id=0 | grep Performance | cut -d: -f2 | cut -dP -f2\' " ', shell=True)
    if (np.int32(bt)>=5):
       gpu_no=0
except:
    print('gpu 0 info failed')
if (gpu_no<0):
    try:
        bt=commands.check_output('ssh amit@marx.uchicago.edu "ssh amit@bernie.uchicago.edu \'nvidia-smi -q --id=1 | grep Performance | cut -d: -f2 | cut -dP -f2\' " ',shell=True)
        if (np.int32(bt) >= 5):
            gpu_no=1
    except:
        print('gpu 1 info failed')

ss='/opt/anaconda/anaconda3/bin/python blobs.py '+' '.join(sys.argv[1:l-1]) + ' ' + str(gpu_no) + ' >'+OUTNAME
if 'loc' in outname:
    os.system(ss)
else:
    f=open('runthingsBR.txt','w')
    f.write(ss+'\n')
    f.close()
    commands.check_output("scp runthingsBR.txt amit@marx.uchicago.edu:/Volumes/amit/Python/blobs/.",shell=True)

    try:
        commands.check_output('rm ' + lOUTNAME, shell=True)
    except:
        print('Failed rm')
    try:
        commands.check_output('./GIT.sh',shell=True)
    except:
        print('GIT failed')
    try:
        commands.check_output('ssh amit@marx.uchicago.edu "ssh amit@bernie.uchicago.edu \'cd /ga/amit/Python; git pull\' " ',shell=True)
    except:
        print('pull failed')



    os.system('ssh amit@marx.uchicago.edu "ssh amit@bernie.uchicago.edu \'cd /ga/amit/Python/blobs; ./runthingsBR.txt & \' & " & ')

    ny='no'
    while (ny != ''):
        time.sleep(10)
        try:
            ss=commands.check_output('scp amit@marx.uchicago.edu:/Volumes/amit/Python/blobs/'+OUTNAME+' '+ lOUTNAME,shell=True)
            print(ss)
            ny=''
        except:
            ny='no'
            print('Initial copy fialed')
    done=False
    while (not done):
        time.sleep(10)
        try:
          ss=commands.check_output('grep DONE '+lOUTNAME,shell=True)
          print('Done grep',ss)
          if (ss!=''):
              done=True
        except:
            done=False
        try:
            commands.check_output('scp amit@marx.uchicago.edu:/Volumes/amit/Python/blobs/'+OUTNAME+' ' + lOUTNAME,shell=True)
        except:
            print('copy failed')
    time.sleep(5)
    dd = commands.check_output('grep model ' + sys.argv[-2] + '.txt  | cut -d":" -f2', shell=True)
    dirname = dd.decode("utf-8").strip('\n')
    commands.check_output('rm -rf ' + dirname, shell=True)
    commands.check_output('mkdir ' + dirname, shell=True)
    commands.check_output('scp amit@marx.uchicago.edu:/Volumes/amit/Python/blobs/'+dirname+'/* ' + dirname + '/.',
                          shell=True)
    commands.check_output('cp ' + sys.argv[-2] + '.txt '+dirname+'/.',shell=True)


