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
# try:
#     bt=commands.check_output('ssh amit@bernie.uchicago.edu \'nvidia-smi -q --id=0 | grep Performance | cut -d: -f2 | cut -dP -f2\' ', shell=True)
#     if (np.int32(bt)>=5):
#        gpu_no=0
# except:
#     print('gpu 0 info failed')
# if (gpu_no<0):
#     try:
#         bt=commands.check_output("ssh amit@bernie.uchicago.edu \'nvidia-smi -q --id=1 | grep Performance | cut -d: -f2 | cut -dP -f2\' ",shell=True)
#         if (np.int32(bt) >= 5):
#             gpu_no=1
#     except:
#         print('gpu 1 info failed')
# print('gpu',gpu_no)

ss='/opt/anaconda3_beta/bin/python '+sys.argv[1]+' $(cat'+' '+sys.argv[2]+') '+' '.join(sys.argv[3:l-1]) + ' >'+OUTNAME
if 'loc' in outname:
    os.system(ss)
else:
    f=open('runthingsBR.txt','w')
    f.write(ss+'\n')
    f.close()
    commands.check_output("scp runthingsBR.txt aitken:Python/STVAE/.",shell=True)

    try:
        commands.check_output('rm ' + lOUTNAME, shell=True)
    except:
        print('Failed rm')
    try:
        commands.check_output('./GIT.sh',shell=True)
    except:
        print('GIT failed')
    try:
        commands.check_output("ssh amit@bernie.uchicago.edu \'cd /ga/amit/Python; git pull\' ",shell=True)
    except:
        print('pull failed')


    #os.system("ssh amit@bernie.uchicago.edu \'cd /ga/amit/Python/STVAE; rm _output/* \' & ")
    os.system("ssh amit@bernie.uchicago.edu \'cd /ga/amit/Python/STVAE; rm _Images/* \' & ")

    os.system("ssh amit@bernie.uchicago.edu \'cd /ga/amit/Python/STVAE; ./runthingsBR.txt \' & ")

    ny='no'
    while (ny != ''):
        time.sleep(10)
        try:
            ss=commands.check_output('scp aitken:Python/STVAE/'+OUTNAME+' '+ lOUTNAME,shell=True)
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
            print('Not Done')
            done=False
        try:
            commands.check_output('scp aitken:Python/STVAE/'+OUTNAME+' ' + lOUTNAME,shell=True)
        except:
            print('copy failed')
    time.sleep(10)
    commands.check_output('scp aitken:Python/STVAE/' + OUTNAME + ' ' + lOUTNAME, shell=True)
    #os.system('mv /Volumes/amit/Python/STVAE/_OUTPUTS/* /Users/amit/Desktop/Dropbox/Python/STVAE/_OUTPUTS/.')
    os.system('scp aitken:Python/STVAE/_output/* /Users/amit/Desktop/Dropbox/Python/STVAE/_output/.')
    os.system('scp aitken:Python/STVAE/_Images/* /Users/amit/Desktop/Dropbox/Python/STVAE/_Images/.')


