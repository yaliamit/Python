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
    commands.check_output("scp runthingsBR.txt amit@marx.uchicago.edu:/Volumes/amit/Python/STVAE/.",shell=True)

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

    os.system('ssh amit@marx.uchicago.edu "ssh amit@bernie.uchicago.edu \'cd /ga/amit/Python/STVAE; ./runthingsBR.txt \' & " & ')

    ny='no'
    while (ny != ''):
        time.sleep(10)
        try:
            ss=commands.check_output('scp amit@marx.uchicago.edu:/Volumes/amit/Python/STVAE/'+OUTNAME+' '+ lOUTNAME,shell=True)
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
            commands.check_output('scp amit@marx.uchicago.edu:/Volumes/amit/Python/STVAE/'+OUTNAME+' ' + lOUTNAME,shell=True)
        except:
            print('copy failed')
    time.sleep(5)

    os.system('ssh amit@marx.uchicago.edu "mv /Volumes/amit/Python/STVAE/_OUTPUTS/* /Users/amit/Desktop/Dropbox/Python/STVAE/_OUTPUTS/."')
    os.system('ssh amit@marx.uchicago.edu "mv /Volumes/amit/Python/STVAE/_output/* /Users/amit/Desktop/Dropbox/Python/STVAE/_output/."')
    os.system('ssh amit@marx.uchicago.edu "mv /Volumes/amit/Python/STVAE/_Images/* /Users/amit/Desktop/Dropbox/Python/STVAE/_Images/."')

    # dd = commands.check_output('grep model ' + sys.argv[-2] + '.txt  | cut -d":" -f2', shell=True)
    # dirname = dd.decode("utf-8").strip('\n')
    # commands.check_output('rm -rf ' + dirname, shell=True)
    # commands.check_output('mkdir ' + dirname, shell=True)
    # commands.check_output('scp amit@marx.uchicago.edu:/Volumes/amit/Python/STVAE/'+dirname+'/* ' + dirname + '/.',
    #                       shell=True)
    # commands.check_output('cp ' + sys.argv[-2] + '.txt '+dirname+'/.',shell=True)


