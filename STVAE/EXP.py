import os
import numpy as np
import pickle
import socket
import time
import sys
import copy
import pprint
import pandas

#Haha

def produce_B(b):
    B = copy.deepcopy(b[0])
    ll=len(b)
    for keya, valuea in B.items():
        print(keya)
        for keyb, valueb in valuea.items():
            print(keyb)
            for keyc, valuec in valueb.items():
                print(keyc)
                for keyd, valued in valuec.items():
                    print(keyd, valued)
                    B[keya][keyb][keyc][keyd] = np.array([0., 0., 0., 0.])
    for j in range(ll):
        for keya, valuea in b[j].items():
            print(keya)
            for keyb, valueb in valuea.items():
                print(keyb)
                for keyc, valuec in valueb.items():
                    print(keyc)
                    for keyd, valued in valuec.items():
                        print(keyd, valued)
                        v = np.array(valued).reshape(1, -1)
                        B[keya][keyb][keyc][keyd][0] += v[0]
                        if (ll>=2):
                            B[keya][keyb][keyc][keyd][1] += v[0] * v[0]
                        if (v.shape[1]>1):
                            B[keya][keyb][keyc][keyd][2] += valued[1]
                            if (ll>=2):
                                B[keya][keyb][keyc][keyd][3] += valued[1] * valued[1]
    for keya, valuea in B.items():
        print(keya)
        for keyb, valueb in valuea.items():
            print(keyb)
            for keyc, valuec in valueb.items():
                print(keyc)
                for keyd, valued in valuec.items():
                    print(keyd, valued)
                    B[keya][keyb][keyc][keyd] /= ll
                    if (ll>=2):
                      for j in [1, 3]:
                        B[keya][keyb][keyc][keyd][j] = np.sqrt(B[keya][keyb][keyc][keyd][j] -
                                                               B[keya][keyb][keyc][keyd][j - 1] *
                                                               B[keya][keyb][keyc][keyd][j - 1])

    BB={}
    for keya, valuea in B.items():
        print(keya)
        for keyb, valueb in valuea.items():
            print(keyb)
            for keyc, valuec in valueb.items():
                print(keyc)
                for keyd, valued in valuec.items():
                    if keyd=='':
                        keydd='REG'
                    else:
                        keydd='OPT'
                    Bd=B[keya][keyb][keyc][keyd]
                    if ('recon' in keya):

                           BB[(keyb,keyc,keydd)]=['{:2.2f} ({:2.2f})'.format(Bd[0],Bd[1])]
                    else:
                           BB[(keyb, keyc, keydd)] += ['{:2.2f} ({:2.2f})'.format(Bd[0],Bd[1]),'{:2.2f} ({:2.2f})'.format(Bd[2],Bd[3])]
                    # if ('recon' in keya):
                    #     BB[(keyb,keyc,keydd,'recon')]=[B[keya][keyb][keyc][keyd][0]]
                    # else:
                    #     BB[(keyb, keyc, keydd,'loss')] = [B[keya][keyb][keyc][keyd][0]] #,B[keya][keyb][keyc][keyd][2]]
                    #
                    #


    return(BB)

def run_exp(nh,mx, type, numt=60000):
    seed = np.random.randint(500000)
    print("SEED",seed)
    if ('marx' in socket.gethostname()):
        scr='mrunber.py'
    else:
        scr='runber.py'
    print('Script',scr)
    TYPE=type.split('+')
    H=['10']#,32,64]
    OPT=['','--OPT']
    #OPT = ['--OPT']
    mm=[]
    for i in np.arange(0,len(mx),2):
        mm=mm+[mx[i:i+2]]
    mx=mm

    RR={}
    RR['test_recon_loss']={}
    RR['test_loss']={}
    ne=300

    for mm in mx:
        mkey='nmix_'+mm[0]+'_'+mm[1]
        m=mm[0]
        h=mm[1]
        for keys,vals in RR.items():
           RR[keys][mkey]={}
        for ty in TYPE:
          for keys, vals in RR.items():
                RR[keys][mkey][ty] = {}
          for OP in OPT:
            print(mkey,ty,OP)
            sys.stdout.flush()
            file='OUT'
            file_br='_OUTPUTS/'+file+'-br.txt'
            com = 'python _scripts/'+scr+' main_opt.py _pars/pars_tvae --type='+ty+' --n_mix='+m+' --num_train='+str(numt)+' --num_hlayers='+str(nh)+' --nepoch='+str(ne)+' --seed='+str(seed)+' --nti=500 --CONS --sdim='+h+' '+OP+' '+file
            print(com)
            os.system(com)
            com = 'grep test '+file_br+' | cut -d":" -f3 | cut -d"," -f1 > junk'
            os.system(com)
            aa=np.loadtxt('junk')
            if len(aa.shape)==0:
                RR['test_recon_loss'][mkey][ty][OP] = np.asscalar(aa)
            else:
                RR['test_recon_loss'][mkey][ty][OP] = aa
            com = 'grep test '+file_br+' | cut -d":" -f4 > junk'
            os.system(com)
            aa=np.loadtxt('junk')
            if (len(aa.shape)==0):
                RR['test_loss'][mkey][ty][OP] = np.asscalar(aa)
            else:
                RR['test_loss'][mkey][ty][OP] = aa
            print(RR)
            sys.stdout.flush()
            FOUT = 'EXP_h' + str(nh) + '.pickle'
            with open(FOUT, 'wb') as handle:
                pickle.dump(RR, handle, protocol=pickle.HIGHEST_PROTOCOL)
    time.sleep(10)
    FOUT='EXP_h'+str(nh)+'.pickle'
    with open(FOUT, 'wb') as handle:
        pickle.dump(RR, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_exp(nh,mx,type,nt):
    with open(dir_name+'/EXP_h'+str(nh)+'.pickle', 'rb') as handle:
         b = pickle.load(handle)


    return b

def run_e(nh,mx,type,numt):
    os.system('mkdir _Images')
    os.system('mkdir _output')
    mm=mx.split('+')
    os.system('mkdir '+dir_name)
    run_exp(nh,mm,type,numt=numt)
    os.system('mv EXP_h* '+dir_name)
    os.system('mv _Images '+dir_name)
    os.system('mv _output '+dir_name)

def read_e(nh,mx,type,nt):
    b0=read_exp(nh,mx,type,nt)
    return b0

l=len(sys.argv)

new=True
ntk=20
if (l>1):
    new=(sys.argv[1]=='new')
if (l>2):
    ntk=np.int32(sys.argv[2])
numt=ntk*1000
mx='1+120+3+40+6+20'
#mx='6+20'

if (l>3):
    mx=sys.argv[3]
mtype='vae+tvae'

if (l>4):
    mtype=sys.argv[4]
OPT='_OPT'
#/Users/amit/Box Sync/EXP_NT60/
dir_name_base='EXP'+'_NT'+str(ntk)+'_'+mx+'_'+mtype
print(dir_name_base)

nh=1
if new:
    for i in range(10):
        dir_name=dir_name_base+'_'+str(i)
        run_e(nh,mx,mtype,numt)
else:
    b=[]
    dd=os.listdir('./')
    nn=0
    for d in dd:
        if dir_name_base in d:
            nn+=1
    for i in range(nn):
        dir_name=dir_name_base+'_'+str(i)
        os.listdir(dir_name)
        b=b+[read_e(nh,mx,mtype,ntk)]

    B=produce_B(b)
    np.set_printoptions(precision=4)
    df = pandas.DataFrame(B)
    df.rename(index={0: 'recon',1:'loss',2:'acc'},inplace=True)
    df=df.unstack(level=-1)
    df=df.unstack(level=2)
    df=df.unstack(level=1)

    #df.reset_index(inplace=True)
    #df.set_index.set_names()
    f=open('try_'+str(ntk)+'.tex','w')

    f.write(df.to_latex())
    f.close()
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(B)

#c0, c1=read_e(61)

print('hello')
