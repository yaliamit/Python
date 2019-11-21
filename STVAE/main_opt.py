import torch
from models_opt import STVAE_OPT
from models_mix import STVAE_mix
from models import STVAE
#from models_opt_mix import STVAE_OPT_mix
from models_mix_by_class import STVAE_mix_by_class
#from models_opt_mix_by_class import STVAE_OPT_mix_by_class
import numpy as np
import os
import sys
import argparse
import time
from Conv_data import get_data
from models import  get_scheduler
import aux
from class_on_hidden import train_new
from classify import classify


def process_strings(args):
    strings={'opt_pre':'', 'mm_pre':'', 'opt_post':'', 'opt_mix':'', 'opt_class':'', 'cll':''}
    if (args.OPT):
        strings['opt_pre']='OPT_'
        strings['opt_post']='_OPT'
    if (args.n_mix>=1):
        strings['opt_mix']='_mix'
    if (args.MM):
        strings['mm_pre']='_MM'
    if (args.n_class>0):
        strings['opt_class']='_by_class'
    if (args.cl is not None):
        strings['cll']=str(args.cl)
    ex_file = strings['opt_pre'] + strings['opt_class'] + args.type + '_' + args.transformation + \
              '_' + str(args.num_hlayers) + '_mx_' + str(args.n_mix) + '_sd_' + str(args.sdim) + '_cl_' + strings['cll']
    return strings, ex_file

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
    description='Variational Autoencoder with Spatial Transformation'
)

args=aux.process_args(parser)

# This autromatically constructs model file name


run_existing=args.run_existing
conf=args.conf

ARGS=[]
STRINGS=[]
EX_FILES=[]
SMS=[]
if (args.run_existing):
    # This overides model file name
    names=args.model
    for i,name in enumerate(names):
        sm=torch.load('_output/'+name+'.pt')
        SMS+=[sm]
        if ('args' in sm):
            args=sm['args']
        ARGS+=[args]
        strings, ex_file = process_strings(args)
        STRINGS+=[strings]
        EX_FILES+=[ex_file]
else:
    ARGS.append(args)
    strings, ex_file = process_strings(args)
    STRINGS+=[strings]
    EX_FILES+=[ex_file

               ]
use_gpu = ARGS[0].gpu and torch.cuda.is_available()
if (use_gpu and not ARGS[0].CONS):
    fout=open('_OUTPUTS/OUT_'+ex_file+'.txt','w')
else:
    ARGS[0].CONS=True
    fout=sys.stdout

fout.write(str(args)+'\n')
fout.flush()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda:"+str(ARGS[0].gpu-1) if use_gpu else "cpu")
fout.write('Device,'+str(device)+'\n')
fout.write('USE_GPU,'+str(use_gpu)+'\n')

# round the number of training data to be a multiple of nbatch size

args=ARGS[0]
PARS={}
PARS['data_set']='mnist'
PARS['num_train']=args.num_train//args.mb_size *args.mb_size
PARS['nval']=args.nval
if args.cl is not None:
    PARS['one_class']=args.cl

train, val, test, image_dim = get_data(PARS)
print('num_train',train[0].shape[0])
# if (args.classify):
#     t1 = time.time()
#     classify(train,test,image_dim,opt_pre,opt_post,opt_mix,opt_class,device,args,fout,locals())
#     fout.write('Classified in {1:5.3f} seconds\n'.format(time.time()-t1))
#     exit()

h=train[0].shape[1]
w=train[0].shape[2]

models=[]
for strings,args in zip(STRINGS,ARGS):
    model = locals()['STVAE' + strings['opt_mix'] + strings['opt_class']](h, w, device, args).to(device)
    tot_pars = 0
    for keys, vals in model.state_dict().items():
        fout.write(keys + ',' + str(np.array(vals.shape)) + '\n')
        tot_pars += np.prod(np.array(vals.shape))
    fout.write('tot_pars,' + str(tot_pars) + '\n')
    models+=[model]

#model = locals()['STVAE' + opt_mix + opt_class](h, w, device, args).to(device)






if (run_existing):
    iid=None
    len_test=len(test[0])
    ACC=[]
    CL_RATE=[]
    print(type(SMS))
    print(len(SMS))
    ls=len(SMS)
    CF=[conf]+list(np.zeros(ls-1))
    print(CF)
    for sm, model,args,cf in zip(SMS,models,ARGS,CF):
        model.load_state_dict(sm['model.state.dict'])
        testMU, testLOGVAR, testPI = model.initialize_mus(test[0], args.OPT)
    #if (not args.sample):
    #    model.run_epoch(test,0,args.nti,testMU, testLOGVAR,testPI, type='test',fout=fout)
    #if (args.classify):
    #   train_new(model,args,train,test,device)
    #else:
    #aux.make_images(test,model,ex_file,args)
        if (iid is not None):
            test=[test[0][iid],test[1][iid]]
        print(cf)
        iid,RY,cl_rate,acc=model.run_epoch_classify(test, 'test',fout=fout, num_mu_iter=args.nti, conf_thresh=cf)
        CL_RATE+=[cl_rate]
        len_conf=len(test[0])-len(iid)
        print("current number",len_conf)
        if (len_conf>0):
            print(float(cl_rate)/len_conf)
        #model.run_epoch_classify(test, 'test',fout=fout, num_mu_iter=args.nti)
        print("Hello")
    print(np.float(np.sum(np.array(CL_RATE)))/len_test)
else:

    model=models[0]
    args=ARGS[0]
    #iic = np.argsort(np.argmax(train[1], axis=1))
    #train = [train[0][iic], train[1][iic]]

    trainMU, trainLOGVAR, trPI = model.initialize_mus(train[0], args.OPT)
    valMU, valLOGVAR, valPI = model.initialize_mus(val[0], args.OPT)
    testMU, testLOGVAR, testPI = model.initialize_mus(test[0], args.OPT)

    scheduler=get_scheduler(args,model)



    for epoch in range(args.nepoch):
        if (scheduler is not None):
            scheduler.step()
        t1=time.time()
        trainMU, trainLOGVAR, trPI= model.run_epoch(train,epoch,args.num_mu_iter,trainMU,trainLOGVAR,trPI, d_type='train',fout=fout)
        if (val[0] is not None):
                model.run_epoch(val,epoch,args.nvi,valMU,valLOGVAR,valPI, d_type='val',fout=fout)

        fout.write('epoch: {0} in {1:5.3f} seconds\n'.format(epoch,time.time()-t1))
        fout.flush()

    fout.write('writing to ' + ex_file + '\n')

    torch.save({'args':args,
                'model.state.dict':model.state_dict()}, '_output/' + ex_file + '.pt')
    aux.make_images(train,model,ex_file,args)
    if (args.n_class):
        model.run_epoch_classify(train, 'train',fout=fout,num_mu_iter=args.nti)
        model.run_epoch_classify(test, 'test',fout=fout, num_mu_iter=args.nti)
    elif args.cl is None:
        model.run_epoch(test, 0, args.nti, testMU, testLOGVAR, testPI, d_type='test', fout=fout)


    trainMU=None;trainLOGVAR=None;trainPI=None
    if args.classify:
        args.nepoch=1000
        args.lr=.01
        train_new(model,args,train,test,device)

fout.write('DONE\n')
fout.flush()
if (not args.CONS):
        fout.close()


    #print("DONE")
    #sys.stdout.flush()

