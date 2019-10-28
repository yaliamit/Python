import torch
from models_opt import STVAE_OPT
from models_mix import STVAE_mix
from models import STVAE
from models_opt_mix import STVAE_OPT_mix
from models_mix_by_class import STVAE_mix_by_class
from models_opt_mix_by_class import STVAE_OPT_mix_by_class
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




os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
    description='Variational Autoencoder with Spatial Transformation'
)

args=aux.process_args(parser)


opt_pre=''; mm_pre=''; opt_post=''; opt_mix=''; opt_class=''
if (args.OPT):
    opt_pre='OPT_';opt_post='_OPT';
if (args.n_mix>=1):
    opt_mix='_mix'
if (args.MM):
    mm_pre='_MM'
if (args.n_class>0):
    opt_class='by_class_'
cll=''
if (args.cl is not None):
    cll=str(args.cl)

ex_file=opt_pre+opt_class+args.type + '_' + args.transformation + \
        '_' + str(args.num_hlayers)+'_mx_'+str(args.n_mix)+'_sd_'+str(args.sdim)+'_cl_'+cll

use_gpu = args.gpu and torch.cuda.is_available()
if (use_gpu and not args.CONS):
    fout=open('_OUTPUTS/OUT_'+ex_file+'.txt','w')
else:
    args.CONS=True
    fout=sys.stdout

fout.write(str(args)+'\n')
args.fout=fout
fout.flush()


torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda:"+str(args.gpu-1) if use_gpu else "cpu")
fout.write('Device,'+str(device)+'\n')
fout.write('USE_GPU,'+str(use_gpu)+'\n')

# round the number of training data to be a multiple of nbatch size

PARS={}
PARS['data_set']='mnist'
PARS['num_train']=args.num_train//args.mb_size *args.mb_size
PARS['nval']=args.nval
if args.cl is not None:
    PARS['one_class']=args.cl

train, val, test, image_dim = get_data(PARS)
print('num_train',train[0].shape[0])
if (args.classify):
    classify(train,test,image_dim,opt_pre,opt_post,opt_mix,opt_class,device,args,fout,locals())
    exit()

h=train[0].shape[1]
w=train[0].shape[2]
model=locals()['STVAE'+opt_post+opt_mix+opt_class](h, w,  device, args).to(device)
tot_pars=0


for keys, vals in model.state_dict().items():
    fout.write(keys+','+str(np.array(vals.shape))+'\n')
    tot_pars+=np.prod(np.array(vals.shape))
fout.write('tot_pars,'+str(tot_pars)+'\n')


if (args.run_existing):
    model.load_state_dict(torch.load(args.output_prefix+'_output/'+ex_file+'.pt',map_location=device))
    testMU, testLOGVAR, testPI = model.initialize_mus(test[0], args.OPT)
    #if (not args.sample):
    #    model.run_epoch(test,0,args.nti,testMU, testLOGVAR,testPI, type='test',fout=fout)
    #if (args.classify):
    #   train_new(model,args,train,test,device)
    #else:
    aux.make_images(test,model,ex_file,args)
    #model.run_epoch_classify(test, 0,fout=fout, num_mu_iter=args.nti)
else:

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
        trainMU, trainLOGVAR, trPI= model.run_epoch(train,epoch,args.num_mu_iter,trainMU,trainLOGVAR,trPI, type='train',fout=fout)
        if (val[0] is not None):
                model.run_epoch(val,epoch,args.nvi,valMU,valLOGVAR,valPI, type='val',fout=fout)

        fout.write('epoch: {0} in {1:5.3f} seconds\n'.format(epoch,time.time()-t1))
        fout.flush()


    aux.make_images(train,model,ex_file,args)
    #model.run_epoch(test,0,args.nti,testMU, testLOGVAR,testPI, type='test',fout=fout)
    if (args.n_class):
        model.run_epoch_classify(train, epoch,fout=fout,num_mu_iter=args.nti)
        model.run_epoch_classify(test, epoch,fout=fout, num_mu_iter=args.nti)

    fout.write('writing to '+ex_file+'\n')
    torch.save(model.state_dict(),'_output/'+ex_file+'.pt')
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

