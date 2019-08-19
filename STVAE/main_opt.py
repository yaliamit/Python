import torch
from models_opt import STVAE_OPT
from models_mix import STVAE_mix
from models import STVAE
from models_opt_mix import STVAE_OPT_mix
import numpy as np
import os
import sys
import argparse
import time
from Conv_data import get_data
from models import  get_scheduler
import aux
from class_on_hidden import train_new

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
    description='Variational Autoencoder with Spatial Transformation'
)

args=aux.process_args(parser)


opt_pre=''; mm_pre=''; opt_post=''; opt_mix=''
if (args.OPT):
    opt_pre='OPT_';opt_post='_OPT';
if (args.n_mix>=1):
    opt_mix='_mix'
if (args.MM):
    mm_pre='_MM'
ex_file=opt_pre+args.type + '_' + args.transformation + '_' + str(args.num_hlayers)+'_mx_'+str(args.n_mix)+'_sd_'+str(args.sdim)+'_'+args.optimizer+mm_pre

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

device = torch.device("cuda:1" if use_gpu else "cpu")
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


h=train[0].shape[1]
w=train[0].shape[2]
model=locals()['STVAE'+opt_post+opt_mix](h, w,  device, args).to(device)
tot_pars=0
for keys, vals in model.state_dict().items():
    fout.write(keys+','+str(np.array(vals.shape))+'\n')
    tot_pars+=np.prod(np.array(vals.shape))
fout.write('tot_pars,'+str(tot_pars)+'\n')





if (args.run_existing):
    model.load_state_dict(torch.load(args.output_prefix+'_output/'+ex_file+'.pt',map_location=device))
    if (args.classify):
        train_new(model,args,train,test,device)
    testMU, testLOGVAR, testPI = model.initialize_mus(test[0], args.OPT)
    model.run_epoch(test,0,args.nti,testMU, testLOGVAR,testPI, type='test',fout=fout)
    aux.show_reconstructed_images(test,model,ex_file,args.nti)
    if args.n_mix>0:
            for clust in range(args.n_mix):
                aux.show_sampled_images(model,ex_file,clust)

else:
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


    aux.make_images(test,model,ex_file,args)
    model.run_epoch(test,epoch,args.nti,testMU, testLOGVAR,testPI, type='test',fout=fout)

    fout.write('writing to '+ex_file+'\n')
    torch.save(model.state_dict(),'_output/'+ex_file+'.pt')
    trainMU=None;trainLOGVAR=None;trainPI=None
    if args.classify:
        args.nepoch=10
        args.lr=.01
        train_new(model,args,train,test,device)

    fout.write('DONE\n')
    fout.flush()
    if (not args.CONS):
        fout.close()


    #print("DONE")
    #sys.stdout.flush()

