import torch
from models_opt import STVAE_OPT
from models_mix import STVAE_mix
from models import STVAE
import numpy as np
import os
import sys
import argparse
import time
from Conv_data import get_data
from models import  get_scheduler
from aux import process_args, re_estimate, rerun_on_train_test, add_clutter, add_occlusion, test_with_noise, show_sampled_images

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
    description='Variational Autoencoder with Spatial Transformation'
)

args=process_args(parser)


opt_pre=''; mm_pre=''; opt_post=''
if (args.OPT):
    opt_pre='OPT_';opt_post='_OPT';
if (args.n_mix>1):
    opt_post='_mix'
if (args.MM):
    mm_pre='_MM'
ex_file=opt_pre+args.type + '_' + args.transformation + '_' + str(args.num_hlayers)+'_sd_'+str(args.sdim)+'_'+args.optimizer+mm_pre

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
model=locals()['STVAE'+opt_post](h, w,  device, args).to(device)
tot_pars=0
for keys, vals in model.state_dict().items():
    fout.write(keys+','+str(np.array(vals.shape))+'\n')
    tot_pars+=np.prod(np.array(vals.shape))
fout.write('tot_pars,'+str(tot_pars)+'\n')


trainMU, trainLOGVAR=model.initialize_mus(train,args.OPT)
valMU, valLOGVAR=model.initialize_mus(val,args.OPT)
testMU, testLOGVAR=model.initialize_mus(test,args.OPT)

if (args.run_existing):
    model.load_state_dict(torch.load('_output/'+ex_file+'.pt',map_location=device))
    #model.eval()
    test_with_noise(test, model)
    #rerun_on_train_test(model,train,test,args)
    show_sampled_images(model,ex_file)
else:
    scheduler=get_scheduler(args,model)

    for epoch in range(args.nepoch):
        if (scheduler is not None):
            scheduler.step()
        t1=time.time()
        trainMU, trainLOGVAR= model.run_epoch(train,epoch,args.num_mu_iter,trainMU,trainLOGVAR,type='train',fout=fout)
        if (val[0] is not None):
                model.run_epoch(val,epoch,args.nvi,valMU,valLOGVAR,type='val',fout=fout)

        fout.write('epoch: {0} in {1:5.3f} seconds\n'.format(epoch,time.time()-t1))
        fout.flush()

    if (args.MM):
        re_estimate()

    show_sampled_images(model, ex_file)
    #trainMU, trainLOGVAR = model.initialize_mus(train,args.OPT)
    #model.run_epoch(train,  epoch, args.nti, trainMU, trainLOGVAR,type='trest',fout=fout)
    model.run_epoch(test,epoch,args.nti,testMU, testLOGVAR,type='test',fout=fout)

    fout.write('writing to '+ex_file+'\n')
    torch.save(model.state_dict(),'_output/'+ex_file+'.pt')
    fout.flush()
    if (not args.CONS):
        fout.close()

    print("DONE")
    sys.stdout.flush()

