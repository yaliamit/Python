import torch
from models_opt import STVAE_OPT
from models import STVAE
import json
import numpy as np
import os
import sys
import argparse
import time
from Conv_data import get_data
from models import show_sampled_images, get_scheduler
import pylab as py

def rerun_on_train_test(model,train,test,args):
    trainMU, trainLOGVAR = model.initialize_mus(train, args.OPT)
    testMU, testLOGVAR = model.initialize_mus(test, args.OPT)
    if (args.OPT):
        model.setup_id(model.bsz)
        model.run_epoch(train, 0, args.nti, trainMU, trainLOGVAR, type='trest', fout=fout)
        model.run_epoch(test, 0, args.nti, testMU, testLOGVAR, type='test', fout=fout)
    else:
        model.run_epoch(test, 0, type='test')

def test_with_noise(test,model):

    ii=np.arange(0,test[0].shape[0],1)
    np.random.shuffle(ii)
    recon_data=[test[0][ii[0:20]].copy(),test[1][ii[0:20]].copy()]
    recon_data[0][0:20,0:13,:,:]=0
    recon_ims=model.recon(recon_data, num_mu_iter=args.nti)
    rec=recon_ims.detach().cpu()
    py.figure(figsize=(3, 20))
    for t in range(20):
            py.subplot(20,3,3*t+1)
            py.imshow(test[0][ii[t],:,:,0])
            py.axis('off')
            py.subplot(20,3,3*t+2)
            py.imshow(recon_data[0][t, :, :, 0])
            py.axis('off')
            py.subplot(20,3,3*t+3)
            py.imshow(rec[t,0,:,:])
            py.axis('off')
    py.show()
    print("hello")


def re_estimate(model):

            fout.write('Means and variances of latent variable before restimation\n')
            fout.write(str(model.MU.data)+'\n')
            fout.write(str(model.LOGVAR.data)+'\n')
            fout.flush()
            trainMU, trainLOGVAR = model.initialize_mus(train, args.OPT,args.MM)
            trainMU, trainLOGVAR = model.run_epoch(train,  epoch, 500,trainMU, trainLOGVAR, type='trest',fout=fout)
            model.MU = torch.nn.Parameter(torch.mean(trainMU, dim=0))
            model.LOGVAR = torch.nn.Parameter(torch.log(torch.var(trainMU, dim=0)))
            fout.write('Means and variances of latent variable after restimation\n')
            fout.write(str(model.MU.data) + '\n')
            fout.write(str(model.LOGVAR.data) + '\n')
            fout.flush()
            model.to(device)

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
    description='Variational Autoencoder with Spatial Transformation'
)

parser.add_argument('--transformation', default='aff',help='type of transformation: aff or tps')
parser.add_argument('--type', default='vae',help='type of transformation: aff or tps')
parser.add_argument('--sdim', type=int, default=26, help='dimension of s')
parser.add_argument('--hdim', type=int, default=256, help='dimension of h')
parser.add_argument('--num_hlayers', type=int, default=0, help='number of hlayers')
parser.add_argument('--nepoch', type=int, default=40, help='number of training epochs')
parser.add_argument('--gpu', type=bool, default=False,help='whether to run in the GPU')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
parser.add_argument('--num_train',type=int,default=60000,help='num train (default: 60000)')
parser.add_argument('--nval',type=int,default=1000,help='num train (default: 1000)')
parser.add_argument('--mb_size',type=int,default=100,help='mb_size (default: 500)')
parser.add_argument('--model',default='base',help='model (default: base)')
parser.add_argument('--optimizer',default='Adam',help='Type of optimiser')
parser.add_argument('--lr',type=float, default=.001,help='Learning rate (default: .001)')
parser.add_argument('--mu_lr',type=float, default=.05,help='Learning rate (default: .05)')
parser.add_argument('--num_mu_iter',type=int, default=10,help='Learning rate (default: .05)')
parser.add_argument('--wd',action='store_true', help='Use weight decay')
parser.add_argument('--cl',type=int,default=None,help='class (default: None)')
parser.add_argument('--run_existing',action='store_true', help='Use existing model')
parser.add_argument('--nti',type=int,default=500,help='num test iterations (default: 100)')
parser.add_argument('--nvi',type=int,default=20,help='num val iterations (default: 20)')
parser.add_argument('--MM',action='store_true', help='Use max max')
parser.add_argument('--OPT',action='store_true',help='Optimization instead of encoding')
parser.add_argument('--CONS',action='store_true',help='Output to consol')


args = parser.parse_args()
opt_pre=''; mm_pre=''; opt_post=''
if (args.OPT):
    opt_pre='OPT_';opt_post='_OPT';
if (args.MM):
    mm_pre='_MM'
ex_file=opt_pre+args.type + '_' + args.transformation + '_' + str(args.num_hlayers)+'_sd_'+str(args.sdim)+'_'+args.optimizer+mm_pre

use_gpu = args.gpu and torch.cuda.is_available()
if (use_gpu and not args.CONS):
    fout=open('_OUTPUTS/OUT_'+ex_file+'.txt','w')
else:
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

        sys.stdout.flush()

    if (args.MM):
        re_estimate()

    show_sampled_images(model, ex_file)
    trainMU, trainLOGVAR = model.initialize_mus(train,args.OPT)
    model.run_epoch(train,  epoch, 500, trainMU, trainLOGVAR,type='trest',fout=fout)
    model.run_epoch(test,epoch,500,testMU, testLOGVAR,type='test',fout=fout)

    fout.write('writing to '+ex_file+'\n')
    torch.save(model.state_dict(),'_output/'+ex_file+'.pt')
    fout.flush()
    if (not args.CONS):
        fout.close()

    print("DONE")

