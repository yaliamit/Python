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

def initialize_mus(train,args):
    trMU=None
    trLOGVAR=None
    if (args.OPT and train[0] is not None):
        trMU = torch.zeros(train[0].shape[0], args.sdim).to(device)
        trLOGVAR = torch.zeros(train[0].shape[0], args.sdim).to(device)
    return trMU, trLOGVAR

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
parser.add_argument('--nti',type=int,default=100,help='num test iterations (default: 100)')
parser.add_argument('--nvi',type=int,default=20,help='num val iterations (default: 20)')
parser.add_argument('--MM',action='store_true', help='Use max max')
parser.add_argument('--OPT',action='store_true',help='Optimization instead of encoding')

args = parser.parse_args()
opt_pre=''; mm_pre=''; opt_post=''
if (args.OPT):
    opt_pre='OPT_';opt_post='_OPT';
if (args.MM):
    mm_pre='_MM'
ex_file=opt_pre+args.type + '_' + args.transformation + '_' + str(args.num_hlayers)+'_sd_'+str(args.sdim)+'_'+args.optimizer+mm_pre



use_gpu = args.gpu and torch.cuda.is_available()
if (use_gpu):
    fout=open('_OUTPUTS/OUT_'+ex_file+'.txt','w')
else:
    fout=sys.stdout

if (fout is not None):
    fout.write(str(args)+'\n')
    fout.flush()
else:
    print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda:1" if use_gpu else "cpu")
print(device)
print('USE_GPU',use_gpu)

# round the number of training data to be a multiple of nbatch size

PARS={}
PARS['data_set']='mnist'
PARS['num_train']=args.num_train//args.mb_size *args.mb_size
PARS['nval']=args.nval
if args.cl is not None:
    PARS['one_class']=args.cl

train, val, test, image_dim = get_data(PARS)

trainMU, trainLOGVAR=initialize_mus(train,args)
valMU, valLOGVAR=initialize_mus(val,args)
testMU, testLOGVAR=initialize_mus(test,args)

h=train[0].shape[1]
w=train[0].shape[2]
model=locals()['STVAE'+opt_post](h, w,  device, args).to(device)

tot_pars=0
for keys, vals in model.state_dict().items():
    fout.write(keys+','+str(np.array(vals.shape))+'\n')
    tot_pars+=np.prod(np.array(vals.shape))
fout.write('tot_pars,'+str(tot_pars)+'\n')

if (args.run_existing):
    model.load_state_dict(torch.load('_output/'+ex_file+'.pt',map_location=device))
    model.eval()
    if (args.OPT):
        model.run_epoch(train, 0, 500, trainMU, trainLOGVAR, type='trest', fout=fout)
        model.run_epoch(test, 0, 500, testMU, testLOGVAR, type='test', fout=fout)
        model.run_epoch(test,0,args.nti,testMU, testLOGVAR,type='test',fout=fout)
    else:
        model.run_epoch(test, 0, type='test')
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
        if (fout is not None):
            fout.write('epoch: {0} in {1:5.3f} seconds\n'.format(epoch,time.time()-t1))
            fout.flush()
        else:
            print('epoch: {0} in {1:5.3f} seconds'.format(epoch,time.time()-t1))
        sys.stdout.flush()

    if (args.MM):
            fout.write('Means and variances of latent variable before restimation\n')
            fout.write(str(model.MU.data)+'\n')
            fout.write(str(model.LOGVAR.data)+'\n')
            fout.flush()
            trainMU, trainLOGVAR = initialize_mus(train, args)
            trainMU, trainLOGVAR = model.run_epoch(train,  epoch, 500,trainMU, trainLOGVAR, type='trest',fout=fout)
            model.MU = torch.nn.Parameter(torch.mean(trainMU, dim=0))
            model.LOGVAR = torch.nn.Parameter(torch.log(torch.var(trainMU, dim=0)))
            fout.write('Means and variances of latent variable after restimation\n')
            fout.write(str(model.MU.data) + '\n')
            fout.write(str(model.LOGVAR.data) + '\n')
            fout.flush()
            model.to(device)
    show_sampled_images(model, ex_file)
    trainMU, trainLOGVAR = initialize_mus(train,args)
    model.run_epoch(train,  epoch, 500, trainMU, trainLOGVAR,type='trest',fout=fout)
    model.run_epoch(test,epoch,500,testMU, testLOGVAR,type='test',fout=fout)
    fout.flush()

    print('writing to ',ex_file)
    torch.save(model.state_dict(),'_output/'+ex_file+'.pt')


    print("DONE")

