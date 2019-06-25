import torch
from torchvision import transforms, datasets
from models import STVAE
from models_opt import STVAE_OPT
import numpy as np
import subprocess as commands
import pylab as py
from torchvision.utils import save_image
import os
import sys
import argparse
import time
from Conv_data import get_data

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
    description='Variational Autoencoder with Spatial Transformation'
)

parser.add_argument('--transformation', default='aff',help='type of transformation: aff or tps')
parser.add_argument('--type', default='vae',help='type of transformation: aff or tps')
parser.add_argument('--sdim', type=int, default=10, help='dimension of s')
parser.add_argument('--zdim', type=int, default=10, help='dimension of z')
parser.add_argument('--hdim', type=int, default=256, help='dimension of h')
parser.add_argument('--num_hlayers', type=int, default=1, help='number of hlayers')
parser.add_argument('--nepoch', type=int, default=40, help='number of training epochs')
parser.add_argument('--gpu', type=bool, default=False,help='whether to run in the GPU')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
parser.add_argument('--num_train',type=int,default=60000,help='num train (default: 10000)')
parser.add_argument('--mb_size',type=int,default=100,help='mb_size (default: 500)')
parser.add_argument('--model',default='base',help='model (default: base)')
parser.add_argument('--optimizer',default='Adam',help='Type of optimiser')
parser.add_argument('--lr',type=float, default=.001,help='Learning rate (default: .001)')
parser.add_argument('--wd',type=bool, default=True, help='Use weight decay')
parser.add_argument('--cl',type=int,default=None,help='class (default: None)')
parser.add_argument('--run_existing',type=bool, default=False, help='Use existing model')

args = parser.parse_args()
print(args)
use_gpu = args.gpu and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda:1" if use_gpu else "cpu")
print(device)
print(use_gpu)

PARS={}
PARS['data_set']='mnist'
PARS['num_train']=args.num_train

PARS['nval']=1000
if args.cl is not None:
    PARS['one_class']=args.cl

train, val, test, image_dim = get_data(PARS)
print('Num Train',train[0].shape[0])
if (PARS['nval']==0):
    val=None
h=train[0].shape[1]
w=train[0].shape[2]
model = STVAE(h, w,  device, args).to(device)

tot_pars=0
for keys, vals in model.state_dict().items():
    print(keys,np.array(vals.shape))
    tot_pars+=np.prod(np.array(vals.shape))
print('tot_pars',tot_pars)
scheduler=None
#if args.wd:
#    l2 = lambda epoch: pow((1.-1. * epoch/args.nepoch),0.9)
#    scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda=l2)
ex_file = 'output/' + args.type + '_' + args.transformation + '_' +args.optimizer+'_'+ str(args.num_hlayers) + '.pt'
print(ex_file)
if (args.run_existing):
    model.load_state_dict(
        torch.load(ex_file,map_location='cpu'))
    model.eval()

    model.run_epoch(test,0,type='test')
    theta = torch.zeros(model.bsz, 6)
    #theta=None
    X=model.sample_from_z_prior(theta)
    XX=X.cpu().detach().numpy()
    py.figure(figsize=(20,20))
    for i in range(100):
        py.subplot(10,10,i+1)
        py.imshow(1.-XX[i].reshape((28,28)),cmap='gray')
        py.axis('off')
    py.savefig(args.type+'_'+str(args.num_hlayers)+'.png')
    print("hello")
else:
    print('scheduler:',scheduler)
    for epoch in range(args.nepoch):
        if (scheduler is not None):
            scheduler.step()
        t1=time.time()
        model.run_epoch(train,epoch,type='train')
        if (val is not None and val):
            model.run_epoch(val,epoch,type='val')
        print('epoch: {0} in {1:5.3f} seconds'.format(epoch,time.time()-t1))
        sys.stdout.flush()

    model.run_epoch(train,epoch,type='trest')
    model.run_epoch(test,epoch,type='test')
    torch.save(model.state_dict(), ex_file)


print("DONE")
#bt = commands.check_output('mv OUTPUT.txt OUTPUT_'+args.type+'_'+args.transformation+'_'+str(args.num_hlayers)+'.txt',shell=True)

