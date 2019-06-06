import torch
from torchvision import transforms, datasets
from models import STVAE
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

parser = argparse.ArgumentParser(
    description='Variational Autoencoder with Spatial Transformation'
)

parser.add_argument('--transformation', default='aff',help='type of transformation: aff or tps')
parser.add_argument('--type', default='tvae',help='type of transformation: aff or tps')
parser.add_argument('--sdim', type=int, default=16, help='dimension of s')
parser.add_argument('--zdim', type=int, default=10, help='dimension of z')
parser.add_argument('--hdim', type=int, default=256, help='dimension of h')
parser.add_argument('--num_hlayers', type=int, default=1, help='number of hlayers')
parser.add_argument('--nepoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--gpu', type=bool, default=False,help='whether to run in the GPU')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
parser.add_argument('--num_train',type=int,default=60000,help='num train (default: 10000)')
parser.add_argument('--mb_size',type=int,default=500,help='mb_size (default: 500)')
parser.add_argument('--model',default='base',help='model (default: base)')
parser.add_argument('--optimizer',default='Adam',help='Type of optimiser')

args = parser.parse_args()
print(args)
use_gpu = args.gpu and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda:1" if use_gpu else "cpu")
print(device)
print(use_gpu)
kwargs = {'num_workers': 8, 'pin_memory': True} if use_gpu else {}

PARS={}
PARS['data_set']='mnist'
PARS['num_train']=args.num_train

PARS['nval']=0
train, val, test, image_dim = get_data(PARS)
if (PARS['nval']==0):
    val=None
h=train[0].shape[1]
w=train[0].shape[2]
# lentr=(args.num_train//args.mb_size * args.mb_size)
# lente=(10000//args.mb_size * args.mb_size)
#
# train=[]
# train.append(traina[0][0:lentr])
# train.append(traina[1][0:lentr])
# test=[]
# test.append(traina[0][0:lente])
# test.append(traina[1][0:lente])
# mnist_tr = datasets.MNIST(root='../MNIST', download=True,transform=transforms.ToTensor())
# tr = torch.utils.data.DataLoader(dataset=mnist_tr,
#                                  batch_size=60000,
#                                  shuffle=False,
#                                  drop_last=True, **kwargs)
# mnist_te = datasets.MNIST(root='../MNIST/', download=True, train=False, transform=transforms.ToTensor())
# te = torch.utils.data.DataLoader(dataset=mnist_te,
#                                  batch_size=10000,
#                                  shuffle=False,
#                                  drop_last=True, **kwargs)
# trl=train[1]
# vl=val[1]
# tel=test[1]
# train=[]
# val=None
# test=[]
#
# for a in tr:
#     print(a[0].shape)
#     train.append(np.array(a[0][0:lentr]))
#     #val.append(np.array(a[0][50000:60000]))
# train.append(trl[0:lentr])
# if (val is not None):
#     val.append(vl)
# for a in te:
#     print(a[0].shape)
#     test.append(np.array(a[0][0:lente]))
# test.append(tel[0:lente])
# #tr=torch.utils.data.Subset(mnist_tr,0)
model = STVAE(h, w,  device, args).to(device)
l2 = lambda epoch: pow((1.-1. * epoch/args.nepoch),0.9)
scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda=l2)


for epoch in range(args.nepoch):
    scheduler.step()
    t1=time.time()
    model.run_epoch(train,epoch,type='train')
    if (val is not None and val):
        model.run_epoch([val[0][0:500],val[1][0:500]],epoch,type='val')
        #model.run_epoch([val[0][0:500], val[1][0:500]], epoch, type='val')
    print('epoch: {0} in {1:5.3f} seconds'.format(epoch,time.time()-t1))
    sys.stdout.flush()
model.run_epoch([val[0][0:500],val[1][0:500]],epoch,type='val')
model.run_epoch(train,epoch,type='trest')
model.run_epoch(test,epoch,type='test')


torch.save(model.state_dict(), 'output/'+args.type+'_'+args.transformation+'_'+str(args.num_hlayers)+'.pt')

if (not use_gpu):
    x = model.sample_from_z_prior(theta=torch.zeros(6))
    aa = x.cpu().numpy().squeeze()
    py.figure(figsize=(10, 10))
    for t in range(100):
        py.subplot(10,10,t+1)
        py.imshow(aa[t],cmap='gray')
        py.axis('off')
    #py.show()
    #print('hello')
    py.savefig('output/fig_'+args.type+'_'+args.transformation+'_'+str(args.num_hlayers))
print("DONE")
#bt = commands.check_output('mv OUTPUT.txt OUTPUT_'+args.type+'_'+args.transformation+'_'+str(args.num_hlayers)+'.txt',shell=True)

