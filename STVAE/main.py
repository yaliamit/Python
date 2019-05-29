import torch
from torch import nn, optim
from torchvision import transforms, datasets
from models import STVAE
import numpy as np
import subprocess as commands
import pylab as py
from torchvision.utils import save_image

import argparse

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
parser.add_argument('--gpu', default=False, action='store_true',help='whether to run in the GPU')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
parser.add_argument('--num_train',type=int,default=10000,help='num train (default: 10000)')
parser.add_argument('--mb_size',type=int,default=500,help='mb_size (default: 500)')
parser.add_argument('--model',type=int,default='base',help='model (default: base)')

args = parser.parse_args()
print(args)
use_gpu = args.gpu and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_gpu else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
# add 'download=True' when use it for the first time
mnist_tr = datasets.MNIST(root='../MNIST/', transform=transforms.ToTensor(),download=True)
mnist_tr.data=mnist_tr.train_data[0:args.num_train]
h=mnist_tr.train_data.shape[1]
w=mnist_tr.train_data.shape[2]
mnist_te = datasets.MNIST(root='../MNIST/', train=False, transform=transforms.ToTensor(),download=True)
tr = torch.utils.data.DataLoader(dataset=mnist_tr,
                                batch_size=args.mb_size,
                                shuffle=True,
                                drop_last=True, **kwargs)
te = torch.utils.data.DataLoader(dataset=mnist_te,
                                batch_size=args.mb_size,
                                shuffle=True,
                                drop_last=True, **kwargs)


model = STVAE(h, w,  device, args).to(device)
#l2 = lambda epoch: pow((1.-1. * epoch/args.nepoch),0.9)
#scheduler = optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda=l2)


for epoch in range(args.nepoch):
    #scheduler.step()
    model.train_epoch(tr,epoch)
    model.test(te,epoch)


torch.save(model.state_dict(), 'output/'+args.type+'_'+args.transformation+'_'+str(args.num_hlayers)+'.pt')
#model1 = STVAE(h, w, device, args).to(device)
#model1.load_state_dict(torch.load('output/'+args.type+'_'+args.transformation+'.pt'))
#model1.eval()


x=model.sample_from_z_prior(theta=torch.zeros(6))

aa=np.array(x.data).squeeze()
py.figure(figsize=(10, 10))
for t in range(100):
    py.subplot(10,10,t+1)
    py.imshow(aa[t],cmap='gray')
    py.axis('off')

py.show()
print('hello')
py.savefig('output/fig_'+args.type+'_'+args.transformation+'_'+str(args.num_hlayers))

bt = commands.check_output('mv OUTPUT.txt OUTPUT_'+args.type+'_'+args.transformation+'_'+str(args.num_hlayers)+'.txt',shell=True)

