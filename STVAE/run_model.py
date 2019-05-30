import torch
from models import STVAE
from torchvision import transforms, datasets
import numpy as np
import pylab as py
import argparse
import os
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
parser.add_argument('--nepoch', type=int, default=10, help='number of training epochs')
parser.add_argument('--gpu', default=False, action='store_true',help='whether to run in the GPU')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
parser.add_argument('--num_train',type=int,default=10000,help='num train (default: 1000)')
parser.add_argument('--mb_size',type=int,default=100,help='num train (default: 1000)')

args = parser.parse_args()

use_gpu = args.gpu and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_gpu else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
mnist_tr = datasets.MNIST(root='../MNIST/', transform=transforms.ToTensor())
mnist_tr.train_data=mnist_tr.train_data[0:args.num_train]
h=mnist_tr.train_data.shape[1]
w=mnist_tr.train_data.shape[2]
mnist_te = datasets.MNIST(root='../MNIST/', train=False, transform=transforms.ToTensor())
tr = torch.utils.data.DataLoader(dataset=mnist_tr,
                                batch_size=args.mb_size,
                                shuffle=True,
                                drop_last=True, **kwargs)
te = torch.utils.data.DataLoader(dataset=mnist_te,
                                batch_size=args.mb_size,
                                shuffle=True,
                                drop_last=True, **kwargs)



model1 = STVAE(h, w, device, args).to(device)
model1.load_state_dict(torch.load('bak/output/'+args.type+'_'+args.transformation+'_'+str(args.num_hlayers)+'.pt'))
model1.eval()

model1.test(te,0)
theta=torch.zeros(6)
x=model1.sample_from_z_prior(theta=theta)

py.figure(figsize=(10, 10))
aa=np.array(x.data).squeeze()

for t in range(100):
    py.subplot(10,10,t+1)
    py.imshow(aa[t],cmap='gray')
    py.axis('off')

py.savefig('output/fig_'+args.type+'_'+args.transformation+'_'+str(args.num_hlayers))
