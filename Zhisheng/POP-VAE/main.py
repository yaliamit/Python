import torch
from torch import nn, optim
from torchvision import transforms, datasets
from model import POPVAE
import torch.nn.functional as F
#from torchvision.utils import save_image
from math import pow

import argparse

parser = argparse.ArgumentParser(
    description='Variational Autoencoder with Spatial Transformation'
)

parser.add_argument('--transformation', default='aff',
                    help='type of transformation: aff or tps')
parser.add_argument('--zdim', type=int, help='dimension of z')
parser.add_argument('--udim', type=int, help='dimension of u')
parser.add_argument('--nepoch', type=int, default=2000, help='number of training epochs')
#parser.add_argument('--lamda',type=float,default=1,help='balancing parameter in front of classification loss')
parser.add_argument('--save', default='output/model.pt')
parser.add_argument('--gpu', default=True, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')

args = parser.parse_args()

use_gpu = args.gpu and torch.cuda.is_available()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if use_gpu else "cpu")

'''Parameters, adjust here'''
mb_size = 256 # batch size

h = 28
w = 28
x_dim = h * w # image size

z_dim = 1
u_dim = 3
nc=1
num_parts=16
part_h=10
part_w=10
num_parts_h = 4
num_parts_w=4
s=20
typ = 'gaussian'
#parK = {'s':20,'typ':'loc'}

epochs = args.nepoch
h_dim = 128 # hidden dimension
log_interval = 100 # for reporting

'''########################################################################'''
def getLabel(dataset,label):
    n = len(dataset)
    idx=[]
    for i in range(n):
        if dataset[i][1] == label:
            idx.append(i)
    return idx


kwargs = {'num_workers': 4, 'pin_memory': True} if use_gpu else {}
# add 'download=True' when use it for the first time
mnist_tr = datasets.MNIST(root='../../MNIST/', download=True, transform=transforms.ToTensor())
mnist_te = datasets.MNIST(root='../../MNIST/', download=True, train=False, transform=transforms.ToTensor())


def kernelMatrix(num_parts,mb_size,nc,h,w,s,typ='loc'):
        #M = torch.zeros(bs*self.num_parts,nc,h,w).to(self.dv)
        center_h = (h-1)/2
        center_w = (w-1)/2
        v= torch.arange(0,h)
        
        J = v.repeat(h,1).float().to(device)
        v = torch.arange(0,w,1).unsqueeze(0)
        I = v.t().repeat(1,w).float().to(device)
        if typ == 'loc':
            Ind1 = (I-center_h).abs()<=s/2
            Ind2 = (J-center_w).abs()<=s/2
            Ind = Ind1*Ind2
            M = torch.zeros(h,w).to(device)
            M[Ind==1]=1
            M = M.expand(mb_size*num_parts,nc,h,w)
            
        elif typ == 'gaussian':
            
            M = torch.exp(-((I-center_h)**2+(J-center_w)**2)/s).expand(mb_size*num_parts,nc,h,w).to(device)


        else:
            raise ValueError( """An invalid option for Kernel type was supplied, options are ['loc' or 'gaussian']""")
        
        M[M<1e-07]=0
        return M
    
def loss_V(recon_x, x, mu2, var2,glb):
    '''loss = reconstruction loss + KL_z + KL_u'''
    BCE = F.binary_cross_entropy(recon_x.squeeze().view(-1, x_dim), x.view(-1, x_dim), size_average=False)
    #KLD1 = -0.5 * torch.sum(1 + 2*torch.log(var1) - mu1**2 - var1**2) # z
    KLD2 = -0.5 * torch.sum(1 + 2*torch.log(var2) - (mu2**2 + var2**2)) # u'
    KLDb = -0.5 * torch.sum(1 + 2*torch.log(glb['var']) - glb['mu']**2 - glb['var']**2) # global u'
    mu_norm = torch.mean(torch.norm(mu2,2,dim=1))
    return BCE, KLD2+KLDb,mu_norm




def train(epoch):
    model.train()
    tr_recon_loss = 0
    for batch_idx, (data, target) in enumerate(tr):
        data = data.to(device)
        optimizer.zero_grad()
    
        recon_batch, umu, uvar, u,glb,parts,KM,theta = model(data)
        recon_loss, kl , mu_norm= loss_V(recon_batch, data, umu, uvar,glb)
        
        loss = recon_loss + kl + 100*mu_norm
        loss.backward()
        tr_recon_loss += recon_loss.item()
        
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tReconstruction-Loss: {:.4f}, KL: {:.4f}'.format(
                epoch, batch_idx * len(data), len(mnist_tr),
                100. * batch_idx / len(tr),
                recon_loss / len(data),kl/len(data),
               ))
        #torch.cuda.empty_cache()

    print('====> Epoch: {} Reconstruction loss: {:.4f}'.format(
          epoch, tr_recon_loss / (len(tr)*mb_size)))

clnum=9
a = getLabel(mnist_tr,clnum)
b = getLabel(mnist_te,clnum)
#a = a[0:5760]
sampler=torch.utils.data.SubsetRandomSampler(a)
sampler2=torch.utils.data.SubsetRandomSampler(b)
#tr = torch.utils.data.DataLoader(mnist_tr,
#                                 batch_size=mb_size, 
#                                 sampler=sampler,
#                                 shuffle=False, **kwargs)
#te = torch.utils.data.DataLoader(mnist_te,
#                                 batch_size=mb_size, 
#                                 sampler=sampler2,
#                                 shuffle=False, **kwargs)

tr = torch.utils.data.DataLoader(dataset=mnist_tr,
                                batch_size=mb_size,
                                shuffle=False,
                                sampler=sampler,
                                drop_last=True, **kwargs)
te = torch.utils.data.DataLoader(dataset=mnist_te,
                                batch_size=mb_size,
                                shuffle=False,
                                sampler = sampler2,
                                drop_last=True, **kwargs)

M = kernelMatrix(num_parts,mb_size,nc,h,w,s,typ)


model = POPVAE(h, w, h_dim, z_dim, u_dim, mb_size, nc,device, num_parts,num_parts_h,num_parts_w, part_h,part_w, args.transformation, M).to(device)
parameters = filter(lambda p: p.requires_grad, model.parameters())

#optimizer and lr scheduling
optimizer = optim.Adadelta(model.parameters())
#optimizer = optim.Adam(model.parameters(),lr=1e-3)
l2 = lambda epoch: pow((1.-1.*epoch/epochs),0.9)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l2)


for epoch in range(epochs):
    scheduler.step()
    train(epoch)
    
    #if epoch==49:
    #    test(epoch)
torch.save(model.state_dict(), './ModDict/n'+str(clnum)+'.model')
