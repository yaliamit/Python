import torch
from torch import nn, optim
from torchvision import transforms, datasets
from models import STVAE
import torch.nn.functional as F
from torchvision.utils import save_image

import argparse

parser = argparse.ArgumentParser(
    description='Variational Autoencoder with Spatial Transformation'
)

parser.add_argument('--transformation', default='aff',
                    help='type of transformation: aff or tps')
parser.add_argument('--sdim', type=int, help='dimension of s')
parser.add_argument('--nepoch', type=int, default=30, help='number of training epochs')
parser.add_argument('--save', default='output/model.pt')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')

args = parser.parse_args()

use_gpu = args.gpu and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_gpu else "cpu")

mb_size = 128 # batch size
h = 28
w = 28
x_dim = h * w # image size
s_dim = args.sdim
nepoch = args.nepoch
h_dim = 256 # hidden dimension
log_interval = 100 # for reporting
#lr = 1e-3

kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
# add 'download=True' when use it for the first time
mnist_tr = datasets.MNIST(root='../MNIST/', transform=transforms.ToTensor())
mnist_te = datasets.MNIST(root='../MNIST/', train=False, transform=transforms.ToTensor())
tr = torch.utils.data.DataLoader(dataset=mnist_tr,
                                batch_size=mb_size,
                                shuffle=True,
                                drop_last=True, **kwargs)
te = torch.utils.data.DataLoader(dataset=mnist_te,
                                batch_size=mb_size,
                                shuffle=True,
                                drop_last=True, **kwargs)


model = STVAE(h, w, h_dim, s_dim, mb_size, device, args.transformation).to(device)
optimizer = optim.Adadelta(model.parameters())
l2 = lambda epoch: pow((1.-1. * epoch/nepoch),0.9)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l2)


def loss_V(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.squeeze().view(-1, x_dim), x.view(-1, x_dim), size_average=False)
    KLD1 = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar)) # z

    return BCE, KLD1

def test(epoch):
    model.eval()
    test_recon_loss = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(te):
            data = data.to(device)
            target = target.to(device)
            recon_batch, smu, slogvar = model(data)
            recon_loss, kl = loss_V(recon_batch, data, smu, slogvar)
            loss = recon_loss + kl
            test_recon_loss += recon_loss.item()
            
    test_recon_loss /= (len(te) * mb_size)
    print('====> Epoch:{} Test reconstruction loss: {:.4f}'.format(epoch, test_recon_loss))

def train(epoch):
    model.train()
    tr_recon_loss = 0
    
    for _, (data, target) in enumerate(tr):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        recon_batch, smu, slogvar = model(data)
        recon_loss, kl = loss_V(recon_batch, data, smu, slogvar)
        
        loss = recon_loss + kl
        loss.backward()
        tr_recon_loss += recon_loss.item()
        optimizer.step()

    print('====> Epoch: {} Reconstruction loss: {:.4f}'.format(
          epoch, tr_recon_loss / (len(tr)*mb_size)))


for epoch in range(nepoch):
    scheduler.step()
    train(epoch)
    test(epoch)
