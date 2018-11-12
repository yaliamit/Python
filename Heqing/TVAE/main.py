import torch
from torch import nn, optim
from torchvision import transforms, datasets
from model import TVAE
import torch.nn.functional as F
from torchvision.utils import save_image
from math import pow

import argparse

parser = argparse.ArgumentParser(
    description='Variational Autoencoder with Spatial Transformation'
)

parser.add_argument('--transformation', default='aff',
                    help='type of transformation: aff or tps')
parser.add_argument('--zdim', type=int, help='dimension of z')
parser.add_argument('--udim', type=int, help='dimension of u')
parser.add_argument('--nepoch', type=int, default=40, help='number of training epochs')
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
z_dim = args.zdim
u_dim = args.udim
epochs = args.nepoch
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


model = TVAE(h, w, h_dim, z_dim, u_dim, mb_size, device, args.transformation).to(device)
#parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adadelta(model.parameters())
l2 = lambda epoch: pow((1.-1.*epoch/epochs),0.9)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l2)
criterion = nn.CrossEntropyLoss(size_average=False)


def loss_V(recon_x, x, mu1, logvar1, mu2, logvar2):
    BCE = F.binary_cross_entropy(recon_x.squeeze().view(-1, x_dim), x.view(-1, x_dim), size_average=False)
    KLD1 = -0.5 * torch.sum(1 + logvar1 - mu1**2 - torch.exp(logvar1)) # z
    KLD2 = -0.5 * torch.sum(1 + logvar2 - mu2**2 - torch.exp(logvar2)) # u'

    return BCE, KLD1 + KLD2

def test(epoch):
    model.eval()
    test_recon_loss = 0
    total_correct = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(te):
            data = data.to(device)
            target = target.to(device)
            recon_batch, zmu, zlogvar, umu, ulogvar = model(data)
            recon_loss, kl = loss_V(recon_batch, data, zmu, zlogvar, umu, ulogvar)
            loss = recon_loss + kl
            test_recon_loss += recon_loss.item()
            
    test_recon_loss /= (len(te) * mb_size)
    print('====> Epoch:{} Test reconstruction loss: {:.4f}'.format(epoch, test_recon_loss))

def train(epoch):
    model.train()
    tr_recon_loss = 0
    c_loss = 0
    total_correct = 0
    
    for batch_idx, (data, target) in enumerate(tr):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        recon_batch, zmu, zlogvar, umu, ulogvar = model(data)
        recon_loss, kl = loss_V(recon_batch, data, zmu, zlogvar, umu, ulogvar)
        loss = recon_loss + kl
        loss.backward()
        tr_recon_loss += recon_loss.item()
        optimizer.step()

    print('====> Epoch: {} Reconstruction loss: {:.4f}'.format(
          epoch, tr_recon_loss / (len(tr)*mb_size)))


for epoch in range(epochs):
    scheduler.step()
    train(epoch)
    test(epoch)
