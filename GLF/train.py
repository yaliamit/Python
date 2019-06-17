"""
Code for training 1-stage GLF
"""
import torch
from torch import optim
from torchvision import transforms, datasets
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from AE.convAE_infoGAN_up import ConvAE
import argparse
from tqdm import tqdm
from nice.models import NICEModel
import os
from nice.loss import GaussianPriorNICELoss
from torch.utils.data import DataLoader
import torchvision.utils as t_utils 
from utils import perceptual_loss
from utils.dataset import return_data
from utils.make_one_hot import make_one_hot

mse = nn.MSELoss(reduction='sum')
def fpl_criterion(recon_features, targets):
    fpl = 0
    for f, target in zip(recon_features, targets):
        fpl += mse(f, target.detach())#.div(f.size(1))
    return fpl

def sample_save(model_ae,model_flow,args,device,epoch,noise = None):
    if noise is None:
        ys = torch.randn(80,args.num_latent).to(device)
    else:
        ys = noise
    model_ae.eval()
    model_flow.eval()
    if args.num_class > 1:
        t = ys.shape[0] // 10
        label = torch.tensor([0,1,2,3,4,5,6,7,8,9]).expand(t,-1).transpose(0,1).reshape(-1,1)      #torch.randint(0,args.num_class, size=(ys.shape[0],1))
        labels = make_one_hot(label,args.num_class).to(device)
    else:
        labels = 0
    zs = model_flow.inverse(ys,labels)
    newx = model_ae.decode(zs)
    
    if not os.path.exists(args.saveimdir):
        os.makedirs(args.saveimdir)
    t_utils.save_image(newx,args.saveimdir+'/sample_epo_'+str(epoch)+'.png',normalize=True)
    #if args.num_class > 1:
    #    np.savetxt(args.saveimdir+'/label_epo_'+str(epoch)+'.txt', label.detach().cpu().numpy(), newline=" ")

def load_check_point(args,device):
    model_ae = ConvAE(args).to(device)
    model_flow = NICEModel(args.num_latent, args.nhidden, args.nlayers, args.nbck, args.coupling,args.num_class).to(device)
    period = (args.check_point // 10) *10
    state_flow = torch.load(os.path.join(args.savedir, 'flowModel_epo_'+str(period)))
    state_ae = torch.load(os.path.join(args.savedir, 'AEModel'))
    model_ae.load_state_dict(state_ae)
    model_flow.load_state_dict(state_flow)
    return model_ae,model_flow

def save_recon(args,recon_batch):
    if not os.path.exists(args.saveimdir):
        os.makedirs(args.saveimdir)
    t_utils.save_image(recon_batch,args.saveimdir + '/recon.png',normalize=True)

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a fresh NICE model and save.")
    # configuration settings:
    parser.add_argument("--dataset", default='cifar', dest='dataset', choices=('mnist', 'cifar', 'celeba', 'Fashion-mnist'),
                        help="Dataset to train the GLF model on.")
    parser.add_argument("--dset_dir", dest='dset_dir', default="./data",
                        help="Where you store the dataset.")
    parser.add_argument("--epochs", dest='num_epochs', default=200, type=int,
                        help="Number of epochs to train on. [1500]")
    parser.add_argument("--batch_size", dest="batch_size", default=256, type=int,
                        help="Number of examples per batch. [16]")
    parser.add_argument("--num_workers", dest="num_workers", default=0, type=int,
                        help="Number of workers when load in dataset.")
    parser.add_argument('--device', dest = 'device',default=0, type=int, 
                        help='Index of device')
    parser.add_argument("--savedir", dest='savedir', default="./saved_models/",
                        help="Where to save the trained model.")
    parser.add_argument("--saveimdir", dest='saveimdir', default="./samples/",
                        help="Where to save the images during training.")
    parser.add_argument('--check_point', default=0, type=int, 
                        help='num epoch when stop, 0 means start from begining')
    parser.add_argument('--loss_type', default='MSE', type=str, 
                        help='Type of loss',choices = ('MSE','Perceptual','cross_entropy'))
    #Auto Encoder settings:
    parser.add_argument("--num_latent",  default=64, type=int,
                        help="dimension of latent code z")
    parser.add_argument("--image_size",  default=32, type=int,
                        help="size of training image")
    # NICE/FLOW model settings:
    parser.add_argument("--nonlinearity_layers", dest='nlayers', default=4, type=int,
                        help="Number of layers in the nonlinearity of each block. [4]")
    parser.add_argument("--nonlinearity_hiddens", dest='nhidden', default=100, type=int,
                        help="Hidden size of inner layers of nonlinearity. [100]")
    parser.add_argument("--num_block", dest='nbck', default=4, type=int,
                        help="num of flow blocks")
    parser.add_argument("--coupling", dest='coupling', default=1, choices = (0,1,2),type = int,
                        help="type of coupling layer: 0=add, 1=affine, 2=sigmoid")
    # class-related model
    parser.add_argument("--num_class", dest='num_class', default=0, type=int,
                        help="num of classes, 0 means non class related model. When > 1, labels should be provided.")
    # optimization settings:
    # todo: learning rate decay
    parser.add_argument("--lr", default=1e-3, dest='lr', type=float,
                        help="Learning rate for ADAM optimizer. [0.001]")
    parser.add_argument("--beta1", default=0.9,  dest= 'beta1', type=float,
                        help="Momentum for ADAM optimizer. [0.9]")
    parser.add_argument("--beta2", default=0.01, dest='beta2', type=float,
                        help="Beta2 for ADAM optimizer. [0.01]")
    parser.add_argument("--eps", default=0.001, dest='eps', type=float,
                        help="Epsilon for ADAM optimizer. [0.0001]")
    args = parser.parse_args()
    if args.loss_type == 'cross_entropy':
        assert (args.dataset == 'mnist' or args.dataset == 'Fashion-mnist'),"Cross entropy loss can only be used for mnist or Fashion-mnist."
    if args.num_class > 1:
        assert (args.dataset != 'celeba'), "class-related model can not be applied on celeba dataset."
    use_gpu = torch.cuda.is_available()

    torch.manual_seed(123)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda:{}".format(args.device) if use_gpu else "cpu")
    
    training_loader = return_data(args)

    if args.loss_type == 'MSE':
        recon_loss_fn = nn.MSELoss(reduction = 'sum')
    elif args.loss_type == 'cross_entropy':
        x_dim  = args.image_size ** 2
        def recon_loss_fn(recon_batch,dat):
            return F. binary_cross_entropy(recon_batch.squeeze().view(-1, x_dim), 
                                             dat.view(-1, x_dim),  reduction='sum')
    else:#perceptual loss
        def distance_metric(sz, force_l2=False):
            if sz == 32:
                return perceptual_loss._VGGDistance(3)
            elif sz == 64:
                return perceptual_loss._VGGDistance(4)
            else:
                assert False, "Perceptual loss only supports 32X32 and 64X64 images."
        
        recon_loss_fn = distance_metric(args.image_size)

    if args.check_point == 0:
        modAE = ConvAE(args).to(device)
        modFlow = NICEModel(args.num_latent, args.nhidden, args.nlayers,args.nbck,args.coupling,args.num_class).to(device)
    else:
        modAE,modFlow = load_check_point(args,device)
    optimizer1 = optim.Adam(modAE.parameters(), lr=args.lr)
    
    optimizer2 = optim.Adam(modFlow.parameters(), 
                            lr=args.lr, betas=(args.beta1,args.beta2), eps=args.eps,amsgrad=True)
    #scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1,50, gamma=0.5, last_epoch=-1)
    #scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 50, gamma=0.5, last_epoch=-1)
    
    nice_loss_fn = GaussianPriorNICELoss(size_average=True)
    def loss_fn(fx):
        return nice_loss_fn(fx, modFlow.scaling_diag)
    
    for epoch_ in range(args.num_epochs):
        epoch = epoch_+args.check_point
        modAE.train()
        modFlow.train()
        recon_losses = []
        like_losses = []
        lik_z = []
        with tqdm(total=len(training_loader.dataset)) as progress_bar:
            for batch_idx, (dat, target) in enumerate(training_loader):
                dat = dat.to(device)
                optimizer1.zero_grad()
                optimizer2.zero_grad()

                recon_batch,z= modAE(dat)
                if args.num_class > 1:
                    target = make_one_hot(target.reshape(-1,1),C=args.num_class).to(device)
                zhat, logd = modFlow(z,target)

                loss_recon = recon_loss_fn(recon_batch, dat)
                lz = loss_fn(zhat)
                lik_z.append(torch.mean(logd).item())
                loss_ll = lz - torch.mean(logd)
                like_losses.append(loss_ll.item())
                recon_losses.append(loss_recon.item()/args.batch_size)
                total_loss = loss_recon + loss_ll
                total_loss.backward(retain_graph = True)
                
                optimizer1.step()
                optimizer2.step()
                progress_bar.set_postfix(loss= np.mean(recon_losses), logd = np.mean(lik_z),
                    likloss = np.mean(like_losses))
                progress_bar.update(dat.size(0))
                if batch_idx == 99:
                    save_recon(args,recon_batch)
                
        #scheduler1.step()
        #scheduler2.step()    
        print('Train Epoch: {} Reconstruction-Loss: {:.4f} loglikelihood loss: {}  log det: {}'.format(
                    epoch, np.mean(recon_losses), np.mean(like_losses),np.mean(lik_z)))
        period = (epoch // 10)*10
        sample_save(modAE,modFlow,args,device,epoch)
        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)
        torch.save(modFlow.state_dict(), os.path.join(args.savedir, 'flowModel_epo_'+str(period)))
        torch.save(modAE.state_dict(), os.path.join(args.savedir, 'AEModel'))
    