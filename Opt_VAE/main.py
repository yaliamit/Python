# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:25:55 2019

@author: Qing Yan
"""
from torch import nn

import torch
from torch import optim
from torchvision import transforms, datasets
from model import VAE
from opt_model import OPT_VAE
import torch.nn.functional as F

from math import pow
import numpy as np
from torch.autograd import Variable

use_gpu = torch.cuda.is_available()

torch.manual_seed(1234)
torch.cuda.manual_seed(143)
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if use_gpu else "cpu")

'''Parameters, adjust here'''
mb_size = 1000 # batch size
h = 28
w = 28
x_dim = h*w
log_interval = 100 # for reporting

epochs = 40

kwargs = {'num_workers': 8, 'pin_memory': True} if use_gpu else {}

mnist_tr = datasets.MNIST(root='../', download=True, transform=transforms.ToTensor())
mnist_te = datasets.MNIST(root='../', download=True, train=False, transform=transforms.ToTensor())


tr = torch.utils.data.DataLoader(dataset=mnist_tr,
                                batch_size=mb_size,
                                shuffle=False,
                                drop_last=True, **kwargs)


te = torch.utils.data.DataLoader(dataset=mnist_te,
                                batch_size=mb_size,
                                shuffle=False,
                                drop_last=True, **kwargs)
def loss_V(recon_x, x, mu, std):
    '''loss = reconstruction loss + KL_z + KL_u'''
    BCE = F.binary_cross_entropy(recon_x.squeeze().view(-1, x_dim), x.view(-1, x_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + 2*torch.log(std) - mu**2 - std**2) # z
    return BCE, KLD

def train_vae(tr_size):
    model = VAE(h,w,256,20,device).to(device)
    optimizer = optim.Adadelta(model.parameters())
    l2 = lambda epoch: pow((1.-1.*epoch/epochs),0.9)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=l2)
    for epoch in range(epochs):
        scheduler.step()
        model.train()
        tr_recon_loss = 0
        for batch_idx, (data, target) in enumerate(tr):
            if batch_idx >= tr_size:
                break
            data = data.to(device)
            optimizer.zero_grad()
    
            recon_batch, zmu, zvar,_= model(data)
            recon_loss, kl = loss_V(recon_batch, data, zmu,torch.exp(0.5*zvar))
        
            loss = recon_loss + kl 
            loss.backward()
            tr_recon_loss += recon_loss.item()
        
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tReconstruction-Loss: {:.4f}, KL: {:.4f}'.format(
                        epoch, batch_idx * len(data), len(mnist_tr),
                        100. * batch_idx / len(tr),
                        recon_loss / len(data),kl/len(data)))

        print('====> Epoch: {} Reconstruction loss: {:.4f}'.format(
                epoch, tr_recon_loss / (tr_size*mb_size)))
        #test(epoch,model)
    return model


def test(model):
    model.eval()
    test_loss = 0
    test_recon = 0
    with torch.no_grad():
        for _, (data, target) in enumerate(te):
            data = data.to(device)
            recon_batch, zmu, zvar,z= model(data)
            recon_loss, kl = loss_V(recon_batch, data, zmu,torch.exp(0.5*zvar))
            loss = recon_loss + kl
            test_loss += loss.item()
            test_recon += recon_loss.item()
            
    test_loss /= (len(te)*mb_size)
    test_recon /= (len(te)*mb_size)
#    print('====> Epoch:{} test ELBO_loss: {:.4f}'.format(test_loss))
#    print('====> Epoch:{} test recon_loss: {:.4f}'.format(test_recon))
    return test_recon, test_loss


def train(epoch,img,model,optimizer):
    model.train()
    data = img.to(device)
    optimizer.zero_grad()
    recon,zmu,zvar, z = model(data)
    recon_loss, kl = loss_V(recon, data, zmu,torch.exp(0.5*zvar))
    loss = recon_loss +kl 
    loss.backward(retain_graph=True)
    
    optimizer.step()
    return recon_loss.item(), kl.item()

def train_opt_vae(tr_size):
    model = OPT_VAE(h, w, 256, 20, torch.zeros(mb_size,20).to(device), 
                    torch.zeros(mb_size,20).to(device), device).to(device)
    
    optimizer2 = optim.Adam(model.parameters(),lr=1e-3)
    optimizer1 = optim.Adam([model.z_mu,model.z_var],lr = 0.2)
    zmu_dict = torch.zeros(len(tr),mb_size,20).to(device)
    zlogvar_dict = torch.zeros(len(tr),mb_size,20).to(device)
    for epoch in range(epochs):
        loss_tr_recon = 0.0
        loss_tr_kl = 0.0
        for batch_idx, (data, target) in enumerate(tr):
            if batch_idx >= tr_size:
                break
            data = data.to(device)
            model.update_z(zmu_dict[batch_idx,:,:],zlogvar_dict[batch_idx,:,:])
            for num in range(2):
                train(epoch,data,model,optimizer1)
                recon_ls,kl = train(epoch,data,model,optimizer2)
            loss_tr_recon += recon_ls
            loss_tr_kl += kl
            zmu_dict[batch_idx,:,:] = model.z_mu.data
            zlogvar_dict[batch_idx,:,:] = model.z_var.data
        print('====> Epoch: {} Reconstruction loss: {:.4f}'.format(epoch,loss_tr_recon/tr_size/mb_size)) 
        print('====> Epoch: {} KL loss: {:.4f}'.format(epoch,loss_tr_kl/tr_size/mb_size)) 
        
    return model

def test_opt(model,epc):
    test_recon_loss = 0.0
    test_ELBO_loss = 0.0
    optimizer = optim.Adam([model.z_mu,model.z_var],lr = 0.2)
    for _, (data, target) in enumerate(te):
        data = data.to(device)
        te_size = data.shape[0]
        model.update_z(torch.zeros(te_size,20).to(device),torch.zeros(te_size,20).to(device))
        for epoch in range(epc):
            model.train()
            optimizer.zero_grad()
            recon,zmu,zvar,z = model(data)
            recon_loss, kl = loss_V(recon, data, zmu,torch.exp(0.5*zvar))
            loss = recon_loss +kl 
            loss.backward()
            optimizer.step()
            
        recon_batch, zmu, zvar,z= model(data)
        recon_loss, kl = loss_V(recon_batch, data, zmu,torch.exp(0.5*zvar))
        loss = recon_loss + kl
        test_recon_loss += recon_loss.item() 
        test_ELBO_loss += loss.item()
    test_recon_loss /= (len(te)*mb_size)
    test_ELBO_loss /= (len(te)*mb_size)
    #print('====> Epoch:{} recon_loss: {:.4f}'.format(epoch, test_recon_loss))
    return test_recon_loss, test_ELBO_loss

def test_opt_decoder(model,epc):
    test_recon_loss = 0.0
    test_ELBO_loss = 0.0
    for _, (data, target) in enumerate(te):
        data = data.to(device)
        te_size = data.shape[0]
        z_mu = Variable(torch.zeros(te_size,20).cuda(),requires_grad = True)
        z_logvar = Variable(torch.zeros(te_size,20).cuda(),requires_grad = True)
        optimizer = optim.Adam([z_mu,z_logvar],lr = 0.2)
        for epoch in range(epc):
            model.train()
            optimizer.zero_grad()
            latent = model.reparameterize(z_mu,z_logvar)
            recon = model.decode(latent)
            recon_loss, kl = loss_V(recon, data, z_mu,torch.exp(0.5*z_logvar))
            loss = recon_loss +kl 
            loss.backward()
            optimizer.step()
        test_recon_loss += recon_loss.item() 
        test_ELBO_loss += loss.item()
    test_recon_loss /= (len(te)*mb_size)
    test_ELBO_loss /= (len(te)*mb_size)
    #print('====> Epoch:{} recon_loss: {:.4f}'.format(epoch, test_recon_loss))
    return test_recon_loss, test_ELBO_loss


if __name__ == '__main__': 
    text_file = open("Output.txt", "w")
    batch_list = [60] #[1,2,3,4,5,6,7,8,9,10]
    for i in range(1):
        tr_size = batch_list[i]
        text_file.write("tr_size: %s \n" % tr_size)

        vae_model = train_vae(tr_size)
        test_recon, test_loss = test(vae_model)

        text_file.write("VAE recon loss: %s \n" % test_recon)
        text_file.write("VAE ELBO: %s \n" % test_loss)

        opt_model = train_opt_vae(tr_size)
        test_recon, test_loss = test_opt(opt_model,500)

        text_file.write("VLO recon loss: %s \n" % test_recon)
        text_file.write("VLO ELBO: %s \n" % test_loss)

        test_recon_loss, test_ELBO_loss = test_opt_decoder(vae_model,500)

        text_file.write("Opt-decoder recon loss: %s \n" % test_recon_loss)
        text_file.write("OPT-decoder ELBO: %s \n" % test_ELBO_loss)
        
    text_file.close()   
