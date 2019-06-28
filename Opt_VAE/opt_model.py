# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:27:00 2019

@author: Qing Yan
"""

import torch
#import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from scipy.misc import imsave

class OPT_VAE(nn.Module):
    '''
    input: x_h,x_w ---- height and width of image
           h_dim ---- dimension of hidden layer
           z_dim ---- dimension of latent variable
           z_mu,z_var ---- initial value of variables: mu and log var.
           device ---- 'cpu' or 'cuda' 
    '''
    def __init__(self, x_h, x_w, h_dim, z_dim, z_mu, z_var,device):
        super(OPT_VAE, self).__init__()

        self.x_dim = x_h * x_w # height * width
        self.h = x_h
        self.w = x_w
        self.h_dim = h_dim # hidden layer
        self.z_dim = z_dim # generic latent variable
        self.dv = device
        
        self.z_mu = Variable(z_mu,requires_grad=True)
        self.z_var = Variable(z_var,requires_grad=True)
        """
        decoder: two fc layers
        """
        self.z2h = nn.Linear(z_dim, h_dim)
        self.h2x = nn.Linear(self.h_dim, self.x_dim)
        
    def update_z(self,z_mu,z_var):
        self.z_mu.data = z_mu
        self.z_var.data = z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(self.dv)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = F.relu(self.z2h(z))
        x = torch.sigmoid(self.h2x(h))
        return x

    def forward(self, inputs):
        z = self.reparameterize(self.z_mu, self.z_var)
        x = self.decode(z)
        x = torch.clamp(x, 1e-6, 1-1e-6)

        return x, self.z_mu, self.z_var,z

    def simul(self,fname):
        z = torch.randn(100, self.z_dim).to(self.dv)
        x = self.decode(z)
        x = torch.clamp(x, 1e-6, 1 - 1e-6)
        XX = x.cpu().detach().numpy()
        x_mat=[]
        t=0
        for i in range(10):
            x_line=[]
            for j in range(10):
                x_line+=[XX[t].reshape((28,28))]
                t+=1
            x_mat+=[np.concatenate(x_line,axis=0)]
        manifold = np.concatenate(x_mat, axis=1)

        manifold = 1. - manifold[np.newaxis, :]
        print(manifold.shape)

        img = np.concatenate([manifold, manifold, manifold], axis=0)
        img = img.transpose(1, 2, 0)
        imsave(fname+'.jpg', img)

        return(x)