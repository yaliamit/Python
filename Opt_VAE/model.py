import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pylab as py

class VAE(nn.Module):

    def __init__(self, x_h, x_w, h_dim, z_dim, device):
        super(VAE, self).__init__()

        self.x_dim = x_h * x_w # height * width
        self.h = x_h
        self.w = x_w
        self.h_dim = h_dim # hidden layer
        self.z_dim = z_dim # generic latent variable

        self.dv = device

        """
        encoder: two fc layers
        """
        self.x2h = nn.Linear(self.x_dim, h_dim)
#        self.x2h = nn.Sequential(
#            nn.Linear(self.x_dim, h_dim),
#            nn.ReLU(),
#            nn.Linear(self.h_dim, self.h_dim)
#            )

        self.h2zmu = nn.Linear(h_dim, z_dim)
        self.h2zvar = nn.Linear(h_dim, z_dim)

        self.z2h = nn.Linear(z_dim, h_dim)
#        self.z2h = nn.Sequential(
#            nn.Linear(self.z_dim, self.h_dim),
#            nn.ReLU(),
#            nn.Linear(self.h_dim, self.h_dim)
#            )
           

        """
        decoder: two fc layers
        """
        self.h2x = nn.Linear(self.h_dim, self.x_dim)
        

    def encode(self, inputs):
        h = F.relu(self.x2h(inputs))
        z_mu = self.h2zmu(h)
        z_var = F.threshold(self.h2zvar(h), -6, -6)

        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(self.dv)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = F.relu(self.z2h(z))
        x = torch.sigmoid(self.h2x(h))
        return x

    def forward(self, inputs):
        z_mu, z_var = self.encode(inputs.view(-1, self.x_dim))
        z = self.reparameterize(z_mu, z_var)
        x = self.decode(z)
        x = torch.clamp(x, 1e-6, 1-1e-6)

        return x, z_mu, z_var,z

    def simul(self,type):

        z = torch.randn(100, self.z_dim).to(self.dv)
        x = self.decode(z)
        x = torch.clamp(x, 1e-6, 1 - 1e-6)
        XX = x.cpu().detach().numpy()
        py.figure(figsize=(20, 20))
        for i in range(100):
            py.subplot(10, 10, i + 1)
            py.imshow(1. - XX[i].reshape((28, 28)), cmap='gray')
            py.axis('off')
        py.savefig('fig_'+type+'.png')
        return(x)