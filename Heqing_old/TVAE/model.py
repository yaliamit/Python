import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tps import TPSGridGen


class TVAE(nn.Module):

    def __init__(self, x_h, x_w, h_dim, z_dim, u_dim, mb_size, device, t='aff'):
        super(TVAE, self).__init__()

        self.x_dim = x_h * x_w # height * width
        self.h = x_h
        self.w = x_w
        self.h_dim = h_dim # hidden layer
        self.z_dim = z_dim # generic latent variable
        self.u1_dim = u_dim # dimension of u'
        self.bsz = mb_size

        self.dv = device

        """
        encoder: two fc layers
        """
        #nn.Sequential(
         #   nn.Linear(self.x_dim, h_dim),
            #nn.LeakyReLU(0.1),
            #nn.Linear(self.h_dim, self.h_dim)
         #   )
        self.x2h = nn.Linear(self.x_dim, h_dim)
        #self.h2mu=nn.Linear(h_dim,z_dim+self.u1_dim)
        #self.h2var=nn.Linear(h_dim,z_dim+self.u1_dim)
        self.h2zmu = nn.Linear(h_dim, z_dim)
        self.h2zvar = nn.Linear(h_dim, z_dim)
        self.tf = t
        if t == 'aff':
            self.u2_dim = 6 # u = Wu' + b
            idty = torch.cat((torch.eye(2),torch.zeros(2).unsqueeze(1)),dim=1)
        elif t == 'tps':
            self.u2_dim = 18 # 2 * 3 ** 2
            self.gridGen = TPSGridGen(out_h=self.h,out_w=self.w,device=self.dv)
            px = self.gridGen.PX.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
            py = self.gridGen.PY.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
            idty = torch.cat((px,py))
        else:
            raise ValueError( """An invalid option for transformation type was supplied, options are ['aff' or 'tps']""")
        
        self.id = idty.expand((mb_size,)+idty.size()).to(self.dv)
        self.h2x = nn.Linear(self.h_dim, self.x_dim)

        self.h2umu = nn.Linear(h_dim, self.u1_dim)
        self.h2uvar = nn.Linear(h_dim, self.u1_dim)

        self.u2u = nn.Linear(self.u1_dim,self.u2_dim)
        self.z2h = nn.Linear(z_dim, h_dim)

        """
        decoder: two fc layers
        """
        #nn.Sequential(
            #nn.Linear(self.h_dim, self.h_dim),
            #nn.LeakyReLU(0.1),
           # nn.Linear(self.h_dim, self.x_dim)
        #)

    def forward_encoder(self, inputs):
        h = F.relu(self.x2h(inputs))
        z_mu = self.h2zmu(h)
        z_var = F.threshold(self.h2zvar(h), -6, -6)
        u_mu = F.tanh(self.h2umu(h))
        u_var = F.threshold(self.h2uvar(h), -6, -6)

        return z_mu, z_var, u_mu, u_var

    def sample_z(self, mu, logvar):
        eps = torch.randn(self.bsz, self.z_dim).to(self.dv)
        return mu + torch.exp(logvar/2) * eps

    def sample_u(self, mu, logvar):
        eps = torch.randn(self.bsz, self.u1_dim).to(self.dv)
        return mu + torch.exp(logvar/2) * eps

    def forward_decoder(self, z):
        h = F.relu(self.z2h(z))
        x = F.sigmoid(self.h2x(h))
        return x

    def forward(self, inputs):
        z_mu, z_var, u_mu, u_var = self.forward_encoder(inputs.view(-1, self.x_dim))
        z = self.sample_z(z_mu, z_var)
        x = self.forward_decoder(z)
        u = self.sample_u(u_mu, u_var)
        if self.tf == 'aff':
            self.theta = self.u2u(u).view(-1, 2, 3) + self.id
            grid = F.affine_grid(self.theta, x.view(-1,self.h,self.w).unsqueeze(1).size())
        else:
            self.theta = self.u2u(u) + self.id
            grid = self.gridGen(self.theta)
        x = F.grid_sample(x.view(-1,self.h,self.w).unsqueeze(1), grid, padding_mode='border')
        x = x.clamp(1e-6, 1-1e-6)
        return x, z_mu, z_var, u_mu, u_var

    def sample_from_z_prior(self, theta=None):
        eps = torch.randn(self.bsz, self.z_dim).to(self.dv)
        x = self.forward_decoder(eps)
        if theta is not None:
            theta = self.u2u(theta)
            if self.tf == 'aff':
                grid = F.affine_grid(theta.view(-1,2,3)+self.id, x.view(-1, self.h, self.w).unsqueeze(1).size())
            else:
                grid = self.gridGen(theta+self.id)
            x = F.grid_sample(x.view(-1,self.h,self.w).unsqueeze(1), grid)
        return x
