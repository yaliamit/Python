import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tps import TPSGridGen

class STVAE(nn.Module):

    def __init__(self, x_h, x_w, h_dim, s_dim, mb_size, device, t='aff'):
        super(STVAE, self).__init__()

        self.x_dim = x_h * x_w # height * width
        self.h = x_h
        self.w = x_w
        self.h_dim = h_dim # hidden layer
        self.s_dim = s_dim # generic latent variable
        self.bsz = mb_size

        self.dv = device

        """
        encoder: two fc layers
        """
        self.x2h = nn.Sequential(
            nn.Linear(self.x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim)
            )

        self.h2smu = nn.Linear(h_dim, s_dim)
        self.h2svar = nn.Linear(h_dim, s_dim)
        self.tf = t
        if t == 'aff':
            self.u_dim = 6
            idty = torch.cat((torch.eye(2),torch.zeros(2).unsqueeze(1)),dim=1)
        elif t == 'tps':
            self.u_dim = 18 # 2 * 3 ** 2
            self.gridGen = TPSGridGen(out_h=self.h,out_w=self.w,device=self.dv)
            px = self.gridGen.PX.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
            py = self.gridGen.PY.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
            idty = torch.cat((px,py))
        else:
            raise ValueError( """An invalid option for transformation type was supplied, options are ['aff' or 'tps']""")
        
        self.id = idty.expand((mb_size,)+idty.size()).to(self.dv)

        self.z_dim = self.s_dim-self.u_dim
        self.s2s = nn.Linear(self.s_dim, self.s_dim)

        self.z2h = nn.Linear(self.z_dim, self.h_dim)

        """
        decoder: two fc layers
        """
        self.h2x = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.x_dim)
        )

    def forward_encoder(self, inputs):
        h = F.relu(self.x2h(inputs))
        s_mu = self.h2smu(h)
        s_var = F.threshold(self.h2svar(h), -6, -6)

        return s_mu, s_var

    def sample_s(self, mu, logvar):
        eps = torch.randn(self.bsz, self.s_dim).to(self.dv)
        return mu + torch.exp(logvar/2) * eps

    def forward_decoder(self, z):
        h = F.relu(self.z2h(z))
        x = F.sigmoid(self.h2x(h))
        return x

    def forward(self, inputs):
        s_mu, s_var = self.forward_encoder(inputs.view(-1, self.x_dim))
        s = self.sample_s(s_mu, s_var)
        s = self.s2s(s)
        u = F.tanh(s.narrow(1,0,self.u_dim))
        z = s.narrow(1,self.u_dim,self.z_dim)
        x = self.forward_decoder(z)
        if self.tf == 'aff':
            self.theta = u.view(-1, 2, 3) + self.id
            grid = F.affine_grid(self.theta, x.view(-1,self.h,self.w).unsqueeze(1).size())
        else:
            self.theta = u + self.id
            grid = self.gridGen(self.theta)
        x = F.grid_sample(x.view(-1,self.h,self.w).unsqueeze(1), grid, padding_mode='border')
        x = x.clamp(1e-6, 1-1e-6)

        return x, s_mu, s_var
