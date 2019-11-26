import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np


class encoder_mix(nn.Module):
    def __init__(self,model):
        super(encoder_mix,self).__init__()
        self.num_layers=model.num_hlayers
        self.n_mix=model.n_mix
        self.x_dim=model.x_dim

        self.feats=0
        if (self.num_layers==1):
            self.h2he = nn.Linear(model.h_dim, model.h_dim)
        self.x2h = nn.Linear(model.x_dim, model.h_dim)
        self.x2hpi = nn.Linear(model.x_dim, model.h_dim)
        self.h2smu = nn.Linear(model.h_dim, model.s_dim * model.n_mix)
        self.h2svar = nn.Linear(model.h_dim, model.s_dim * model.n_mix, bias=False)
        self.h2pi = nn.Linear(model.h_dim, model.n_mix)
        if (model.feats):
            self.feats=model.feats
            self.conv=model.conv
            self.pool=model.pool



    def forward(self,inputs):
        if (self.feats):
            inputs=self.pool(self.conv(inputs))
        inputs=inputs.view(-1, self.x_dim)
        h = F.relu(self.x2h(inputs))
        hpi = F.relu(self.x2hpi(inputs))
        if (self.num_layers == 1):
            h = F.relu(self.h2he(h))
        s_mu = self.h2smu(h)
        s_logvar = F.threshold(self.h2svar(h), -6, -6)
        hm = self.h2pi(hpi).clamp(-10., 10.)
        pi = torch.softmax(hm, dim=1)
        return s_mu, s_logvar, pi


class diag(nn.Module):
    def __init__(self,dim):
        super(diag,self).__init__()

        self.dim=dim
        rim=(torch.rand(self.dim) - .5) / np.sqrt(self.dim)
        self.mu=nn.Parameter(rim)
        ris=(torch.rand(self.dim) - .5) / np.sqrt(self.dim)
        self.sigma=nn.Parameter(ris)

    def forward(self,z):

        u=z*self.sigma+self.mu
        return(u)

class bias(nn.Module):
    def __init__(self,dim):
        super(bias,self).__init__()

        self.dim=dim
        self.bias=nn.Parameter(6*(torch.rand(self.dim) - .5)/ np.sqrt(self.dim))

    def forward(self,z):
        return(self.bias.repeat(z.shape[0],1))

class ident(nn.Module):
    def __init__(self):
        super(ident,self).__init__()

    def forward(self,z):
        return(torch.ones(z.shape[0]))

class iden_copy(nn.Module):
    def __init__(self):
        super(iden_copy,self).__init__()

    def forward(self,z):
        return(z)

# Each set of s_dim normals gets multiplied by its own matrix to correlate
class fromNorm_mix(nn.Module):
    def __init__(self,model):
        super(fromNorm_mix,self).__init__()

        self.z2h=[]
        self.n_mix=model.n_mix
        self.z_dim=model.z_dim
        self.h_dim=model.h_dim
        self.u_dim=model.u_dim
        self.diag=model.diag
        self.type=model.type
        self.h_dim_dec = model.h_dim_dec
        if (self.h_dim_dec is None):
            h_dim=self.h_dim
        else:
            h_dim=self.h_dim_dec


        if (self.z_dim > 0):
            # If full matrix we get correlated gaussian in next level
            if (not self.diag):
                self.z2z=nn.ModuleList([nn.Linear(self.z_dim, self.z_dim) for i in range(self.n_mix)])
            # Diagonal covariance matrix for next level
            else:
                self.z2z=nn.ModuleList(diag(self.z_dim) for i in range(self.n_mix))
        # No free latent variables
        else:
            # Dummy step for z2z
            self.z2z=nn.ModuleList(ident() for i in range(self.n_mix))
        if (self.type == 'tvae'):
            self.u2u = nn.ModuleList([nn.Linear(self.u_dim, self.u_dim, bias=False) for i in range(self.n_mix)])
            for ll in self.u2u:
                ll.weight.data.fill_(0.)

        if (self.z_dim>0):
                self.z2h=nn.ModuleList([nn.Linear(self.z_dim, h_dim) for i in range(self.n_mix)])
        # No free latent variables - so we just make a fixed bias term 'template'
        else:
                self.z2h=nn.ModuleList([bias(h_dim) for i in range(self.n_mix)])

    def forward(self,z,u,rng=None):

        h=[]
        v=[]
        for i,zz,vv in zip(rng,z,u):
            h = h + [self.z2h[i](self.z2z[i](zz))]
            if (self.type=='tvae'):
                v=v+[self.u2u[i](vv)]

        hh=torch.stack(h,dim=0)
        hh=F.relu(hh)
        return hh, v


# Each correlated normal coming out of fromNorm_mix goes through same network to produce an image these get mixed.
class decoder_mix(nn.Module):
    def __init__(self,model,args):
        super(decoder_mix,self).__init__()
        self.x_dim=model.x_dim
        self.n_mix=model.n_mix
        self.u_dim=model.u_dim
        self.z_dim=model.z_dim
        self.h_dim=model.h_dim
        self.h_dim_dec=args.hdim_dec
        self.num_layers=model.num_hlayers
        self.type=model.type
        self.diag=args.Diag
        if (self.num_layers==1):
            if self.h_dim_dec is None:
                self.h2hd = nn.Linear(self.h_dim, self.h_dim)
            else:
                self.h2hd = nn.ModuleList([nn.Linear(self.h_dim_dec,self.h_dim) for i in range(self.n_mix)])
        # The bias term estimated in fromNorm is the template.

        if self.h_dim_dec is None:
            if (self.h_dim==self.x_dim):
                self.h2x=nn.Identity()
            else:
                self.h2x = nn.Linear(self.h_dim, self.x_dim)
        else:
            if (self.h_dim==self.x_dim):
                self.h2x=nn.ModuleList(nn.Identity() for i in range(self.n_mix))
            else:
                self.h2x = nn.ModuleList([nn.Linear(self.h_dim, self.x_dim) for i in range(self.n_mix)])
        self.fromNorm_mix = fromNorm_mix(self)
        self.feats=0
        if (model.feats):
            self.feats=model.feats
            self.deconv=model.deconv
            self.x_hf = model.x_hf
            self.x_h=model.x_h


    def forward(self,s,rng=None):

            if (rng is None):
                rng=range(s.shape[0])
            u = s.narrow(len(s.shape) - 1, 0, self.u_dim)
            z = s.narrow(len(s.shape) - 1, self.u_dim, self.z_dim)
            h, u = self.fromNorm_mix.forward(z, u, rng)
            if (self.num_layers==1):
                hh=[]
                for h_,r in zip(h,rng):
                    if self.h_dim_dec is None:
                        hh=hh+[self.h2hd(h_)]
                    else:
                        hh = hh + [self.h2hd[r](h_)]
                h=torch.stack(hh,dim=0)
                h=F.relu(h)
            x=[]


            for h_, r in zip(h, rng):
                if self.h_dim_dec is None:
                    xx=self.h2x(h_)
                    if (self.feats):
                        xx=self.deconv(xx.reshape(-1,self.feats,self.x_hf,self.x_hf))
                        xx=xx.reshape(-1,self.x_h*self.x_h)
                    x += [xx]
                else:
                    xx=self.h2x[r]
                    if (self.feats):
                        xx=self.deconv(xx)
                    x += [xx]

            xx=torch.stack(x,dim=0)
            xx=torch.sigmoid(xx)
            return(xx,u)

