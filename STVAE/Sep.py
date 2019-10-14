import torch
from torch import nn
import torch.nn.functional as F

class toNorm_mix_sep(nn.Module):
    def __init__(self,h_dim,s_dim, n_mix):
        super(toNorm_mix_sep,self).__init__()
        self.h2smu = nn.ModuleList([nn.Linear(h_dim, s_dim) for i in range(n_mix)])
        self.h2svar = nn.ModuleList([nn.Linear(h_dim, s_dim,bias=False) for i in range(n_mix)])
        self.h2pi = nn.Linear(h_dim,n_mix) #nn.Parameter(torch.zeros(h_dim,n_mix))


class encoder_mix_sep(nn.Module):
    def __init__(self,x_dim,h_dim,s_dim,n_mix):
        super(encoder_mix_sep,self).__init__()
        self.n_mix=n_mix
        self.s_dim=s_dim
        self.x2h=nn.ModuleList([nn.Linear(x_dim, h_dim) for i in range(n_mix)])
        self.x2hpi = nn.Linear(x_dim, h_dim)
        self.toNorm_mix_sep=toNorm_mix_sep(h_dim,s_dim,n_mix)

    def forward(self,inputs):
        s_mu = []
        s_logvar = []
        for i in range(self.n_mix):
            h = F.relu(self.x2h[i](inputs))
            s_mu += [self.toNorm_mix_sep.h2smu[i](h)]
            s_logvar += [F.threshold(self.toNorm_mix_sep.h2svar[i](h), -6, 6)]
        s_mu = torch.stack(s_mu, dim=0).transpose(0, 1).reshape(-1, self.n_mix * self.s_dim)
        s_logvar = torch.stack(s_logvar, dim=0).transpose(0, 1).reshape(-1, self.n_mix * self.s_dim)
        hpi = F.relu(self.x2hpi(inputs))
        pi = torch.softmax(self.toNorm_mix_sep.h2pi(hpi).clamp(-10., 10.), dim=1)

        return s_mu, s_logvar, pi
