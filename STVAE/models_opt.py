import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np

import models

class STVAE_OPT(models.STVAE):

    def __init__(self, x_h, x_w, device, args):
        super(STVAE_OPT, self).__init__(x_h, x_w, device, args)

        self.MM=args.MM
        if (self.MM):
            self.MU=nn.Parameter(torch.zeros(self.s_dim))
            self.LOGVAR=nn.Parameter(torch.zeros(self.s_dim))

        self.mu_lr=args.mu_lr #torch.full([self.s_dim],args.mu_lr).to(self.dv)
        #if 'tvae' in self.type:
        #    self.mu_lr[0:self.u_dim]*=.1

        self.mu = torch.autograd.Variable(torch.zeros(self.bsz,self.s_dim), requires_grad=True)
        self.logvar = torch.autograd.Variable(torch.zeros(self.bsz,self.s_dim), requires_grad=True)
        self.s2s=None
        self.u2u=None

        if (args.optimizer=='Adam'):
            self.optimizer=optim.Adam(self.parameters(),lr=args.lr)
        elif (args.optimizer=='Adadelta'):
            self.optimizer = optim.Adadelta(self.parameters())
        else:
            self.optimizer = optim.SGD(lr=args.lr)
        self.optimizer_s = optim.Adam([self.mu, self.logvar], lr=0.2)
        print('s_dim',self.s_dim,'u_dim',self.u_dim,'z_dim',self.z_dim,self.type)


    def update_s(self,mu,logvar):
        self.mu.data = mu
        self.logvar.data = logvar

    def forw(self, mub,logvarb):

        if (self.type is not 'ae' and not self.MM):
            s = self.sample(mub, logvarb, self.s_dim)
        else:
            s=mub
        x=self.decoder_and_trans(s)
        return x



    def loss_V(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.squeeze().view(-1, self.x_dim), x.view(-1, self.x_dim), reduction='sum')
        if self.MM:
            KLD1 = 0.5*torch.sum((mu-self.MU)*(mu-self.MU)/torch.exp(self.LOGVAR)+self.LOGVAR)
        else:
            KLD1 = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar))  # z
        return BCE, KLD1

    def compute_loss_and_grad(self,data, type, optim,opt='par'):

        if (type == 'train' or opt=='mu'):
            optim.zero_grad()
        recon_batch = self.forw(self.mu, self.logvar)
        recon_loss, kl = self.loss_V(recon_batch, data, self.mu, self.logvar)
        loss = recon_loss + kl
        if (type == 'train' or opt=='mu'):
            loss.backward()
            optim.step()

        return recon_batch, recon_loss, loss


    def run_epoch(self, train,  epoch,num_mu_iter,MU, LOGVAR,type='test',fout=None):
        if (type=='train'):
            self.train()
        tr_recon_loss = 0; tr_full_loss=0
        ii = np.arange(0, train[0].shape[0], 1)
        #if (type=='train'):
        #   np.random.shuffle(ii)
        tr =train[0][ii].transpose(0,3,1,2)
        y = train[1][ii]
        mu=MU[ii]
        logvar=LOGVAR[ii]
        batch_size = self.bsz
        for j in np.arange(0, len(y), batch_size):

            data = torch.tensor(tr[j:j + batch_size]).float()
            data = data.to(self.dv)

            #target = torch.tensor(y[j:j + batch_size]).float()

            self.update_s(mu[j:j+batch_size, :], logvar[j:j+batch_size, :])
            self.optimizer_s = optim.Adam([self.mu, self.logvar], lr=self.mu_lr)
            for it in range(num_mu_iter):
                self.compute_loss_and_grad(data, type,self.optimizer_s,opt='mu')
            recon_batch, recon_loss, loss = self.compute_loss_and_grad(data,type,self.optimizer,opt='par')

            mu[j:j + batch_size] = self.mu.data
            logvar[j:j + batch_size] = self.logvar.data

            tr_recon_loss += recon_loss
            tr_full_loss += loss
        if (fout is  None):
            print('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}, KL loss: {:.4F}'.format(type,
                    epoch, tr_recon_loss/len(tr), tr_full_loss/len(tr),-tr_recon_loss/len(tr)+tr_full_loss/len(tr)))
        else:
            fout.write('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F},KL loss: {:.4F}\n'.format(type,
                    epoch, tr_recon_loss/len(tr), tr_full_loss/len(tr),-tr_recon_loss/len(tr)+tr_full_loss/len(tr)))
        return mu,logvar


    def recon_from_zero(self,input,num_mu_iter=10):

        num_inp=input.shape[0]
        if ('tvae' in self.type):
            self.id = self.idty.expand((num_inp,) + self.idty.size()).to(self.dv)

        mub = torch.autograd.Variable(torch.zeros(num_inp,self.s_dim, requires_grad=True).float(),
                                      requires_grad=True)
        logvarb = torch.autograd.Variable(torch.zeros(num_inp,self.s_dim, requires_grad=True).float(),
                                          requires_grad=True)
        inp=input.transpose(0,3,1,2)
        data = (torch.tensor(inp).float()).to(self.dv)
        mub = mub.to(self.dv)
        logvarb = logvarb.to(self.dv)
        for muit in range(num_mu_iter):
            recon_batch = self.forw(data, mub, logvarb)
            recon_loss, kl = self.loss_V(recon_batch, data, mub, logvarb)
            loss = recon_loss + kl
            dd = torch.autograd.grad(loss, mub)
            mub = mub - (self.mu_lr * dd[0])

        return recon_batch

    def sample_from_z_prior(self,theta=None):
        s = torch.randn(self.bsz, self.s_dim).to(self.dv)
        if self.MM:
            s=s*torch.exp(self.LOGVAR/2)+self.MU
        theta=theta.to(self.dv)
        if (theta is not None and self.u_dim>0):
            s[:,0:self.u_dim]=theta

        x=self.decoder_and_trans(s)

        return x


