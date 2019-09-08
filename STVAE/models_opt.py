import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import time
import models

class STVAE_OPT(models.STVAE):

    def __init__(self, x_h, x_w, device, args):
        super(STVAE_OPT, self).__init__(x_h, x_w, device, args)

        self.MM=args.MM
        if (self.MM):
            self.MU=nn.Parameter(torch.zeros(self.s_dim))
            self.LOGVAR=nn.Parameter(torch.zeros(self.s_dim))

        self.mu_lr=args.mu_lr
        self.s2s=None
        self.u2u=None

    def update_s(self,mu,logvar):
        self.mu=torch.autograd.Variable(mu, requires_grad=True)
        self.logvar = torch.autograd.Variable(logvar, requires_grad=True)
        self.optimizer_s = optim.Adam([self.mu, self.logvar], lr=self.mu_lr)

    def forw(self, mub,logvarb):

        if (self.type is not 'ae' and not self.MM):
            s = self.sample(mub, logvarb, self.s_dim)
        else:
            s=mub
        x=self.decoder_and_trans(s)
        ss_prior = torch.sum((s * s) / 2)
        sd = torch.exp(logvarb * .5)
        ss_posterior = -torch.sum(.5 * ((s - mub) * (s - mub) / (sd * sd) + logvarb))
        return x, ss_prior, ss_posterior



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
        recon_batch, ss_prior, ss_posterior = self.forw(self.mu, self.logvar)
        recon_loss, kl = self.loss_V(recon_batch, data, self.mu, self.logvar)
        loss = recon_loss.item() + ss_prior.item() + ss_posterior.item()
        if (type == 'train' or opt=='mu'):
            loss.backward()
            optim.step()

        return recon_batch.item(), recon_loss.item(), loss.item()

    def update_MU_LOGVAR(self,mu):
        self.MU = torch.nn.Parameter(torch.mean(mu, dim=0))
        self.LOGVAR = torch.nn.Parameter(torch.log(torch.var(mu, dim=0)))

    def run_epoch(self, train,  epoch,num_mu_iter,MU, LOGVAR,PI, type='test',fout=None):
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
            #t1 = time.time()
            for it in range(num_mu_iter):
                self.compute_loss_and_grad(data, type,self.optimizer_s,opt='mu')
            #print('mu time', time.time() - t1)
            mu[j:j + batch_size] = self.mu.data
            logvar[j:j + batch_size] = self.logvar.data
            # Update decoder and encoder parameters.
            #t1 = time.time()
            recon_batch, recon_loss, loss = self.compute_loss_and_grad(data,type,self.optimizer)
            #print('par time', time.time() - t1)
            tr_recon_loss += recon_loss
            tr_full_loss += loss
        if (fout is  None):
            print('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}'.format(type,
                    epoch, tr_recon_loss/len(tr), tr_full_loss/len(tr)))
        else:
            fout.write('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}\n'.format(type,
                    epoch, tr_recon_loss/len(tr), tr_full_loss/len(tr)))
        return mu,logvar, PI


    def recon(self,input,num_mu_iter=10):

        num_inp=input.shape[0]
        self.setup_id(num_inp)
        mu, logvar, _ =self.initialize_mus(input,True)
        data=input.to(self.dv)
        self.update_s(mu, logvar)
        for it in range(num_mu_iter):
            self.compute_loss_and_grad(data, type, self.optimizer_s, opt='mu')
        recon_batch, recon_loss, loss = self.compute_loss_and_grad(data, 'test', self.optimizer)

        return recon_batch

    def sample_from_z_prior(self,theta=None,clust=None):
        self.setup_id(self.bsz)
        s = torch.randn(self.bsz, self.s_dim).to(self.dv)
        if self.MM:
            s=s*torch.exp(self.LOGVAR/2)+self.MU
        theta=theta.to(self.dv)
        if (theta is not None and self.u_dim>0):
            s[:,0:self.u_dim]=theta

        x=self.decoder_and_trans(s)

        return x


