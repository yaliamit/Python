import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import models_mix
import models

class STVAE_OPT_mix(models_mix.STVAE_mix):

    def __init__(self, x_h, x_w, device, args):
        super(STVAE_OPT_mix, self).__init__(x_h, x_w, device, args)

        self.MM=args.MM
        if (self.MM):
            self.MU=nn.Parameter(torch.zeros(self.s_dim))  #, requires_grad=False)
            self.LOGVAR=nn.Parameter(torch.zeros(self.s_dim)) #, requires_grad=False)

        self.mu_lr=args.mu_lr
        self.s2s=None
        self.u2u=None

    def update_s(self,mu,logvar,pi):
        self.mu=torch.autograd.Variable(mu, requires_grad=True)
        self.logvar = torch.autograd.Variable(logvar, requires_grad=True)
        self.pi=torch.autograd.Variable(pi, requires_grad=True)
        self.optimizer_s = optim.Adam([self.mu, self.logvar, self.pi], lr=self.mu_lr)

    def forward(self,data):


        if (self.type is not 'ae' and not self.MM):
            s = self.sample(self.mu, self.logvar, self.s_dim*self.n_mix)
        else:
            s=self.mu

        s = s.view(-1, self.n_mix, self.s_dim)
        pii=torch.softmax(self.pi,dim=1)
        pit =pii.reshape(self.pi.shape[0], 1, self.pi.shape[1])
        # Apply linear map to entire sampled vector.

        x = self.decoder_and_trans(s)
        prior, post = self.dens_apply(s, self.mu, self.logvar, pit)
        recon_loss=self.mixed_loss(x,data,pii)
        return recon_loss, prior, post, x


    def compute_loss_and_grad(self,data, type, optim, opt='par'):

        if (type == 'train' or opt=='mu'):
            optim.zero_grad()

        recon_loss, prior, post,recon = self.forward(data)

        loss = recon_loss + prior + post

        if (type == 'train' or opt=='mu'):
            loss.backward()
            optim.step()

        return recon_loss, loss, recon

    def run_epoch(self, train,  epoch,num_mu_iter,MU, LOGVAR, PI, type='test',fout=None):
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
        pi=PI[ii]
        batch_size = self.bsz
        for j in np.arange(0, len(y), batch_size):

            data = torch.tensor(tr[j:j + batch_size]).float()
            data = data.to(self.dv)

            #target = torch.tensor(y[j:j + batch_size]).float()

            self.update_s(mu[j:j+batch_size, :], logvar[j:j+batch_size, :], pi[j:j+batch_size])
            for it in range(num_mu_iter):
               self.compute_loss_and_grad(data, type,self.optimizer_s,opt='mu')
            mu[j:j + batch_size] = self.mu.data
            logvar[j:j + batch_size] = self.logvar.data
            pi[j:j + batch_size]=self.pi.data
            recon_loss, loss, _ = self.compute_loss_and_grad(data,type,self.optimizer)
            tr_recon_loss += recon_loss
            tr_full_loss += loss
        if (fout is  None):
            print('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}'.format(type,
                    epoch, tr_recon_loss/len(tr), tr_full_loss/len(tr)))
        else:
            fout.write('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}\n'.format(type,
                    epoch, tr_recon_loss/len(tr), tr_full_loss/len(tr)))
        return mu,logvar, pi


    def recon(self,input,num_mu_iter=10):

        num_inp=input.shape[0]
        self.setup_id(num_inp)
        mu, logvar, pi=self.initialize_mus(input,True)
        data = input.to(self.dv)
        self.update_s(mu, logvar, pi)
        for it in range(num_mu_iter):
            self.compute_loss_and_grad(data, type, self.optimizer_s, opt='mu')
        ii = torch.argmax(self.pi, dim=1)
        jj = torch.arange(0, num_inp, dtype=torch.int64).to(self.dv)
        kk = ii + jj * self.n_mix
        recon_loss, loss, recon_batch = self.compute_loss_and_grad(data, 'test', self.optimizer)
        recon = recon_batch.reshape(self.n_mix * num_inp, -1)
        rr = recon[kk]
        return rr




