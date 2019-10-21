import torch
from torch import nn, optim
import numpy as np
import models_mix

import contextlib

@contextlib.contextmanager
def dummy_context_mgr():
    yield None

class STVAE_OPT_mix(models_mix.STVAE_mix):

    def __init__(self, x_h, x_w, device, args):
        super(STVAE_OPT_mix, self).__init__(x_h, x_w, device, args)

        self.mu_lr=args.mu_lr
        self.eyy=torch.eye(self.n_mix).to(self.dv)
        if (args.optimizer=='Adam'):
            self.optimizer=optim.Adam(self.parameters(),lr=args.lr)
        elif (args.optimizer=='Adadelta'):
            self.optimizer = optim.Adadelta(self.parameters())




    def update_s(self,mu,logvar,pi,epoch):
        mu_lr=self.mu_lr[0]
        if epoch>200:
            mu_lr=self.mu_lr[1]
        self.mu=torch.autograd.Variable(mu.to(self.dv), requires_grad=True)
        self.logvar = torch.autograd.Variable(logvar.to(self.dv), requires_grad=True)
        self.pi = torch.autograd.Variable(pi.to(self.dv), requires_grad=True)
        self.optimizer_s = optim.Adam([self.mu, self.logvar,self.pi], mu_lr)

    def forward(self,data,opt):

        pit=torch.softmax(self.pi, dim=1)
        return self.get_loss(data,self.mu, self.logvar, pit) #recon_loss, tot


    def compute_loss_and_grad(self,data, type, optim, opt='par'):

        optim.zero_grad()

        recon_loss, tot= self.forward(data,opt)

        loss = recon_loss + tot #prior + post


        if (type == 'train' or opt=='mu'):
            loss.backward()
            optim.step()

        ls=loss.item()
        rcs=recon_loss.item()

        return rcs, ls

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
            self.update_s(mu[j:j+batch_size, :], logvar[j:j+batch_size, :], pi[j:j+batch_size],epoch)
            for it in range(num_mu_iter):
               self.compute_loss_and_grad(data, type,self.optimizer_s,opt='mu')
            with torch.no_grad() if (type !='train') else dummy_context_mgr():
                recon_loss, loss = self.compute_loss_and_grad(data,type,self.optimizer)
            mu[j:j + batch_size] = self.mu.data
            logvar[j:j + batch_size] = self.logvar.data
            pi[j:j + batch_size] = self.pi.data
            del self.mu, self.logvar, self.pi
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
        self.update_s(mu, logvar, pi,0)
        for it in range(num_mu_iter):
            self.compute_loss_and_grad(data, type, self.optimizer_s, opt='mu')
        ii = torch.argmax(self.pi, dim=1)
        jj = torch.arange(0, num_inp, dtype=torch.int64).to(self.dv)
        kk = ii + jj * self.n_mix
        s_mu = (self.mu).view(-1, self.n_mix, self.s_dim)
        recon_batch=self.decoder_and_trans(s_mu)
        #recon_loss, loss, recon_batch = self.compute_loss_and_grad(data, 'test', self.optimizer)
        recon = recon_batch.reshape(self.n_mix * num_inp, -1)
        rr = recon[kk]
        return rr
    
    

