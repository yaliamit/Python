import torch
from torch import nn, optim
import numpy as np
import models_mix_by_class
from models_opt_mix import STVAE_OPT_mix
from models_mix_by_class import dens_apply
import contextlib
@contextlib.contextmanager
def dummy_context_mgr():
    yield None

class STVAE_OPT_mix_by_class(models_mix_by_class.STVAE_mix_by_class):

    def __init__(self, x_h, x_w, device, args):
        super(STVAE_OPT_mix_by_class, self).__init__(x_h, x_w, device, args)

        self.n_class=args.n_class
        self.n_mix_perclass=np.int32(self.n_mix/self.n_class)

        self.mu_lr = args.mu_lr
        self.eyy = torch.eye(self.n_mix).to(self.dv)
        if (args.optimizer == 'Adam'):
            self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        elif (args.optimizer == 'Adadelta'):
            self.optimizer = optim.Adadelta(self.parameters())

    def update_s(self, mu, logvar, pi, mu_lr, wd=0):

        self.mu = torch.autograd.Variable(mu.to(self.dv), requires_grad=True)
        self.logvar = torch.autograd.Variable(logvar.to(self.dv), requires_grad=True)
        self.pi = torch.autograd.Variable(pi.to(self.dv), requires_grad=True)
        self.optimizer_s = optim.Adam([self.mu, self.logvar, self.pi], mu_lr, weight_decay=wd)

    def forward(self,data,targ):

       pit=torch.softmax(self.pi, dim=1)
       return self.get_loss(data,targ, self.mu, self.logvar, pit)



    def compute_loss_and_grad(self,data, targ, type, optim, opt='par'):


        optim.zero_grad()
        
        rc,tot=self.forward(data,targ)


        loss = rc + tot
        if (type == 'train' or opt=='mu'):
            loss.backward()
            self.optimizer.step()
        rcs = rc.item()
        ls = loss.item()

        return rcs, ls


    def run_epoch(self, train, epoch,num_mu_iter, MU, LOGVAR,PI, type='test',fout=None):

        if (type=='train'):
            self.train()
        tr_recon_loss = 0;tr_full_loss = 0
        ii = np.arange(0, train[0].shape[0], 1)
        # if (type=='train'):
        #   np.random.shuffle(ii)
        tr = train[0][ii].transpose(0, 3, 1, 2)
        y = np.argmax(train[1][ii],axis=1)
        mu = MU #[ii]
        logvar = LOGVAR #[ii]
        pi = PI #[ii]
        batch_size = self.bsz
        for j in np.arange(0, len(y), self.bsz):
            data = torch.from_numpy(tr[j:j + self.bsz]).float().to(self.dv)
            target = torch.from_numpy(y[j:j + self.bsz]).float().to(self.dv)
            mulr=self.mu_lr[0]
            if (epoch>200):
                mulr=self.mu_lr[1]
            self.update_s(mu[j:j + batch_size, :], logvar[j:j + batch_size, :], pi[j:j + batch_size], mulr)
            for it in range(num_mu_iter):
                self.compute_loss_and_grad(data, target, type, self.optimizer_s, opt='mu')
            with torch.no_grad() if (type != 'train') else dummy_context_mgr():
                recon_loss, loss = self.compute_loss_and_grad(data, target, type, self.optimizer)
            mu[j:j + batch_size] = self.mu.data
            logvar[j:j + batch_size] = self.logvar.data
            pi[j:j + batch_size] = self.pi.data
            del self.mu, self.logvar, self.pi
            tr_recon_loss += recon_loss
            tr_full_loss += loss


        fout.write('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}\n'.format(type,
        epoch, tr_recon_loss / len(tr), tr_full_loss/len(tr)))

        return mu, logvar, pi


def get_scheduler(args,model):
    scheduler=None
    if args.wd:
        l2 = lambda epoch: pow((1.-1. * epoch/args.nepoch),0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda=l2)

    return scheduler

