import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import models
import time
from models_mix import STVAE_mix

import contextlib
@contextlib.contextmanager
def dummy_context_mgr():
    yield None



class STVAE_mix_by_class(STVAE_mix):

    def __init__(self, x_h, x_w, device, args):
        super(STVAE_mix_by_class, self).__init__(x_h, x_w, device, args)

        self.n_class=args.n_class
        self.n_mix_perclass=np.int32(self.n_mix/self.n_class)

        self.mu_lr = args.mu_lr
        self.eyy = torch.eye(self.n_mix).to(self.dv)
        #self.lamda1=args.lamda1


    def run_epoch_classify(self, train, d_type, fout=None, num_mu_iter=None, conf_thresh=0):


        self.eval()
        if self.opt:
            mu, logvar, ppi = self.initialize_mus(train[0], True)
            mu = mu.reshape(-1, self.n_class, self.n_mix_perclass * self.s_dim).transpose(0, 1)
            logvar = logvar.reshape(-1, self.n_class, self.n_mix_perclass * self.s_dim).transpose(0, 1)
            ppi = ppi.reshape(-1, self.n_class, self.n_mix_perclass).transpose(0, 1)

        ii = np.arange(0, train[0].shape[0], 1)
        tr = train[0][ii].transpose(0, 3, 1, 2)
        y = np.argmax(train[1][ii],axis=1)
        acc=0
        accb=0
        DF=[]; RY=[]
        for j in np.arange(0, len(y), self.bsz):
            KD = []
            BB = []
            fout.write('Batch '+str(j)+'\n')
            fout.flush()
            data_in = torch.from_numpy(tr[j:j + self.bsz]).float().to(self.dv)
            data = self.preprocess(data_in)
            if (len(data)<self.bsz):
                self.setup_id(len(data))
            if self.opt:
                for c in range(self.n_class):
                    #t1=time.time()
                    rng = range(c * self.n_mix_perclass, (c + 1) * self.n_mix_perclass)
                    self.update_s(mu[c][j:j + self.bsz], logvar[c][j:j + self.bsz], ppi[c][j:j + self.bsz], self.mu_lr[0])
                    for it in range(num_mu_iter):
                            self.compute_loss_and_grad(data, data_in, None, 'test', self.optimizer_s, opt='mu',rng=rng)
                    ss_mu = self.mu.reshape(-1, self.n_mix_perclass, self.s_dim).transpose(0, 1)
                    pi = torch.softmax(self.pi, dim=1)
                    lpi=torch.log(pi)
                    recon_batch = self.decoder_and_trans(ss_mu, rng)
                    b=self.mixed_loss_pre(recon_batch, data)
                    B = torch.sum(pi * b, dim=1)
                    BB += [B]
                    kd=self.dens_apply(self.mu, self.logvar, lpi, pi)[1]
                    KD += [kd]
            else:

                s_mu, s_var, pi = self.encoder_mix(data.view(-1, self.x_dim))
                ss_mu = s_mu.view(-1, self.n_mix, self.s_dim).transpose(0,1)
                recon_batch = self.decoder_and_trans(ss_mu)
                b = self.mixed_loss_pre(recon_batch, data)
                b = b.reshape(-1,self.n_class,self.n_mix_perclass)
                s_mu = s_mu.reshape(-1, self.n_class, self.n_mix_perclass * self.s_dim)
                s_var = s_var.reshape(-1, self.n_class, self.n_mix_perclass * self.s_dim)
                tpi=pi.reshape(-1,self.n_class,self.n_mix_perclass)
                lpi=torch.log(tpi)

                for c in range(self.n_class):
                    kd= self.dens_apply(s_mu[:,c,:], s_var[:,c,:], lpi[:,c,:], tpi[:,c,:])[1]
                    KD += [kd]
                    tpic=tpi[:,c,:]/torch.sum(tpi[:,c,:],dim=1).unsqueeze(1)
                    BB += [torch.sum(tpic*b[:,c,:],dim=1)]
            KD=torch.stack(KD,dim=1)
            BB=torch.stack(BB, dim=1)
            rr = BB + KD
            vy, ry = torch.min(rr, 1)
            ry = np.int32(ry.detach().cpu().numpy())
            RY+=[ry]
            rr=rr.detach().cpu().numpy()
            ii=np.argsort(rr,axis=1)
            DF+=[np.diff(np.take_along_axis(rr, ii[:, 0:2], axis=1), axis=1)]
            acc += np.sum(np.equal(ry, y[j:j + self.bsz]))
            acc_temp = acc/(j+self.bsz)
            fout.write('====> Epoch {}: Accuracy: {:.4f}\n'.format(d_type, acc_temp))
            fout.flush()
            #accb += np.sum(np.equal(by, y[j:j + self.bsz]))
        RY=np.concatenate(RY)
        DF=np.concatenate(DF,axis=0)
        iip = DF[:,0]>=conf_thresh
        iid = np.logical_not(iip)
        cl_rate=np.sum(np.equal(RY[iip],y[iip]))
        acc/=len(tr)
        fout.write('====> Epoch {}: Accuracy: {:.4f}\n'.format(d_type,acc))
        return(iid,RY,cl_rate,acc)

    def recon(self,input,num_mu_iter,cl):

        if self.opt:
            mu, logvar, ppi = self.initialize_mus(input, True)
            mu=mu.reshape(-1,self.n_class,self.n_mix_perclass*self.s_dim).transpose(0,1)
            logvar=logvar.reshape(-1,self.n_class,self.n_mix_perclass*self.s_dim).transpose(0,1)
            ppi=ppi.reshape(-1,self.n_class,self.n_mix_perclass).transpose(0,1)

        num_inp=input.shape[0]
        self.setup_id(num_inp)
        input = input.to(self.dv)
        inp = self.preprocess(input)
        c = cl
        rng = range(c * self.n_mix_perclass, (c + 1) * self.n_mix_perclass)

        #print('Class ' + str(c) + '\n')
        if self.opt:
                self.update_s(mu[c], logvar[c], ppi[c], self.mu_lr[0])
                for it in range(num_mu_iter):
                    self.compute_loss_and_grad(inp, input, None, 'test', self.optimizer_s, opt='mu',rng=rng)
                s_mu = self.mu.reshape(-1,self.n_mix_perclass,self.s_dim).transpose(0,1)
                s_var = self.logvar.reshape(-1,self.n_mix_perclass,self.s_dim).transpose(0,1)
                pi = torch.softmax(self.pi, dim=1)
        else:
            s_mu, s_var, pi = self.encoder_mix(inp.view(-1, self.x_dim))
            s_mu = s_mu.view(-1, self.n_class, self.n_mix_perclass*self.s_dim).transpose(0,1)
            s_mu = s_mu[cl].reshape(-1,self.n_mix_perclass,self.s_dim).transpose(0,1)
            s_var = s_var.view(-1, self.n_class, self.n_mix_perclass * self.s_dim).transpose(0, 1)
            s_var = s_var[cl].reshape(-1,self.n_mix_perclass,self.s_dim).transpose(0,1)
            pi = pi.view(-1,self.n_class,self.n_mix_perclass).transpose(0,1)
            pi  = pi[cl]

        recon_batch = self.decoder_and_trans(s_mu,rng)
        recon_batch = recon_batch.transpose(0, 1)
        ii = torch.argmax(pi, dim=1)
        jj = torch.arange(0,num_inp,dtype=torch.int64).to(self.dv)
        kk = ii+jj*self.n_mix_perclass
        recon=recon_batch.reshape(self.n_mix_perclass*num_inp,-1)
        rr=recon[kk]

        return rr


    def sample_from_z_prior(self,theta=None, clust=None):
        self.eval()
        ee=torch.eye(self.n_mix).to(self.dv)
        rho_dist=torch.exp(self.rho-torch.logsumexp(self.rho,dim=0))
        if (clust is not None):
            ii=clust*torch.ones(self.bsz, dtype=torch.int64).to(self.dv)
        else:
            ii=torch.multinomial(rho_dist,self.bsz,replacement=True)
        self.setup_id(self.bsz)
        s = torch.randn(self.bsz, self.s_dim*self.n_mix).to(self.dv)
        s = s.view(-1, self.n_mix, self.s_dim)
        if (theta is not None and self.u_dim>0):
            theta = theta.to(self.dv)
            for i in range(self.n_mix):
                s[:,i,0:self.u_dim]=theta
        s=s.transpose(0,1)
        x=self.decoder_and_trans(s)
        x=x.transpose(0,1)
        jj = torch.arange(0, self.bsz, dtype=torch.int64).to(self.dv)
        kk = ii + jj * self.n_mix
        recon = x.reshape(self.n_mix * self.bsz, -1)
        rr = recon[kk]

        return rr



def get_scheduler(args,model):
    scheduler=None
    if args.wd:
        l2 = lambda epoch: pow((1.-1. * epoch/args.nepoch),0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda=l2)

    return scheduler

