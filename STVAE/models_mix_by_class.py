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

        self.n_mix_perclass=np.int32(self.n_mix/self.n_class)

        self.mu_lr = args.mu_lr
        self.eyy = torch.eye(self.n_mix).to(self.dv)
        self.lamda1=args.lamda1


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
                            data_d=data.detach()
                            self.compute_loss_and_grad(data_d, data_in, None, 'test', self.optimizer_s, opt='mu',rng=rng)
                    ss_mu = self.mu.reshape(-1, self.n_mix_perclass, self.s_dim).transpose(0, 1)
                    pi = torch.softmax(self.pi, dim=1)
                    lpi=torch.log(pi)
                    with torch.no_grad():
                        recon_batch = self.decoder_and_trans(ss_mu, rng)
                        b=self.mixed_loss_pre(recon_batch, data)
                        B = torch.sum(pi * b, dim=1)
                    BB += [B]
                    KD += [self.dens_apply(self.mu, self.logvar, lpi, pi)[1]]
            else:
                with torch.no_grad():
                    s_mu, s_var, pi = self.encoder_mix(data)
                    ss_mu = s_mu.reshape(-1, self.n_mix, self.s_dim).transpose(0,1)
                    recon_batch = self.decoder_and_trans(ss_mu)
                    b = self.mixed_loss_pre(recon_batch, data)
                    b = b.reshape(-1,self.n_class,self.n_mix_perclass)
                    s_mu = s_mu.reshape(-1, self.n_class, self.n_mix_perclass * self.s_dim)
                    s_var = s_var.reshape(-1, self.n_class, self.n_mix_perclass * self.s_dim)
                    tpi=pi.reshape(-1,self.n_class,self.n_mix_perclass)
                    lpi=torch.log(tpi)

                    for c in range(self.n_class):
                        KD += [self.dens_apply(s_mu[:,c,:], s_var[:,c,:], lpi[:,c,:], tpi[:,c,:])[1]]
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
            acc_temp = acc/(j+len(data))
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
                    self.compute_loss_and_grad(inp,input, None, 'test', self.optimizer_s, opt='mu',rng=rng)
                s_mu = self.mu.reshape(-1,self.n_mix_perclass,self.s_dim).transpose(0,1)
                pi = torch.softmax(self.pi, dim=1)
        else:
            s_mu, s_var, pi = self.encoder_mix(inp)
            s_mu = s_mu.reshape(-1, self.n_class, self.n_mix_perclass*self.s_dim).transpose(0,1)
            s_mu = s_mu[cl].reshape(-1,self.n_mix_perclass,self.s_dim).transpose(0,1)
            pi = pi.reshape(-1,self.n_class,self.n_mix_perclass).transpose(0,1)
            pi  = pi[cl]

        recon_batch = self.decoder_and_trans(s_mu,rng)
        if (self.feats and not self.feats_back):
            rec_b=[]
            for rc in recon_batch:
                rec_b+=[self.conv.bkwd(rc.reshape(-1, self.feats, self.conv.x_hf, self.conv.x_hf))]
            recon_batch=torch.stack(rec_b,dim=0)
        recon_batch = recon_batch.transpose(0, 1)
        ii = torch.argmax(pi, dim=1)
        jj = torch.arange(0,num_inp,dtype=torch.int64).to(self.dv)
        kk = ii+jj*self.n_mix_perclass
        recon=recon_batch.reshape(self.n_mix_perclass*num_inp,-1)
        rr=recon[kk]
        if (self.feats and not self.feats_back):
            rrm=torch.min(rr,dim=1)[0].unsqueeze(dim=1)
            rrM=torch.max(rr,dim=1)[0].unsqueeze(dim=1)
            rr=(rr-rrm)/(rrM-rrm)

        return rr






