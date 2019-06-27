import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np

import models

class STVAE_OPT(models.STVAE):

    def __init__(self, x_h, x_w, device, args):
        super(STVAE_OPT, self).__init__(x_h, x_w, device, args)

        # Stuff for Adam on optimization of mu's var's for each image
        self.beta1=torch.tensor(.5).to(self.dv)
        self.beta2=torch.tensor(.999).to(self.dv)
        self.updates={}
        self.updates['one']=torch.tensor(1.).to(self.dv)
        self.updates['epsilon']=torch.tensor(1e-8).to(self.dv)
        self.updates['t_prev']=torch.tensor(0.).to(self.dv)
        self.updates['lr']=torch.tensor(.01).to(self.dv)
        self.MM=args.MM
        if (self.MM):
            self.MU=nn.Parameter(torch.zeros(self.s_dim))
            self.LOGVAR=nn.Parameter(torch.zeros(self.s_dim))

        self.mu_lr=torch.full([self.s_dim],args.mu_lr).to(self.dv)
        if 'tvae' in self.type:
            self.mu_lr[0:self.u_dim]*=.1


        self.s2s=None
        self.u2u=None

        if (args.optimizer=='Adam'):
            self.optimizer=optim.Adam(self.parameters(),lr=args.lr)
        elif (args.optimizer=='Adadelta'):
            self.optimizer = optim.Adadelta(self.parameters())
        else:
            self.optimizer = optim.SGD(lr=args.lr)
        print('s_dim',self.s_dim,'u_dim',self.u_dim,'z_dim',self.z_dim,self.type)




    def forw(self, inputs,mub,logvarb):

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

    def compute_loss_and_grad(self,data, mub, logvarb, type):

        if (type == 'train'):
            self.optimizer.zero_grad()
        recon_batch = self.forw(data, mub, logvarb)
        recon_loss, kl = self.loss_V(recon_batch, data, mub, logvarb)
        loss = recon_loss + kl
        if (type == 'train'):
            loss.backward()
            self.optimizer.step()

        return recon_batch, recon_loss, loss

    def sgdloc(self, grads, params):

        for i in range(len(params)):
            params[i] = params[i] - self.mu_lr * grads[i]

        return params

    def iterate_mu_logvar(self, data, mub, logvarb, num_mu_iter):
        muit=0.
        oldloss=0.
        while  muit < num_mu_iter:
            recon_batch = self.forw(data, mub, logvarb)
            recon_loss, kl = self.loss_V(recon_batch, data, mub, logvarb)
            loss = recon_loss + kl
            if (self.MM):
                dd = torch.autograd.grad(loss, [mub])
                mub,=self.sgdloc(dd,[mub])
            else:
                dd = torch.autograd.grad(loss, [mub, logvarb])
                mub,logvarb=self.sgdloc(dd,[mub,logvarb])




            muit+=1
            #print(muit, loss)
        #self.updates['t_prev']=0
        #print('mub',torch.mean(mub),torch.std(mub))
        #print('logvarb',torch.mean(logvarb),torch.std(logvarb))
        return mub, logvarb, loss, recon_loss

    def run_epoch(self, train,  epoch,num_mu_iter,MU, LOGVAR,type='test',fout=None):
        self.train()
        tr_recon_loss = 0
        tr_full_loss=0
        numt= train[0].shape[0]//self.bsz * self.bsz
        ii = np.arange(0, numt, 1)
        if (type=='train'):
            np.random.shuffle(ii)
        tr =train[0][ii].transpose(0,3,1,2)
        y = train[1][ii]
        mu= MU[ii]
        logvar=LOGVAR[ii]
        batch_size = self.bsz
        for j in np.arange(0, len(y), batch_size):

            data = torch.tensor(tr[j:j + batch_size]).float()
            mub=torch.autograd.Variable(torch.tensor(mu[j:j+batch_size],requires_grad=True).float(),requires_grad=True)
            logvarb=torch.autograd.Variable(torch.tensor(logvar[j:j+batch_size],requires_grad=True).float(),requires_grad=True)

            target = torch.tensor(y[j:j + batch_size]).float()
            data = data.to(self.dv)
            mub = mub.to(self.dv)
            logvarb = logvarb.to(self.dv)


            for it in range(1):
                mub, logvarb, loss, recon_loss=self.iterate_mu_logvar(data,mub,logvarb,num_mu_iter)
                recon_batch, recon_loss, loss = self.compute_loss_and_grad(data, mub, logvarb, type)

            mu[j:j + batch_size] = mub.cpu().detach().numpy()
            logvar[j:j + batch_size] = logvarb.cpu().detach().numpy()

            tr_recon_loss += recon_loss
            tr_full_loss += loss
        if (fout is  None):
            print('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}'.format(type,
                    epoch, tr_recon_loss/len(tr), tr_full_loss/len(tr)))
        else:
            fout.write('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}\n'.format(type,
                    epoch, tr_recon_loss/len(tr), tr_full_loss/len(tr)))
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



    def adamloc(self, grads, params):

        if self.updates['t_prev']==0:
            self.updates['m_prev']=torch.zeros(params[0].shape[1]).to(self.dv)
            self.updates['v_prev']=torch.zeros(params[0].shape[1]).to(self.dv)

        t = self.updates['t_prev'] + 1
        a_t = self.updates['lr'] * torch.sqrt(self.updates['one'] - self.beta2 ** t) / (self.updates['one'] - self.beta1 ** t)
        for i in range(len(params)):
            g_t=grads[i]
            m_t = self.beta1 * self.updates['m_prev'] + (1. - self.beta1) * g_t
            v_t = self.beta2 * self.updates['v_prev'] + (1. - self.beta2) * g_t * g_t
            # STEPS.append(a_t/T.sqrt(v_t)+epsilon)
            step = a_t * m_t / (torch.sqrt(v_t) + self.updates['epsilon'])
            self.updates['m_prev'] = m_t
            self.updates['v_prev'] = v_t
            params[i] = params[i] - step

        self.updates['t_prev'] = t
        return params