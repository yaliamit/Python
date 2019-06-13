import torch
import torch.nn.functional as F
from torch import nn, optim
from tps import TPSGridGen
import numpy as np
import pylab as py

class STVAE_OPT(nn.Module):

    def __init__(self, x_h, x_w, device, args):
        super(STVAE_OPT, self).__init__()

        self.x_dim = x_h * x_w # height * width
        self.h = x_h
        self.w = x_w
        self.h_dim = args.hdim # hidden layer
        self.s_dim = args.sdim # generic latent variable
        self.bsz = args.mb_size
        self.num_hlayers=args.num_hlayers
        self.dv=device

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

        #self.update.to(self.dv)
        """
        encoder: two fc layers
        """

        self.tf = args.transformation
        self.type=args.type
        if 'tvae' in self.type:
            if self.tf == 'aff':
                self.u_dim = 6
                self.idty = torch.cat((torch.eye(2),torch.zeros(2).unsqueeze(1)),dim=1)
            elif self.tf == 'tps':
                self.u_dim = 18 # 2 * 3 ** 2
                self.gridGen = TPSGridGen(out_h=self.h,out_w=self.w,device=self.dv)
                px = self.gridGen.PX.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
                py = self.gridGen.PY.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
                self.idty = torch.cat((px,py))
            self.id = self.idty.expand((self.bsz,) + self.idty.size()).to(self.dv)
        else:
            self.u_dim=0

        self.mu_lr=args.mu_lr
        #if 'tvae' in self.type:
        self.mu_lr=torch.full([self.s_dim],args.mu_lr).to(self.dv)
        if 'tvae' in self.type:
            self.mu_lr[0:self.u_dim]*=.1
        self.z_dim = self.s_dim-self.u_dim

        self.s2s=None
        self.u2u=None

        if (self.num_hlayers==1):
            self.h2hd=nn.Linear(self.h_dim, self.h_dim)

        self.h2x = nn.Linear(self.h_dim, self.x_dim)
        if (self.type=='tvae'):
            self.u2u = nn.Linear(self.u_dim, self.u_dim)
        elif (self.type=='stvae' ):
            self.s2s = nn.Linear(self.s_dim, self.s_dim)

        self.z2h = nn.Linear(self.z_dim, self.h_dim)

        if (args.optimizer=='Adam'):
            self.optimizer=optim.Adam(self.parameters(),lr=args.lr)
        elif (args.optimizer=='Adadelta'):
            self.optimizer = optim.Adadelta(self.parameters())
        print('s_dim',self.s_dim,'u_dim',self.u_dim,'z_dim',self.z_dim,self.type)


    def apply_trans(self,x,u):
            # Apply transformation
            # u = F.tanh(u)
            # u = self.u2u(u)
            # Apply linear only to dedicated transformation part of sampled vector.
            if self.tf == 'aff':
                self.theta = u.view(-1, 2, 3) + self.id
                grid = F.affine_grid(self.theta, x.view(-1, self.h, self.w).unsqueeze(1).size())
            elif self.tf == 'tps':
                self.theta = u + self.id
                grid = self.gridGen(self.theta)
            x = F.grid_sample(x.view(-1, self.h, self.w).unsqueeze(1), grid, padding_mode='border')
            x = x.clamp(1e-6, 1 - 1e-6)
            return x

    def sample(self, mu, logvar, dim):
        eps = torch.randn(mu.shape[0], dim).to(self.dv)
        return mu + torch.exp(logvar/2) * eps

    def forward_decoder(self, z):
        h=F.relu(self.z2h(z))
        if (self.num_hlayers==1):
            h=F.relu(self.h2hd(h))
        x=torch.sigmoid(self.h2x(h))
        return x

    def full_decoder(self,s):
        # Apply linear map to entire sampled vector.
        if (self.type == 'tvae'):  # Apply map separately to each component - transformation and z.
            u = s.narrow(1, 0, self.u_dim)
            u = self.u2u(u)
            z = s.narrow(1, self.u_dim, self.z_dim)
            # self.z2z(z)
        else:
            if (self.type == 'stvae'):
                s = self.s2s(s)
                #s[:,0:self.u_dim]=s[:,0:self.u_dim]/5.
            z = s.narrow(1, self.u_dim, self.z_dim)
            u = s.narrow(1, 0, self.u_dim)
        # Create image
        x = self.forward_decoder(z)
        if (self.u_dim>0):
            x=self.apply_trans(x,u)

        return x


    def forw(self, inputs,mub,logvarb):

        #if (self.type is not 'ae'):
        #    s = self.sample(mub, logvarb, self.s_dim)
        #else:
        s=mub
        x=self.full_decoder(s)
        return x



    def loss_V(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.squeeze().view(-1, self.x_dim), x.view(-1, self.x_dim), reduction='sum')
        if self.MM:
            KLD1 = 0.5*torch.sum((mu-self.MU)*(mu-self.MU)/torch.exp(self.LOGVAR)+self.LOGVAR)
        else:
            KLD1 = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar))  # z
        return BCE, KLD1

    def compute_loss(self,data, mub, logvarb, type):

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
            #if (muit>1 and loss/oldloss>1.1):
            #    break
            #else:
            #    oldloss=loss
            dd = torch.autograd.grad(loss, [mub])
            mub,=self.sgdloc(dd,[mub])



            muit+=1
            #print(muit, loss)
        #self.updates['t_prev']=0
        #print('mub',torch.mean(mub),torch.std(mub))
        #print('logvarb',torch.mean(logvarb),torch.std(logvarb))
        return mub, logvarb, loss, recon_loss

    def run_epoch(self, train, MU, LOGVAR, epoch,num_mu_iter,type='test'):
        self.train()
        tr_recon_loss = 0
        tr_full_loss=0
        numt=train[0].shape[0]
        numt= numt//self.bsz * self.bsz
        ii = np.arange(0, numt, 1)
        #if (type=='train'):
        #    np.random.shuffle(ii)
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

            mub, logvarb, loss, recon_loss=self.iterate_mu_logvar(data,mub,logvarb,num_mu_iter)
            recon_batch, recon_loss, loss = self.compute_loss(data, mub, logvarb, type)

            mu[j:j + batch_size] = mub.cpu().detach().numpy()
            logvar[j:j + batch_size] = logvarb.cpu().detach().numpy()

            tr_recon_loss += recon_loss
            tr_full_loss += loss
        print('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}'.format(type,
            epoch, tr_recon_loss / len(tr), tr_full_loss/len(tr)))
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

        x=self.full_decoder(s)

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