import torch
import torch.nn.functional as F
from torch import nn, optim
from tps import TPSGridGen
import numpy as np

import pylab as py

class toNorm(nn.Module):
    def __init__(self,h_dim,s_dim):
        super(toNorm,self).__init__()
        self.h2smu = nn.Linear(h_dim, s_dim)
        self.h2svar = nn.Linear(h_dim, s_dim)

class fromNorm(nn.Module):
    def __init__(self,h_dim,z_dim):
        super(fromNorm,self).__init__()
        self.z2h = nn.Linear(z_dim, h_dim)

class encoder(nn.Module):
    def __init__(self,x_dim,h_dim,num_layers):
        super(encoder,self).__init__()
        if (num_layers==1):
            self.h2he = nn.Linear(h_dim, h_dim)
        self.x2h = nn.Linear(x_dim, h_dim)

class decoder(nn.Module):
    def __init__(self,x_dim,h_dim,s_dim,u_dim,num_layers,type):
        super(decoder,self).__init__()
        if (num_layers==1):
            self.h2hd = nn.Linear(h_dim, h_dim)
        self.h2x = nn.Linear(h_dim, x_dim)
        if (type == 'tvae'):
            self.u2u = nn.Linear(u_dim, u_dim, bias=False)
        elif (type == 'stvae'):
            self.s2s = nn.Linear(s_dim, s_dim)


class STVAE(nn.Module):

    def __init__(self, sh, device, args):
        super(STVAE, self).__init__()
        x_h=sh[1]; x_w=sh[2]
        self.x_dim = x_h * x_w *sh[0] # height * width
        self.input_channels=sh[0]
        self.h = x_h
        self.w = x_w
        self.h_dim = args.hdim # hidden layer
        self.s_dim = args.sdim # generic latent variable
        self.bsz = args.mb_size
        self.n_mix=args.n_mix
        self.num_hlayers=args.num_hlayers
        self.dv=device
        self.tps_num=args.tps_num

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
                self.u_dim = self.tps_num*self.tps_num*2 # 2 * 3 ** 2
                self.gridGen = TPSGridGen(out_h=self.h,out_w=self.w,grid_size=self.tps_num,device=self.dv)
                px = self.gridGen.PX.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
                py = self.gridGen.PY.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
                self.idty = torch.cat((px,py))
            self.setup_id(self.bsz)
        else:
            self.u_dim=0


        self.z_dim = self.s_dim-self.u_dim


        self.s2s=None
        self.u2u=None
        if not args.OPT and args.n_mix==0:
            self.toNorm=toNorm(self.h_dim,self.s_dim)
        if (args.n_mix==0):
            self.fromNorm =fromNorm(self.h_dim, self.z_dim)
        if not args.OPT and args.n_mix==0:
            self.encoder=encoder(self.x_dim,self.h_dim,self.num_hlayers)
        if args.n_mix==0:
            self.decoder=decoder(self.x_dim,self.h_dim,self.s_dim,self.u_dim,self.num_hlayers,self.type)
            if (args.optimizer=='Adam'):
                self.optimizer=optim.Adam(self.parameters(),lr=args.lr)
            elif (args.optimizer=='Adadelta'):
                self.optimizer = optim.Adadelta(self.parameters())


    def setup_id(self,size):
        if ('tvae' in self.type):
            self.id = self.idty.expand((size,) + self.idty.size()).to(self.dv)



    def initialize_mus(self,train,OPT=None):
        trMU=None
        trLOGVAR=None
        trPI=None
        sdim=self.s_dim
        if self.n_mix>0:
            sdim=self.s_dim*self.n_mix
        if (train is not None):
            trMU = torch.zeros(train.shape[0], sdim) #.to(self.dv)
            trLOGVAR = torch.zeros(train.shape[0], sdim) #.to(self.dv)
            #EE = (torch.eye(self.n_mix) * 5. + torch.ones(self.n_mix)).to(self.dv)
            #ii=torch.randint(0,self.n_mix,[train.shape[0]])
            #trPI=EE[ii]
            trPI=torch.zeros(train.shape[0], self.n_mix) #.to(self.dv)
        return trMU, trLOGVAR, trPI

    def forward_encoder(self, inputs):
        h=F.relu(self.encoder.x2h(inputs))
        if (self.num_hlayers==1):
            h=F.relu(self.encoder.h2he(h))
        s_mu=self.toNorm.h2smu(h)
        s_var=F.threshold(self.toNorm.h2svar(h),-6,-6)
        return s_mu, s_var


    def forward_decoder(self, z):
        h=F.relu(self.fromNorm.z2h(z))
        if (self.num_hlayers==1):
            h=F.relu(self.decoder.h2hd(h))
        x=torch.sigmoid(self.decoder.h2x(h))
        x = torch.clamp(x, 1e-6, 1 - 1e-6)
        return x

    def decoder_and_trans(self,s):

        if (self.type=='tvae'): # Apply map separately to each component - transformation and z.
            u = s.narrow(1,0,self.u_dim)
            u = self.decoder.u2u(u)
            z = s.narrow(1,self.u_dim,self.z_dim)
        else:
            if (self.type == 'stvae'):
                s = self.decoder.s2s(s)
            z = s.narrow(1, self.u_dim, self.z_dim)
            u = s.narrow(1, 0, self.u_dim)
        # Create image
        x = self.forward_decoder(z)
        # Transform
        if (self.u_dim>0):
            x=self.apply_trans(x,u)
        x = x.clamp(1e-6, 1 - 1e-6)
        return x

    def apply_trans(self,x,u):
        # Apply transformation
        if 'tvae' in self.type:
            # Apply linear only to dedicated transformation part of sampled vector.
            if self.tf == 'aff':
                self.theta = u.view(-1, 2, 3) + self.id
                grid = F.affine_grid(self.theta, x.view(-1,self.input_channels,self.h,self.w).size(),align_corners=True)
            elif self.tf=='tps':
                self.theta = u + self.id
                grid = self.gridGen(self.theta)
            x = F.grid_sample(x.view(-1,self.input_channels,self.h,self.w), grid, padding_mode='border',align_corners=True)

        return x

    def sample(self, mu, logvar, dim):
        eps = torch.randn(mu.shape[0],dim).to(self.dv)
        #eps = torch.randn(self.bsz, dim).to(self.dv)
        return mu + torch.exp(logvar/2) * eps


    def forward(self, inputs):
        s_mu, s_logvar = self.forward_encoder(inputs.view(-1, self.x_dim))
        if (self.type is not 'ae'):
            s = self.sample(s_mu, s_logvar, self.s_dim)
        else:
            s=s_mu
        # Apply linear map to entire sampled vector.
        x=self.decoder_and_trans(s)
        ss_prior = torch.sum((s * s) / 2)
        sd=torch.exp(s_logvar*.5)
        ss_posterior = -torch.sum(.5*((s-s_mu)*(s-s_mu)/(sd*sd) + s_logvar))
        return x, s_mu, s_logvar, ss_prior, ss_posterior

    def loss_V(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.squeeze().view(-1, self.x_dim), x.view(-1, self.x_dim), reduction='sum')
        KLD1 = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar))  # z
        return BCE, KLD1

    def compute_loss_and_grad(self,data,type):

        if (type == 'train'):
            self.optimizer.zero_grad()

        recon_batch, smu, slogvar, ss_prior, ss_posterior = self(data)
        recon_loss, kl = self.loss_V(recon_batch, data, smu, slogvar)
        loss = recon_loss+ kl #ss_prior.item() + ss_posterior.item()

        if (type == 'train'):
            loss.backward()
            self.optimizer.step()

        return recon_loss.item(), loss.item()

    def run_epoch(self, train, epoch,num, MU, LOGVAR, PI, type='test',fout=None):

        if (type=='train'):
            self.train()
        tr_recon_loss = 0;tr_full_loss = 0
        ii = np.arange(0, train[0].shape[0], 1)
        # if (type=='train'):
        #   np.random.shuffle(ii)
        tr = train[0][ii].transpose(0, 3, 1, 2)
        y = train[1][ii]

        if epoch==100:
            for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/2

        for j in np.arange(0, len(y), self.bsz):
            data = torch.from_numpy(tr[j:j + self.bsz]).float().to(self.dv)
            target = torch.from_numpy(y[j:j + self.bsz]).float().to(self.dv)
            recon_loss, loss=self.compute_loss_and_grad(data,type)
            tr_recon_loss += recon_loss
            tr_full_loss += loss

        fout.write('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}\n'.format(type,
        epoch, tr_recon_loss / len(tr), tr_full_loss/len(tr)))

        return MU, LOGVAR, PI

    def recon(self,input,num_mu_iter=None):

        num_inp=input[0].shape[0]
        inp = input.to(self.dv)
        self.setup_id(num_inp)
        s_mu, s_var = self.forward_encoder(inp.view(-1, self.x_dim))
        recon_batch = self.decoder_and_trans(s_mu)

        return recon_batch


    def sample_from_z_prior(self,theta=None,clust=None):
        self.eval()
        self.setup_id(self.bsz)
        s = torch.randn(self.bsz, self.s_dim).to(self.dv)
        if (theta is not None and self.u_dim>0):
            theta = theta.to(self.dv)
            s[:,0:self.u_dim]=theta

        x=self.decoder_and_trans(s)

        return x



def get_scheduler(args,model):
    scheduler=None
    if args.wd:
        l2 = lambda epoch: pow((1.-1. * epoch/args.nepoch),0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda=l2)

    return scheduler

