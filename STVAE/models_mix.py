import torch
import torch.nn.functional as F
from torch import nn, optim
from tps import TPSGridGen
import numpy as np
import models


class toNorm_mix(nn.Module):
    def __init__(self,h_dim,s_dim, n_mix):
        super(toNorm_mix,self).__init__()
        self.h2smu = nn.Linear(h_dim, s_dim*n_mix)
        self.h2svar = nn.Linear(h_dim, s_dim*n_mix)
        self.h2pi = nn.Linear(h_dim,n_mix)

class fromNorm_mix(nn.Module):
    def __init__(self,h_dim,z_dim, n_mix):
        super(fromNorm_mix,self).__init__()
        self.z2h=[]
        self.n_mix=n_mix
        self.z_dim=z_dim
        for i in range(n_mix):
            self.z2h=self.z2h+[nn.Linear(z_dim, h_dim)]

    def forward(self,input):
        hh=[]
        z=input.view(-1,self.n_mix,self.z_dim)
        for i in range(self.n_mix):
            zi=z[:,i,:].squeeze()
            hh=hh+[self.z2h[i](zi)]
        h=torch.stack(hh)
        return h



class encoder(nn.Module):
    def __init__(self,x_dim,h_dim,num_layers):
        super(encoder,self).__init__()
        if (num_layers==1):
            self.h2he = nn.Linear(h_dim, h_dim)
        self.x2h = nn.Linear(x_dim, h_dim)

class decoder_mix(nn.Module):
    def __init__(self,x_dim,h_dim,s_dim,u_dim,n_mix,num_layers,type):
        super(decoder_mix,self).__init__()
        self.n_mix=n_mix
        self.num_layers=num_layers
        if (num_layers==1):
            self.h2hd = nn.Linear(h_dim, h_dim)
        self.h2x = nn.Linear(h_dim, x_dim)
        if (type == 'tvae'):
            self.u2u = nn.Linear(u_dim, u_dim, bias=False)
        elif (type == 'stvae'):
            self.s2s = nn.Linear(s_dim, s_dim)

    def forward(self,input):
            h=input
            if (self.num_layers==1):
                for i in range(self.n_mix):
                    h[i]=self.h2hd(h[i])
            xx=[]
            for i in range(self.n_mix):
                xx=xx+[self.h2x(h[i])]
            x=torch.stack(xx)
            return(x)


class STVAE_mix(models.STVAE):

    def __init__(self, x_h, x_w, device, args):
        super(STVAE_mix, self).__init__(x_h,x_w,device, args)

        self.x_dim = x_h * x_w # height * width
        self.h = x_h
        self.w = x_w
        self.h_dim = args.hdim # hidden layer
        self.s_dim = args.sdim # generic latent variable
        self.n_mix = args.n_mix
        self.bsz = args.mb_size
        self.num_hlayers=args.num_hlayers
        self.dv=device

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
            self.setup_id(self.bsz)
        else:
            self.u_dim=0


        self.z_dim = self.s_dim-self.u_dim


        self.s2s=None
        self.u2u=None
        if not args.OPT:
            self.toNorm_mix=toNorm_mix(self.h_dim,self.s_dim, self.n_mix)
        self.fromNorm_mix =fromNorm_mix(self.h_dim, self.z_dim,self.n_mix)
        if not args.OPT:
            self.encoder=encoder(self.x_dim,self.h_dim,self.num_hlayers)
        self.decoder_mix=decoder_mix(self.x_dim,self.h_dim,self.s_dim,self.u_dim,self.n_mix,self.num_hlayers,self.type)

        if (args.optimizer=='Adam'):
            self.optimizer=optim.Adam(self.parameters(),lr=args.lr)
        elif (args.optimizer=='Adadelta'):
            self.optimizer = optim.Adadelta(self.parameters())
        else:
            self.optimizer = optim.SGD([
                {'params':self.encoder.parameters()},
                {'params': self.decoder.parameters()},
                {'params':self.toNorm.parameters(),'lr':1e-5},
                {'params': self.fromNorm.parameters(), 'lr': 1e-5}
                ],lr=args.lr)
        args.fout.write('s_dim,'+str(self.s_dim)+', u_dim,'+str(self.u_dim)+', z_dim,' + str(self.z_dim)+','+self.type+'\n')


    def setup_id(self,size):
        if ('tvae' in self.type):
            self.id = self.idty.expand((size,) + self.idty.size()).to(self.dv)



    def initialize_mus(self,train,OPT=False):
        trMU=None
        trLOGVAR=None
        if (OPT and train[0] is not None):
            if (not self.MM):
                trMU = torch.zeros(train[0].shape[0], self.s_dim).to(self.dv)
            else:
                trMU = self.MU.repeat(train[0].shape[0], 1)
            trLOGVAR = torch.zeros(train[0].shape[0], self.s_dim).to(self.dv)

        return trMU, trLOGVAR

    def forward_encoder(self, inputs):
        h=F.relu(self.encoder.x2h(inputs))
        if (self.num_hlayers==1):
            h=F.relu(self.encoder.h2he(h))
        s_mu=self.toNorm_mix.h2smu(h)
        s_var=F.threshold(self.toNorm_mix.h2svar(h),-6,-6)
        pi = torch.softmax(self.toNorm_mix.h2pi(h),dim=1)
        return s_mu, s_var, pi


    def forward_decoder(self, z, pi):
        h=F.relu(self.fromNorm_mix.forward(z))
        xx=torch.sigmoid(self.decoder_mix.forward(h))
        x=torch.bmm(pi[:,None,:],xx.transpose(0,1)).squeeze()
        x = torch.clamp(x, 1e-6, 1 - 1e-6)
        return x

    def decoder_and_trans(self,s, pi):

        if (self.type=='tvae'): # Apply map separately to each component - transformation and z.
            u = s.narrow(1,0,self.u_dim)
            u = self.decoder.u2u(u)
            z = s.narrow(1,self.u_dim,self.z_dim)
        else:
            if (self.type == 'stvae'):
                s = self.decoder.s2s(s)
            z = s.narrow(1, self.u_dim*self.n_mix, self.z_dim*self.n_mix)
            u = s.narrow(1, 0, self.u_dim*self.n_mix)
        # Create image
        x = self.forward_decoder(z,pi)
        # Transform
        if (self.u_dim>0):
            x=self.apply_trans(x,u)
        return x

    def apply_trans(self,x,u):
        # Apply transformation
        if 'tvae' in self.type:
            # Apply linear only to dedicated transformation part of sampled vector.
            if self.tf == 'aff':
                self.theta = u.view(-1, 2, 3) + self.id
                grid = F.affine_grid(self.theta, x.view(-1,self.h,self.w).unsqueeze(1).size())
            elif self.tf=='tps':
                self.theta = u + self.id
                grid = self.gridGen(self.theta)
            x = F.grid_sample(x.view(-1,self.h,self.w).unsqueeze(1), grid, padding_mode='border')

        return x

    def sample(self, mu, logvar, dim):
        eps = torch.randn(mu.shape[0],dim).to(self.dv)
        #eps = torch.randn(self.bsz, dim).to(self.dv)
        return mu + torch.exp(logvar/2) * eps

    def dens_apply(self,s,s_mu,s_logvar,pi):
        sd=torch.exp(s_logvar/2)
        var=sd*sd
        f=torch.exp(-.5*(s-s_mu)*(s-s_mu)/var)/sd

    def forward(self, inputs):
        s_mu, s_logvar, pi = self.forward_encoder(inputs.view(-1, self.x_dim))
        if (self.type is not 'ae'):
            s = self.sample(s_mu, s_logvar, self.s_dim*self.n_mix)
        else:
            s=s_mu
        # Apply linear map to entire sampled vector.

        x=self.decoder_and_trans(s,pi)
        self.dens_apply(s,s_mu,s_logvar,pi)
        return x, s_mu, s_logvar, pi

    def loss_V(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.squeeze().view(-1, self.x_dim), x.view(-1, self.x_dim), reduction='sum')
        KLD1 = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar))  # z
        return BCE, KLD1

    def compute_loss_and_grad(self,data,type):

        if (type == 'train'):
            self.optimizer.zero_grad()

        recon_batch, smu, slogvar = self(data)
        recon_loss, kl = self.loss_V(recon_batch, data, smu, slogvar)
        loss = recon_loss + kl

        if (type == 'train'):
            loss.backward()
            self.optimizer.step()

        return recon_loss, loss

    def run_epoch(self, train, epoch,num, MU, LOGVAR, type='test',fout=None):

        if (type=='train'):
            self.train()
        tr_recon_loss = 0;tr_full_loss = 0
        ii = np.arange(0, train[0].shape[0], 1)
        # if (type=='train'):
        #   np.random.shuffle(ii)
        tr = train[0][ii].transpose(0, 3, 1, 2)
        y = train[1][ii]

        for j in np.arange(0, len(y), self.bsz):
            data = torch.from_numpy(tr[j:j + self.bsz]).float().to(self.dv)
            target = torch.from_numpy(y[j:j + self.bsz]).float().to(self.dv)
            recon_loss, loss=self.compute_loss_and_grad(data,type)
            tr_recon_loss += recon_loss
            tr_full_loss += loss

        fout.write('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}\n'.format(type,
        epoch, tr_recon_loss / len(tr), tr_full_loss/len(tr)))

        return MU, LOGVAR

    def recon(self,input,num_mu_iter=None):

        num_inp=input[0].shape[0]
        inp = torch.from_numpy(input[0].transpose(0, 3, 1, 2)).to(self.dv)
        self.setup_id(num_inp)
        s_mu, s_var = self.forward_encoder(inp.view(-1, self.x_dim))
        recon_batch = self.decoder_and_trans(s_mu)

        return recon_batch


    def sample_from_z_prior(self,theta=None):
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

