import torch
import torch.nn.functional as F
from torch import nn, optim
from tps import TPSGridGen
import numpy as np

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
    def __init__(self,x_dim,h_dim,s_dim,u_dim,num_layers):
        super(decoder,self).__init__()
        if (num_layers==1):
            self.h2hd = nn.Linear(h_dim, h_dim)
        self.h2x = nn.Linear(h_dim, x_dim)
        if (self.type == 'tvae'):
            self.u2u = nn.Linear(u_dim, u_dim, bias=False)
            # self.z2z = nn.Linear(self.z_dim, self.z_dim)
        elif (self.type == 'stvae'):
            self.s2s = nn.Linear(s_dim, s_dim)


class STVAE(nn.Module):

    def __init__(self, x_h, x_w, device, args):
        super(STVAE, self).__init__()

        self.x_dim = x_h * x_w # height * width
        self.h = x_h
        self.w = x_w
        self.h_dim = args.hdim # hidden layer
        self.s_dim = args.sdim # generic latent variable
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
                idty = torch.cat((torch.eye(2),torch.zeros(2).unsqueeze(1)),dim=1)
            elif self.tf == 'tps':
                self.u_dim = 18 # 2 * 3 ** 2
                self.gridGen = TPSGridGen(out_h=self.h,out_w=self.w,device=self.dv)
                px = self.gridGen.PX.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
                py = self.gridGen.PY.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
                idty = torch.cat((px,py))
            self.id = idty.expand((self.bsz,) + idty.size()).to(self.dv)
        else:
            self.u_dim=0


        self.z_dim = self.s_dim-self.u_dim


        self.s2s=None
        self.u2u=None

        self.toNorm=toNorm(self.h_dim,self.s_dim)
        self.fromNorm =fromNorm(self.h_dim, self.s_dim)
        self.encoder=encoder(self.x_dim,self.h_dim,self.num_hlayers)
        self.decoder=decoder(self.x_dim,self.h_dim,self.s_dim,self.u_dim,self.num_hlayers)

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
        print('s_dim',self.s_dim,'u_dim',self.u_dim,'z_dim',self.z_dim,self.type)

    def forward_encoder(self, inputs):
        h=F.relu(self.encoder.x2h(inputs))
        if (self.num_hlayers==1):
            h=F.relu(self.encoder.h2he(h))
        s_mu=self.toNorm.h2smu(h)
        #s_prevar=self.toNorm.h2svar(h)
        s_var=F.threshold(self.toNorm.h2svar(h),-6,-6)

        #if (self.u_dim>0 and self.type=='tvae'):
        #    s_var=torch.cat([F.threshold(s_prevar.narrow(1,0,self.u_dim), -6, -6), s_prevar.narrow(1,self.u_dim,self.z_dim)],dim=1)
        #    s_mu=F.tanh(s_mu)
        #elif (self.type=='stvae'):
        #    s_var = F.threshold(s_prevar, -6, -6)
        #else:
        #s_var=s_prevar
        return s_mu, s_var


    def forward_decoder(self, z):
        h=F.relu(self.fromNorm.z2h(z))
        if (self.num_hlayers==1):
            h=F.relu(self.decoder.h2hd(h))
        x=torch.sigmoid(self.decoder.h2x(h))
        x = torch.clamp(x, 1e-6, 1 - 1e-6)
        return x

    def full_decoder(self,s):

        if (self.type=='tvae'): # Apply map separately to each component - transformation and z.
            u = s.narrow(1,0,self.u_dim)
            u = self.u2u(u)
            z = s.narrow(1,self.u_dim,self.z_dim)
            #self.z2z(z)
        else:
            if (self.type == 'stvae'):
                s = self.s2s(s)
            z = s.narrow(1, self.u_dim, self.z_dim)
            u = s.narrow(1, 0, self.u_dim)
        # Create image
        x = self.forward_decoder(z)
        # Transform
        if (self.u_dim>0):
            x=self.apply_trans(x,u)

        return x

    def apply_trans(self,x,u):
        # Apply transformation
        if 'tvae' in self.type:
            #u = F.tanh(u)
            #u = self.u2u(u)
            # Apply linear only to dedicated transformation part of sampled vector.
            if self.tf == 'aff':
                self.theta = u.view(-1, 2, 3) + self.id
                grid = F.affine_grid(self.theta, x.view(-1,self.h,self.w).unsqueeze(1).size())
            elif self.tf=='tps':
                self.theta = u + self.id
                grid = self.gridGen(self.theta)
            x = F.grid_sample(x.view(-1,self.h,self.w).unsqueeze(1), grid, padding_mode='border')
        x = x.clamp(1e-6, 1-1e-6)
        return x

    def sample(self, mu, logvar, dim):
        eps = torch.randn(self.bsz, dim).to(self.dv)
        return mu + torch.exp(logvar/2) * eps


    def forward(self, inputs):
        s_mu, s_var = self.forward_encoder(inputs.view(-1, self.x_dim))
        if (self.type is not 'ae'):
            s = self.sample(s_mu, s_var, self.s_dim)
        else:
            s=s_mu
        # Apply linear map to entire sampled vector.
        x=self.full_decoder(s)

        return x, s_mu, s_var

    def loss_V(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.squeeze().view(-1, self.x_dim), x.view(-1, self.x_dim), reduction='sum')
        KLD1 = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar))  # z
        return BCE, KLD1


    def run_epoch(self, train, epoch,type='test'):
        if (type=='train'):
            self.train()
        else:
            self.eval()
        self.train()
        tr_recon_loss = 0
        tr_full_loss=0
        numt=train[0].shape[0]
        numt= numt//self.bsz * self.bsz
        ii = np.arange(0, numt, 1)
        if (type=='train'):
            np.random.shuffle(ii)
        tr = train[0][ii]
        y = train[1][ii]
        batch_size = self.bsz
        for j in np.arange(0, len(y), batch_size):
            data = torch.from_numpy(tr[j:j + batch_size]).float()
            target = torch.from_numpy(y[j:j + batch_size]).float()
            data = data.to(self.dv)
            target = target.to(self.dv)
            if (type=='train'):
                self.optimizer.zero_grad()
            recon_batch, smu, slogvar = self(data)
            recon_loss, kl = self.loss_V(recon_batch, data, smu, slogvar)
            loss = recon_loss + kl
            tr_recon_loss += recon_loss
            tr_full_loss += loss
            if (type=='train'):
                loss.backward()
                self.optimizer.step()

        print('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}'.format(type,
            epoch, tr_recon_loss / len(tr), tr_full_loss/len(tr)))



    def sample_from_z_prior(self,theta=None):
        self.eval()
        s = torch.randn(self.bsz, self.s_dim).to(self.dv)
        if (theta is not None and self.u_dim>0):
            theta = theta.to(self.dv)
            s[:,0:self.u_dim]=theta

        x=self.full_decoder(s)

        return x
