import torch
import torch.nn.functional as F
from torch import nn, optim
from tps import TPSGridGen
import numpy as np
from scipy.misc import imsave
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
        self.fromNorm =fromNorm(self.h_dim, self.z_dim)
        self.encoder=encoder(self.x_dim,self.h_dim,self.num_hlayers)
        self.decoder=decoder(self.x_dim,self.h_dim,self.s_dim,self.u_dim,self.num_hlayers,self.type)

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
                grid = F.affine_grid(self.theta, x.view(-1,self.h,self.w).unsqueeze(1).size())
            elif self.tf=='tps':
                self.theta = u + self.id
                grid = self.gridGen(self.theta)
            x = F.grid_sample(x.view(-1,self.h,self.w).unsqueeze(1), grid, padding_mode='border')

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
        x=self.decoder_and_trans(s)

        return x, s_mu, s_var

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

    def run_epoch(self, train, epoch,num, MU, LOGVAR, type='test'):

        if (type=='train'):
            self.train()
        else:
            self.eval()
        self.train()
        tr_recon_loss = 0
        tr_full_loss=0
        numt = train[0].shape[0] // self.bsz * self.bsz
        ii = np.arange(0, numt, 1)
        if (type=='train'):
            np.random.shuffle(ii)
        tr = train[0][ii].transpose(0, 3, 1, 2)
        y = train[1][ii]

        for j in np.arange(0, len(y), self.bsz):
            data = torch.from_numpy(tr[j:j + self.bsz]).float().to(self.dv)
            target = torch.from_numpy(y[j:j + self.bsz]).float().to(self.dv)
            recon_loss, loss=self.compute_loss_and_grad(data,type)
            tr_recon_loss += recon_loss
            tr_full_loss += loss
        print('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}'.format(type,
            epoch, tr_recon_loss / len(tr), tr_full_loss/len(tr)))

        return MU, LOGVAR

    def sample_from_z_prior(self,theta=None):
        self.eval()
        s = torch.randn(self.bsz, self.s_dim).to(self.dv)
        if (theta is not None and self.u_dim>0):
            theta = theta.to(self.dv)
            s[:,0:self.u_dim]=theta

        x=self.decoder_and_trans(s)

        return x

def show_sampled_images(model,opt_pre,mm_pre):
    theta = torch.zeros(model.bsz, 6)
    X=model.sample_from_z_prior(theta)
    XX=X.cpu().detach().numpy()
    mat = []
    #py.figure(figsize=(20,20))
    t=0
    for i in range(10):
        line = []
        for j in range(10):
            line += [XX[t].reshape((28,28))]
            t+=1
        mat+=[np.concatenate(line,axis=0)]
        #py.subplot(10,10,i+1)
        #py.imshow(1.-XX[i].reshape((28,28)),cmap='gray')
        #py.axis('off')
    manifold = np.concatenate(mat, axis=1)

    manifold = 1. - manifold[np.newaxis, :]
    print(manifold.shape)

    img = np.concatenate([manifold, manifold, manifold], axis=0)
    img = img.transpose(1, 2, 0)
    imsave('_Images/'+opt_pre+model.type+'_'+str(model.num_hlayers)+mm_pre+'.png', img)
    #py.savefig()
    print("hello")

def get_scheduler(args,model):
    scheduler=None
    if args.wd:
        l2 = lambda epoch: pow((1.-1. * epoch/args.nepoch),0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda=l2)

    return scheduler

