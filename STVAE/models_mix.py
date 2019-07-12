import torch
import torch.nn.functional as F
from torch import nn, optim
from tps import TPSGridGen
import numpy as np
import models

class encoder_mix(nn.Module):
    def __init__(self,x_dim,h_dim,num_layers):
        super(encoder_mix,self).__init__()
        if (num_layers==1):
            self.h2he = nn.Linear(h_dim, h_dim)
        self.x2h = nn.Linear(x_dim, h_dim)

# Create i.i.d normals for each mixture component and a logit for weights
class toNorm_mix(nn.Module):
    def __init__(self,h_dim,s_dim, n_mix):
        super(toNorm_mix,self).__init__()
        self.h2smu = nn.Linear(h_dim, s_dim*n_mix)
        self.h2svar = nn.Linear(h_dim, s_dim*n_mix)
        self.h2pi = nn.Parameter(torch.zeros(h_dim,n_mix))

# Each set of s_dim normals gets multiplied by its own matrix to correlate
class fromNorm_mix(nn.Module):
    def __init__(self,h_dim,z_dim, u_dim, n_mix, type, dv):
        super(fromNorm_mix,self).__init__()
        self.z2h=[]
        self.n_mix=n_mix
        self.z_dim=z_dim
        self.h_dim=h_dim
        self.u_dim=u_dim
        self.type=type
        self.dv=dv
        self.z2h=nn.ModuleList([nn.Linear(z_dim, h_dim) for i in range(n_mix)])
        self.z2z=nn.ModuleList([nn.Linear(z_dim, z_dim) for i in range(n_mix)])
        if (type == 'tvae'):
            self.u2u = nn.ModuleList([nn.Linear(u_dim, u_dim) for i in range(n_mix)])

    def forward(self,z,u):

        #z=z.view(-1,self.n_mix,self.z_dim)
        #if (self.type=='tvae'):
        #   u=u.view(-1,self.n_mix,self.u_dim)
        h=torch.zeros(z.shape[0],self.n_mix,self.h_dim)

        for i in range(self.n_mix):
            h[:,i,:]=self.z2h[i](self.z2z[i](z[:,i,:])).to(self.dv)
            if (self.type=='tvae'):
                v=torch.zeros_like(u)
                v[:,i,:]=self.u2u[i](u[:,i,:])
            else:
                v=u
        h=F.relu(h)
        return h, v




# Each correlated normal coming out of fromNorm_mix goes through same network to produce an image these get mixed.
class decoder_mix(nn.Module):
    def __init__(self,x_dim,h_dim,n_mix,num_layers,dv):
        super(decoder_mix,self).__init__()
        self.n_mix=n_mix
        self.x_dim=x_dim
        self.num_layers=num_layers
        self.dv=dv
        if (num_layers==1):
            self.h2hd = nn.Linear(h_dim, h_dim)
        self.h2x = nn.Linear(h_dim, x_dim)

    def forward(self,input):
            h=input
            if (self.num_layers==1):
                for i in range(self.n_mix):
                    h[:,i,:]=self.h2hd(h[:,i,:])
            x=torch.zeros(h.shape[0],self.n_mix,self.x_dim).to(self.dv)
            for i in range(self.n_mix):
                x[:,i,:]=self.h2x(h[:,i,:])
            x=torch.sigmoid(x)
            return(x)


class STVAE_mix(models.STVAE):

    def __init__(self, x_h, x_w, device, args):
        super(STVAE_mix, self).__init__(x_h, x_w, device, args)


        self.n_mix = args.n_mix
        self.toNorm_mix=toNorm_mix(self.h_dim, self.s_dim, self.n_mix)
        self.fromNorm_mix=fromNorm_mix(self.h_dim, self.z_dim,self.u_dim,self.n_mix, self.type, self.dv)
        self.encoder_mix = encoder_mix(self.x_dim, self.h_dim, self.num_hlayers)
        self.decoder_mix=decoder_mix(self.x_dim,self.h_dim,self.n_mix,self.num_hlayers,self.dv)

        self.rho = nn.Parameter(torch.zeros(self.n_mix))

        if (args.optimizer=='Adam'):
            self.optimizer=optim.Adam(self.parameters(),lr=args.lr)
        elif (args.optimizer=='Adadelta'):
            self.optimizer = optim.Adadelta(self.parameters())

    def forward_encoder(self, inputs):
        h=F.relu(self.encoder_mix.x2h(inputs))
        if (self.num_hlayers==1):
            h=F.relu(self.encoder.h2he(h))
        s_mu=self.toNorm_mix.h2smu(h)
        s_logvar=F.threshold(self.toNorm_mix.h2svar(h),-6,-6)
        hm=torch.matmul(h,self.toNorm_mix.h2pi)
        pi = torch.softmax(hm,dim=1)
        return s_mu, s_logvar, pi

    def decoder_and_trans(self,s, pi):

        #if (self.type=='tvae'): # Apply map separately to each component - transformation and z.
        u = s.narrow(len(s.shape)-1,0,self.u_dim)
        z = s.narrow(len(s.shape)-1,self.u_dim,self.z_dim)
        # Create image
        h,u = self.fromNorm_mix.forward(z,u)
        x = self.decoder_mix.forward(h)
        # Transform

        if (self.u_dim>0):
           xt = torch.zeros(x.shape[0],x.shape[1],self.h,self.w)
           for i in range(self.n_mix):
                xt[:,i,:,:]=self.apply_trans(x[:,i,:],u[:,i,:]).squeeze()
            #x[:,i,:]=xt.squeeze()
           xt=xt.view(-1,self.n_mix,self.x_dim)
        else:
            xt=x
        xx = torch.bmm(pi, xt).squeeze()
        xx = torch.clamp(xx, 1e-6, 1 - 1e-6)
        return xx


    def sample(self, mu, logvar, dim):
        eps = torch.randn(mu.shape[0],dim).to(self.dv)
        return mu + torch.exp(logvar/2) * eps

    def dens_apply(self,s,s_mu,s_logvar,pi):

        s_mu = s_mu.view(-1, self.n_mix, self.s_dim)
        s_logvar = s_logvar.view(-1, self.n_mix, self.s_dim)
        sd=torch.exp(s_logvar/2)
        var=sd*sd
        f=torch.zeros(self.bsz,self.n_mix)
        for i in range(self.n_mix):
            ss=torch.zeros_like(s)
            si=s[:,i,:]
            # Apply the mixture model to the samples from each of the mixture components.
            for j in range(self.n_mix):
                sss=si-s_mu[:,j,:]
                ss[:,j,:]=-.5*((sss*sss)/var[:,j,:]+s_logvar[:,j,:])
            ss=torch.sum(ss,dim=2)
            ss=torch.exp(ss)
            f[:,i] = torch.log(torch.bmm(pi, ss[:,:,None]).squeeze())
        posterior=torch.sum(torch.bmm(pi,f[:,:,None]))
        # Sum along last coordinate to get negative log density of each component.
        pr=torch.sum((s*s),dim=2)/2
        # Substract log-prior
        pr=pr-self.rho+torch.logsumexp(self.rho,0)
        prior=torch.sum(torch.bmm(pi,pr[:,:,None])) #+10*torch.sum(self.rho*self.rho)

        return prior, posterior

    def forward(self, inputs):
        prior=0; post=0;
        s_mu, s_logvar, pi = self.forward_encoder(inputs.view(-1, self.x_dim))

        if (self.type is not 'ae'):
            s = self.sample(s_mu, s_logvar, self.s_dim*self.n_mix)
        else:
            s=s_mu
        s=s.view(-1,self.n_mix,self.s_dim)
        pit = pi.reshape(pi.shape[0], 1, pi.shape[1])
        # Apply linear map to entire sampled vector.
        print(s.is_cuda, pit.is_cuda)

        x=self.decoder_and_trans(s,pit)
        prior, post = self.dens_apply(s,s_mu,s_logvar,pit)
        return x, prior, post



    def compute_loss_and_grad(self,data,type):

        if (type == 'train'):
            self.optimizer.zero_grad()

        recon_batch, prior, post = self.forward(data)
        recon_loss = F.binary_cross_entropy(recon_batch.squeeze().view(-1, self.x_dim), data.view(-1, self.x_dim), reduction='sum')

        loss = recon_loss + prior + post

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

        num_inp=input.shape[0]
        self.setup_id(num_inp)
        inp = input.to(self.dv)
        s_mu, s_var, pi = self.forward_encoder(inp.view(-1, self.x_dim))
        ii = torch.argmax(pi, dim=1)
        ee = torch.eye(self.n_mix)
        pia = ee[ii]
        pia=pia[:,None,:]
        s_mu = s_mu.view(-1, self.n_mix, self.s_dim)
        recon_batch = self.decoder_and_trans(s_mu, pia)

        return recon_batch


    def sample_from_z_prior(self,theta=None):
        self.eval()
        self.setup_id(self.bsz)
        ee=torch.eye(self.n_mix)
        rho_dist=torch.exp(self.rho-torch.logsumexp(self.rho,dim=0))
        kk=torch.multinomial(rho_dist,self.bsz,replacement=True)
        pi=ee[kk]
        pi=pi[:,None,:]
        s = torch.randn(self.bsz, self.s_dim*self.n_mix).to(self.dv)

        s = s.view(-1, self.n_mix, self.s_dim)
        if (theta is not None and self.u_dim>0):
            theta = theta.to(self.dv)
            for i in range(self.n_mix):
                s[:,i,0:self.u_dim]=theta
        x=self.decoder_and_trans(s,pi)

        return x



def get_scheduler(args,model):
    scheduler=None
    if args.wd:
        l2 = lambda epoch: pow((1.-1. * epoch/args.nepoch),0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda=l2)

    return scheduler

