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
    def __init__(self,h_dim,z_dim, u_dim, n_mix, type):
        super(fromNorm_mix,self).__init__()
        self.z2h=[]
        self.n_mix=n_mix
        self.z_dim=z_dim
        self.h_dim=h_dim
        self.u_dim=u_dim
        self.type=type
        self.z2h=nn.ModuleList([nn.Linear(z_dim, h_dim) for i in range(n_mix)])
        self.z2z=nn.ModuleList([nn.Linear(z_dim, z_dim) for i in range(n_mix)])
        if (type == 'tvae'):
            self.u2u = nn.ModuleList([nn.Linear(u_dim, u_dim) for i in range(n_mix)])

    def forward(self,z,u):

        h=[]
        for i in range(self.n_mix):
            h=h+[self.z2h[i](self.z2z[i](z[:,i,:]))]
            if (self.type=='tvae'):
                v=torch.zeros_like(u)
                v[:,i,:]=self.u2u[i](u[:,i,:])
            else:
                v=u
        hh=torch.stack(h,dim=0).transpose(0,1)
        hh=F.relu(hh)
        return hh, v




# Each correlated normal coming out of fromNorm_mix goes through same network to produce an image these get mixed.
class decoder_mix(nn.Module):
    def __init__(self,x_dim,h_dim,n_mix,num_layers):
        super(decoder_mix,self).__init__()
        self.x_dim=x_dim
        self.n_mix=n_mix
        self.num_layers=num_layers
        if (num_layers==1):
            self.h2hd = nn.Linear(h_dim, h_dim)
        self.h2x = nn.Linear(h_dim, x_dim)

    def forward(self,input):
            h=input
            if (self.num_layers==1):
                hh=[]
                for i in range(self.n_mix):
                    hh=hh+[self.h2hd(h[:,i,:])]
                h=torch.stack(hh,dim=0).transpose(0,1)
            x=[]
            for i in range(self.n_mix):
                x=x+[self.h2x(h[:,i,:])]
            xx=torch.stack(x,dim=0).transpose(0,1)
            xx=torch.sigmoid(xx)
            return(xx)


class STVAE_mix(models.STVAE):

    def __init__(self, x_h, x_w, device, args):
        super(STVAE_mix, self).__init__(x_h, x_w, device, args)


        self.n_mix = args.n_mix
        if (not args.OPT):
            self.toNorm_mix=toNorm_mix(self.h_dim, self.s_dim, self.n_mix)
        self.fromNorm_mix=fromNorm_mix(self.h_dim, self.z_dim,self.u_dim,self.n_mix, self.type)
        if (not args.OPT):
            self.encoder_mix = encoder_mix(self.x_dim, self.h_dim, self.num_hlayers)
        self.decoder_mix=decoder_mix(self.x_dim,self.h_dim,self.n_mix,self.num_hlayers)

        self.rho = nn.Parameter(torch.zeros(self.n_mix))

        if (args.optimizer=='Adam'):
            self.optimizer=optim.Adam(self.parameters(),lr=args.lr)
        elif (args.optimizer=='Adadelta'):
            self.optimizer = optim.Adadelta(self.parameters())

    def forward_encoder(self, inputs):
        h=F.relu(self.encoder_mix.x2h(inputs))
        if (self.num_hlayers==1):
            h=F.relu(self.encoder_mix.h2he(h))
        s_mu=self.toNorm_mix.h2smu(h)
        s_logvar=F.threshold(self.toNorm_mix.h2svar(h),-6,-6)
        hm=torch.matmul(h,self.toNorm_mix.h2pi).clamp(-10.,10.)
        pi = torch.softmax(hm,dim=1)
        return s_mu, s_logvar, pi

    def decoder_and_trans(self,s):

        u = s.narrow(len(s.shape)-1,0,self.u_dim)
        z = s.narrow(len(s.shape)-1,self.u_dim,self.z_dim)
        # Create image
        h,u = self.fromNorm_mix.forward(z,u)

        x = self.decoder_mix.forward(h)
        # Transform

        if (self.u_dim>0):
           xt = []
           for i in range(self.n_mix):
                xt=xt+[self.apply_trans(x[:,i,:],u[:,i,:]).squeeze()]
           x=torch.stack(xt,dim=0).transpose(0,1).view(-1,self.n_mix,self.x_dim)
        xx = torch.clamp(x, 1e-6, 1 - 1e-6)
        return xx


    def sample(self, mu, logvar, dim):
        eps = torch.randn(mu.shape[0],dim).to(self.dv)
        return mu + torch.exp(logvar/2) * eps

    def dens_apply(self,s,s_mu,s_logvar,pi):
        lpi=torch.log(pi.squeeze())
        s_mu = s_mu.view(-1, self.n_mix, self.s_dim)
        s_logvar = s_logvar.view(-1, self.n_mix, self.s_dim)
        sd=torch.exp(s_logvar/2)
        var=sd*sd
        f=[]
        for i in range(self.n_mix):
            ss=[]
            si=s[:,i,:]
            # Apply the mixture model to the samples from each of the mixture components.
            for j in range(self.n_mix):
                sss=si-s_mu[:,j,:]
                ss=ss+[-.5*((sss*sss)/var[:,j,:]+s_logvar[:,j,:])]
            ss=torch.stack(ss,dim=0).transpose(0,1)
            ss=torch.sum(ss,dim=2)
            u=torch.logsumexp(ss+lpi,dim=1)
            f = f+[u]
        f=torch.stack(f,dim=0).transpose(0,1)
        posterior=torch.sum(torch.bmm(pi,f[:,:,None]))
        # Sum along last coordinate to get negative log density of each component.
        pr=torch.sum((s*s),dim=2)/2
        # Substract log-prior
        pr=pr-self.rho+torch.logsumexp(self.rho,0)
        prior=torch.sum(torch.bmm(pi,pr[:,:,None])) #+10*torch.sum(self.rho*self.rho)

        return prior, posterior

    def mixed_loss(self,x,data,lpi):
        b = []

        for i in range(self.n_mix):
            a = F.binary_cross_entropy(x[:, i, :].squeeze().view(-1, self.x_dim), data.view(-1, self.x_dim),
                                       reduction='none')
            a = lpi[:, i] + torch.sum(a, dim=1)
            b = b + [a]
        b = torch.stack(b).transpose(0, 1)

        recloss = torch.sum(torch.logsumexp(b, dim=1))
        return recloss, b

    def forward(self, inputs):

        s_mu, s_logvar, pi = self.forward_encoder(inputs.view(-1, self.x_dim))

        if (self.type is not 'ae'):
            s = self.sample(s_mu, s_logvar, self.s_dim*self.n_mix)
        else:
            s=s_mu
        s=s.view(-1,self.n_mix,self.s_dim)
        pit = pi.reshape(pi.shape[0], 1, pi.shape[1])
        # Apply linear map to entire sampled vector.
        x=self.decoder_and_trans(s)

        prior, post = self.dens_apply(s,s_mu,s_logvar,pit)
        recloss=self.mixed_loss(x,inputs,pi)
        return recloss, prior, post



    def compute_loss_and_grad(self,data,type):

        if (type == 'train'):
            self.optimizer.zero_grad()

        recloss, prior, post = self.forward(data)

        loss = recloss + prior + post

        if (type == 'train'):
            loss.backward()
            self.optimizer.step()

        return recloss, loss

    def run_epoch(self, train, epoch,num, MU, LOGVAR,PI, type='test',fout=None):

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

        return MU, LOGVAR, PI

    def recon(self,input,num_mu_iter=None):

        num_inp=input.shape[0]
        self.setup_id(num_inp)
        inp = input.to(self.dv)
        s_mu, s_var, pi = self.forward_encoder(inp.view(-1, self.x_dim))
        s_mu = s_mu.view(-1, self.n_mix, self.s_dim)
        ii = torch.argmax(pi, dim=1)
        jj = torch.arange(0,num_inp,dtype=torch.int64).to(self.dv)
        kk = ii+jj*self.n_mix
        recon_batch = self.decoder_and_trans(s_mu)
        recon=recon_batch.reshape(self.n_mix*num_inp,-1)
        rr=recon[kk]

        return rr


    def sample_from_z_prior(self,theta=None):
        self.eval()
        self.setup_id(self.bsz)
        ee=torch.eye(self.n_mix).to(self.dv)
        rho_dist=torch.exp(self.rho-torch.logsumexp(self.rho,dim=0))
        ii=torch.multinomial(rho_dist,self.bsz,replacement=True)
        s = torch.randn(self.bsz, self.s_dim*self.n_mix).to(self.dv)
        s = s.view(-1, self.n_mix, self.s_dim)
        if (theta is not None and self.u_dim>0):
            theta = theta.to(self.dv)
            for i in range(self.n_mix):
                s[:,i,0:self.u_dim]=theta
        x=self.decoder_and_trans(s)
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

