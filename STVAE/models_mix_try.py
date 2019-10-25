import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import models
from Sep import encoder_mix_sep


# Create i.i.d normals for each mixture component and a logit for weights
class toNorm_mix(nn.Module):
    def __init__(self,model):
        super(toNorm_mix,self).__init__()
        self.h2smu = nn.Linear(model.h_dim, model.s_dim*model.n_mix)
        self.h2svar = nn.Linear(model.h_dim, model.s_dim*model.n_mix,bias=False)
        self.h2pi =  nn.Linear(model.h_dim,model.n_mix)

class encoder_mix(nn.Module):
    def __init__(self,model):
        super(encoder_mix,self).__init__()
        self.num_layers=model.num_hlayers
        self.n_mix=model.n_mix
        if (self.num_layers==1):
            self.h2he = nn.Linear(model.h_dim, model.h_dim)
        self.x2h = nn.Linear(model.x_dim, model.h_dim)
        self.x2hpi = nn.Linear(model.x_dim, model.h_dim)
        self.toNorm_mix=toNorm_mix(model)

    def forward(self,inputs):
        h = F.relu(self.x2h(inputs))
        hpi = F.relu(self.x2hpi(inputs))
        if (self.num_layers == 1):
            h = F.relu(self.h2he(h))
        s_mu = self.toNorm_mix.h2smu(h)
        s_logvar = F.threshold(self.toNorm_mix.h2svar(h), -6, -6)
        hm = self.toNorm_mix.h2pi(hpi).clamp(-10., 10.)
        pi = torch.softmax(hm, dim=1)
        return s_mu, s_logvar, pi


class diag(nn.Module):
    def __init__(self,dim):
        super(diag,self).__init__()

        self.dim=dim
        rim=(torch.rand(self.dim) - .5) / np.sqrt(self.dim)
        self.mu=nn.Parameter(rim)
        ris=(torch.rand(self.dim) - .5) / np.sqrt(self.dim)
        self.sigma=nn.Parameter(ris)

    def forward(self,z):

        u=z*self.sigma+self.mu
        return(u)

class bias(nn.Module):
    def __init__(self,dim):
        super(bias,self).__init__()

        self.dim=dim
        self.bias=nn.Parameter((torch.rand(self.dim) - .5) / np.sqrt(self.dim))

    def forward(self,z):
        return(self.bias.repeat(z.shape[0],1))

class ident(nn.Module):
    def __init__(self):
        super(ident,self).__init__()

    def forward(self,z):
        return(torch.ones(z.shape[0]))

# Each set of s_dim normals gets multiplied by its own matrix to correlate
class fromNorm_mix(nn.Module):
    def __init__(self,model):
        super(fromNorm_mix,self).__init__()
        self.z2h=[]
        self.n_mix=model.n_mix
        self.z_dim=model.z_dim
        self.h_dim=model.h_dim
        self.u_dim=model.u_dim
        self.diag=model.diag
        self.type=model.type
        self.h_dim_dec = model.h_dim_dec
        if (self.h_dim_dec is None):
            h_dim=self.h_dim
        else:
            h_dim=self.h_dim_dec
        if (self.z_dim>0):
                self.z2h=nn.ModuleList([nn.Linear(self.z_dim, h_dim) for i in range(self.n_mix)])
        else:
                self.z2h=nn.ModuleList([bias(h_dim) for i in range(self.n_mix)])
        if (self.z_dim > 0):
            if (not self.diag):
                self.z2z=nn.ModuleList([nn.Linear(self.z_dim, self.z_dim) for i in range(self.n_mix)])
            else:
                self.z2z=nn.ModuleList(diag(self.z_dim) for i in range(self.n_mix))
        else:
            self.z2z=nn.ModuleList(ident() for i in range(self.n_mix))
        if (self.type == 'tvae'):
            self.u2u = nn.ModuleList([nn.Linear(self.u_dim, self.u_dim, bias=False) for i in range(self.n_mix)])

    def forward(self,z,u):

        h=[]
        v=[]
        for i,(zz,vv) in enumerate(zip(z,u)):
            h=h+[self.z2h[i](self.z2z[i](zz))]
            if (self.type=='tvae'):
                v=v+[self.u2u[i](vv)]

        hh=torch.stack(h,dim=0) #.transpose(0,1)
        hh=F.relu(hh)
        return hh, v


# Each correlated normal coming out of fromNorm_mix goes through same network to produce an image these get mixed.
class decoder_mix(nn.Module):
    def __init__(self,model,args):
        super(decoder_mix,self).__init__()
        self.x_dim=model.x_dim
        self.n_mix=model.n_mix
        self.u_dim=model.u_dim
        self.z_dim=model.z_dim
        self.h_dim=model.h_dim
        self.h_dim_dec=args.hdim_dec
        self.num_layers=model.num_hlayers
        self.type=model.type
        self.diag=args.Diag
        if (self.num_layers==1):
            if self.h_dim_dec is None:
                self.h2hd = nn.Linear(self.h_dim, self.h_dim)
                self.h2x = nn.Linear(self.h_dim, self.x_dim)
            else:
                self.h2hd = nn.ModuleList([nn.Linear(self.h_dim_dec,self.h_dim) for i in range(self.n_mix)])
                self.h2x = nn.ModuleList([nn.Linear(self.h_dim, self.x_dim) for i in range(self.n_mix)])

        self.fromNorm_mix = fromNorm_mix(self)

    def forward(self,s):

            u = s.narrow(len(s.shape) - 1, 0, self.u_dim)
            z = s.narrow(len(s.shape) - 1, self.u_dim, self.z_dim)
            h, u = self.fromNorm_mix.forward(z, u)
            if (self.num_layers==1):
                hh=[]
                for i,h_ in enumerate(h):
                    if self.h_dim_dec is None:
                        hh=hh+[self.h2hd(h_)]
                    else:
                        hh=hh+[self.h2hd[i](h_)]
                h=torch.stack(hh,dim=0)
                h=F.relu(h)
            x=[]
            for i,h_ in enumerate(h):
                if self.h_dim_dec is None:
                    x=x+[self.h2x(h_)]
                else:
                    x=x+[self.h2x[i](h_)]
            xx=torch.stack(x,dim=0)
            xx=torch.sigmoid(xx)
            return(xx,u)


class STVAE_mix(models.STVAE):

    def __init__(self, x_h, x_w, device, args):
        super(STVAE_mix, self).__init__(x_h, x_w, device, args)


        self.n_mix = args.n_mix
        self.sep=args.sep
        self.num_hlayers=args.num_hlayers
        if (not args.OPT):
            if args.sep:
                self.encoder_mix = encoder_mix_sep(self)
            else:
                self.encoder_mix = encoder_mix(self)

        self.decoder_mix=decoder_mix(self,args)

        self.rho = nn.Parameter(torch.zeros(self.n_mix),requires_grad=False)

        if (args.optimizer=='Adam'):
            self.optimizer=optim.Adam(self.parameters(),lr=args.lr)
        elif (args.optimizer=='Adadelta'):
            self.optimizer = optim.Adadelta(self.parameters())


    def decoder_and_trans(self,s):

        x, u = self.decoder_mix.forward(s)
        # Transform
        if (self.u_dim>0):
           xt = []
           for xx,uu in zip(x,u):
                xt=xt+[self.apply_trans(xx,uu).squeeze()]
           x=torch.stack(xt,dim=0).view(self.n_mix,-1,self.x_dim)
        xx = torch.clamp(x, 1e-6, 1 - 1e-6)
        return xx


    def sample(self, mu, logvar, dim):
        eps = torch.randn(mu.shape[0],dim).to(self.dv)
        return mu + torch.exp(logvar/2) * eps

    def dens_apply(self,s_mu,s_logvar,lpi,pi,rho):
        n_mix=pi.shape[1]
        s_mu = s_mu.view(-1, n_mix, self.s_dim)
        s_logvar = s_logvar.view(-1, n_mix, self.s_dim)
        sd=torch.exp(s_logvar/2)
        var=sd*sd

        #ss=-.5*((s-s_mu)*(s-s_mu)/var+s_logvar)
        #ss=torch.sum(ss,dim=2)+lpi
        #posterior=torch.sum(pi*ss)
        # Sum along last coordinate to get negative log density of each component.
        KD_dens=-0.5 * torch.sum(1 + s_logvar - s_mu ** 2 - var, dim=2)
        KD_disc=lpi-rho+torch.logsumexp(rho,0)
        tot=torch.sum(pi*(KD_dens+KD_disc))
        #pr=-.5*torch.sum((s*s),dim=2)+self.rho-torch.logsumexp(self.rho,0)
        #prior=-torch.sum(pi*pr)
        #prior=0; posterior=0
        return tot #prior, posterior

    def mixed_loss_pre(self,x,data,n_mix):
        b = []

        for xx in x:
            a = F.binary_cross_entropy(xx, data.view(-1, self.x_dim),
                                       reduction='none')
            a = torch.sum(a, dim=1)
            b = b + [a]
        b = torch.stack(b).transpose(0, 1)
        return(b)

    def mixed_loss(self,x,data,pi):

        b=self.mixed_loss_pre(x,data, pi.shape[1])
        recloss = torch.sum(pi*b)
        return recloss #, b

    def get_loss(self,data,mu,logvar,pi):
        if (self.type is not 'ae'):
            s = self.sample(mu, logvar, self.s_dim*self.n_mix)
        else:
            s=mu
        s=s.view(-1,self.n_mix,self.s_dim).transpose(0,1)
        # Apply linear map to entire sampled vector.
        x=self.decoder_and_trans(s)
        lpi=torch.log(pi)

        tot = self.dens_apply(mu,logvar,lpi,pi,self.rho)
        recloss=self.mixed_loss(x,data,pi)
        return recloss, tot


    def forward(self, data):

        mu, logvar, pi = self.encoder_mix(data.view(-1, self.x_dim))

        return self.get_loss(data,mu,logvar,pi)

    def compute_loss_and_grad(self,data,type):

        self.optimizer.zero_grad()

        recloss, tot = self.forward(data)

        loss = recloss + tot #prior + post


        if (type == 'train'):
            loss.backward()
            self.optimizer.step()

        return recloss.item(), loss.item()

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
            #print(j)
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
        s_mu, s_var, pi = self.encoder_mix(inp.view(-1, self.x_dim))

        s_mu = s_mu.view(-1, self.n_mix, self.s_dim).transpose(0,1)
        ii = torch.argmax(pi, dim=1)
        jj = torch.arange(0,num_inp,dtype=torch.int64).to(self.dv)
        kk = ii+jj*self.n_mix
        recon_batch = self.decoder_and_trans(s_mu)
        recon=recon_batch.reshape(self.n_mix*num_inp,-1)
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
        s = torch.randn(self.bsz, self.s_dim*self.n_mix).to(self.dv)
        s = s.view(-1, self.n_mix, self.s_dim).transpose(0,1)
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

