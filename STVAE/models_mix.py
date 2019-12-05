import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import models
from Sep import encoder_mix_sep
from encoder_decoder import encoder_mix, decoder_mix

import contextlib
@contextlib.contextmanager
def dummy_context_mgr():
    yield None


class conv2(nn.Module):
    def __init__(self,x_h, x_w, args):

        super(conv2,self).__init__()
        pp = np.int32(np.floor(args.filts / 2))
        self.feats=args.feats
        self.x_dim = np.int32((x_h / args.pool) * (x_w / args.pool) * args.feats)
        self.x_h=x_h
        self.x_hf = np.int32(x_h / args.pool)

        self.conv = torch.nn.Conv2d(args.input_channels, args.feats, args.filts, stride=1, bias=False,
                                    padding=pp)
        self.deconv = torch.nn.ConvTranspose2d(args.feats, args.input_channels, args.filts, stride=args.pool,
                                               padding=pp, output_padding=1, bias=False)
        self.deconv.weight.data = self.conv.weight.data
        # self.orthogo()
        if (np.mod(args.pool, 2) == 1):
            pad = np.int32(args.pool / 2)
        else:
            pad = np.int32((args.pool - 1) / 2)
        self.pool = nn.MaxPool2d(args.pool, stride=args.pool_stride, padding=(pad, pad))

    def fwd(self,xx):
            xx=self.pool(self.conv(xx))
            return xx


    def bkwd(self,xx):
            xx = self.deconv(xx.reshape(-1, self.feats, self.x_hf, self.x_hf))
            return xx

class STVAE_mix(models.STVAE):

    def orthogo(self):
        #u, s, v = torch.svd(self.conv.weight.view(self.feats, self.filts * self.filts * self.input_channels))
        #self.conv.weight.data = v.transpose(0, 1).reshape(self.feats, self.input_channels, self.filts, self.filts)
        aa=self.conv.weight.view(self.feats, self.filts * self.filts * self.input_channels)
        q,r=torch.qr(aa.transpose(0,1))
        self.conv.weight.data=q.transpose(0,1).reshape(self.feats, self.input_channels, self.filts, self.filts)

    def __init__(self, x_h, x_w, device, args):
        super(STVAE_mix, self).__init__(x_h, x_w, device, args)

        self.lim=args.lim
        self.opt = args.OPT
        self.mu_lr = args.mu_lr
        self.n_mix = args.n_mix
        self.flag=True
        self.sep=args.sep
        self.n_parts=args.n_parts
        self.n_part_locs=args.n_part_locs
        self.part_dim=args.part_dim
        self.feats=args.feats
        self.feats_back=args.feats_back
        self.filts=args.filts
        self.x_h=x_h
        self.diag=args.Diag
        self.output_cont=args.output_cont
        self.h_dim_dec=args.hdim_dec
        self.n_class=args.n_class
        if self.n_parts:
            self.u_dim=self.n_parts*2
            self.s_dim=self.u_dim
        self.num_hlayers=args.num_hlayers



        if (self.feats>0):
            self.conv=conv2(x_h, x_w, args)
            self.x_dim=self.conv.x_dim

        if (not args.OPT):
            if args.sep:
                self.encoder_mix = encoder_mix_sep(self)
            else:
                self.encoder_mix = encoder_mix(self)

        self.decoder_mix=decoder_mix(self,args)



        self.rho = nn.Parameter(torch.zeros(self.n_mix),requires_grad=False)

        if (args.optimizer=='Adam'):
                ppd = []
                # ppd=self.decoder_mix.state_dict()
                for keys, vals in self.decoder_mix.named_parameters():  # state_dict().items():
                    if ('conv' not in keys):
                        ppd += [vals]
                PP = [{'params': ppd, 'lr': args.lr}]
                if (not self.opt):
                    ppe = []
                    for keys, vals in self.encoder_mix.named_parameters():
                        if ('conv' not in keys):
                            ppe += [vals]
                    PP += [{'params': ppe, 'lr': args.lr}]

                if (self.feats):  # and not self.feats_back):
                    PP += [{'params': self.conv.parameters(), 'lr': args.ortho_lr}]
                self.optimizer = optim.Adam(PP)
            #self.optimizer=optim.Adam(self.parameters(),lr=args.lr)
        elif (args.optimizer=='Adadelta'):
            self.optimizer = optim.Adadelta(self.parameters())



    def update_s(self,mu,logvar,pi,mu_lr,wd=0):
        # mu_lr=self.mu_lr[0]
        # if epoch>200:
        #     mu_lr=self.mu_lr[1]
        self.mu=torch.autograd.Variable(mu.to(self.dv), requires_grad=True)
        self.logvar = torch.autograd.Variable(logvar.to(self.dv), requires_grad=True)
        self.pi = torch.autograd.Variable(pi.to(self.dv), requires_grad=True)
        self.optimizer_s = optim.Adam([self.mu, self.logvar,self.pi], lr=mu_lr,weight_decay=wd)

    def update_s_parts(self,pi_parts,mu_lr,wd=0):
        self.pi_parts = torch.autograd.Variable(pi_parts.to(self.dv), requires_grad=True)
        self.optimizer_s = optim.Adam([self.mu, self.logvar, self.pi,self.pi_parts], lr=mu_lr, weight_decay=wd)

    def preprocess(self,data):


        with torch.no_grad() if self.flag else dummy_context_mgr():
            if (self.feats>0 and not self.feats_back):
                data = F.relu(self.conv.fwd(data))

        return data

    def decoder_and_trans(self,s, rng=None):

        n_mix=s.shape[0]
        x, u = self.decoder_mix.forward(s,rng)
        # Transform
        if (self.u_dim>0):
            if self.n_parts==0:
                xt = []
                for xx,uu in zip(x,u):
                    xt=xt+[self.apply_trans(xx,uu).squeeze()]

            x=torch.stack(xt,dim=0).reshape(n_mix,-1,self.x_dim)
        xx = torch.clamp(x, 1e-6, 1 - 1e-6)
        return xx


    def sample(self, mu, logvar, dim):
        eps = torch.randn(mu.shape[0],dim).to(self.dv)
        if (self.s_dim>1):
            z = mu + torch.exp(logvar/2) * eps
        else:
            z = torch.ones(mu.shape[0],dim).to(self.dv)
        return z

    def dens_apply(self,s_mu,s_logvar,lpi,pi):
        n_mix=pi.shape[1]
        s_mu = s_mu.reshape(-1, n_mix, self.s_dim)
        s_logvar = s_logvar.reshape(-1, n_mix, self.s_dim)
        sd=torch.exp(s_logvar/2)
        var=sd*sd

        # Sum along last coordinate to get negative log density of each component.
        KD_dens=-0.5 * torch.sum(1 + s_logvar - s_mu ** 2 - var, dim=2)
        KD_disc=lpi-torch.log(torch.tensor(n_mix,dtype=torch.float))
        KD = torch.sum(pi * (KD_dens + KD_disc), dim=1)
        tot=torch.sum(KD)

        return tot, KD

    def mixed_loss_pre(self,x,data):
        b = []

        if (not self.output_cont):
            for xx in x:
                a = F.binary_cross_entropy(xx, data.reshape(data.shape[0], -1),
                                           reduction='none')
                a = torch.sum(a, dim=1)
                b = b + [a]
        else:
            for xx in x:
                data=data.view(-1,self.x_dim)
                a=(data-xx)*(data-xx)
                #a = F.binary_cross_entropy(xx,data.view(-1, self.x_dim),
                #                           reduction='none')
                a = torch.sum(a, dim=1)
                b = b + [a]
        b = torch.stack(b).transpose(0, 1)
        return(b)

    def weighted_sum_of_likelihoods(self,lpi,b):
        return(-torch.logsumexp(lpi-b,dim=1))

    def mixed_loss(self,x,data,pi):

        b=self.mixed_loss_pre(x,data)
        recloss=torch.sum(pi*b)
        return recloss


    def get_loss(self,data,targ,mu,logvar,pi,rng=None):

        if (targ is not None):
            pi=pi.reshape(-1,self.n_class,self.n_mix_perclass)
            pis=torch.sum(pi,2)
            pi = pi/pis.unsqueeze(2)
        lpi = torch.log(pi)
        n_mix = self.n_mix
        if (targ is None and self.n_class > 0):
            n_mix = self.n_mix_perclass
        if (self.type is not 'ae'):
            s = self.sample(mu, logvar, self.s_dim*n_mix)
        else:
            s=mu
        s=s.reshape(-1,n_mix,self.s_dim).transpose(0,1)
        # Apply linear map to entire sampled vector.
        x=self.decoder_and_trans(s,rng)

        if (targ is not None):
            x=x.transpose(0,1)
            x=x.reshape(-1,self.n_class,self.n_mix_perclass,x.shape[-1])
            mu=mu.reshape(-1,self.n_class,self.n_mix_perclass*self.s_dim)
            logvar=logvar.reshape(-1,self.n_class,self.n_mix_perclass*self.s_dim)
            tot=0
            recloss=0
            if (type(targ) == torch.Tensor):
                for c in range(self.n_class):
                    ind = (targ == c)
                    tot += self.dens_apply(mu[ind,c,:],logvar[ind,c,:],lpi[ind,c,:],pi[ind,c,:])[0]
                    recloss+=self.mixed_loss(x[ind,c,:,:].transpose(0,1),data[ind],pi[ind,c,:])
            else:
                 tot += self.dens_apply(mu[:, targ, :], logvar[:, targ, :], lpi[:, targ, :], pi[:, targ, :])[0]
                 recloss += self.mixed_loss(x[:, targ, :, :].transpose(0, 1), data, pi[:, targ, :])
        else:
            tot = self.dens_apply(mu, logvar, lpi, pi)[0]
            recloss = self.mixed_loss(x, data, pi)
        return recloss, tot

    def encoder_and_loss(self, data,targ, rng):

        with torch.no_grad() if not self.flag else dummy_context_mgr():
            if (self.opt):
                pi = torch.softmax(self.pi, dim=1)
                logvar=self.logvar
                mu=self.mu
            else:
                mu, logvar, pi = self.encoder_mix(data)

        return self.get_loss(data,targ, mu,logvar,pi,rng)

    def compute_loss_and_grad(self,data,data_orig,targ,d_type,optim, opt='par', rng=None):

        optim.zero_grad()

        recloss, tot = self.encoder_and_loss(data,targ,rng)
        # if (self.feats and not opt=='mu' and not self.feats_back):
        #     #out=self.conv(self.ID)
        #     #dd=self.deconv(out[:,:,0:self.ndim:2,0:self.ndim:2])
        #     dd=self.conv.bkwd(data)
        #     rec=self.lamda1*torch.sum((dd-data_orig)*(dd-data_orig))
        #    tot+=rec

        loss = recloss + tot


        if (d_type == 'train' or opt=='mu'):
            loss.backward()
            optim.step()

        return recloss.item(), loss.item()

    def run_epoch(self, train, epoch,num_mu_iter, MU, LOGVAR,PI, d_type='test',fout=None):


        if (d_type=='train'):
            self.train()
        tr_recon_loss = 0;tr_full_loss = 0
        ii = np.arange(0, train[0].shape[0], 1)
        # if (d_type=='train'):
        #   np.random.shuffle(ii)
        tr = train[0][ii].transpose(0, 3, 1, 2)
        y = np.argmax(train[1][ii], axis=1)
        mu = MU[ii]
        logvar = LOGVAR[ii]
        pi = PI[ii]
        for j in np.arange(0, len(y), self.bsz):

            data_in = torch.from_numpy(tr[j:j + self.bsz]).float().to(self.dv)
            data = self.preprocess(data_in)
            target=None
            if (self.n_class>0):
                target = torch.from_numpy(y[j:j + self.bsz]).float().to(self.dv)
            if self.opt:
                self.update_s(mu[j:j + self.bsz, :], logvar[j:j + self.bsz, :], pi[j:j + self.bsz], self.mu_lr[0])
                for it in range(num_mu_iter):
                    data_d = data.detach()
                    self.compute_loss_and_grad(data_d,data_in, target, d_type, self.optimizer_s, opt='mu')
            with torch.no_grad() if (d_type != 'train') else dummy_context_mgr():
                recon_loss, loss=self.compute_loss_and_grad(data, data_in, target,d_type,self.optimizer)

            if self.opt:
                mu[j:j + self.bsz] = self.mu.data
                logvar[j:j + self.bsz] = self.logvar.data
                pi[j:j + self.bsz] = self.pi.data
                del self.mu, self.logvar, self.pi
            tr_recon_loss += recon_loss
            tr_full_loss += loss


        fout.write('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}\n'.format(d_type,
        epoch, tr_recon_loss / len(tr), tr_full_loss/len(tr)))

        return mu, logvar, pi

    def recon(self,input,num_mu_iter=None):


        if self.opt:
            mu, logvar, ppi = self.initialize_mus(input, True)

        num_inp=input.shape[0]
        self.setup_id(num_inp)
        input = input.to(self.dv)
        inp = self.preprocess(input)
        if self.opt:
            self.update_s(mu, logvar, ppi, self.mu_lr[0])
            for it in range(num_mu_iter):
                self.compute_loss_and_grad(inp,input, None, 'test', self.optimizer_s, opt='mu')
            s_mu = self.mu
            s_var = self.logvar
            pi = torch.softmax(self.pi, dim=1)
        else:
            s_mu, s_var, pi = self.encoder_mix(inp)

        ss_mu = s_mu.reshape(-1, self.n_mix, self.s_dim).transpose(0,1)
        ii = torch.argmax(pi, dim=1)
        jj = torch.arange(0,num_inp,dtype=torch.int64).to(self.dv)
        kk = ii+jj*self.n_mix
        lpi = torch.log(pi)
        recon_batch = self.decoder_and_trans(ss_mu)
        tot = self.dens_apply(s_mu, s_var, lpi, pi)
        recloss = self.mixed_loss(recon_batch, inp,pi)
        print('LOSS', (tot + recloss)/num_inp)
        recon_batch = recon_batch.transpose(0, 1)
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
        if (self.s_dim>1):
            s = torch.randn(self.bsz, self.s_dim*self.n_mix).to(self.dv)
        else:
            s = torch.ones(self.bsz, self.s_dim*self.n_mix).to(self.dv)

        s = s.reshape(-1, self.n_mix, self.s_dim)
        if (theta is not None and self.u_dim>0):
            theta = theta.to(self.dv)
            for i in range(self.n_mix):
                s[:,i,0:self.u_dim]=theta
        s = s.transpose(0,1)
        x=self.decoder_and_trans(s)
        if (self.feats and not self.feats_back):
            rec_b = []
            for rc in x:
                rec_b += [self.conv.bkwd(rc.reshape(-1, self.feats, self.conv.x_hf, self.conv.x_hf))]
            x = torch.stack(rec_b, dim=0)

        x=x.transpose(0,1)
        jj = torch.arange(0, self.bsz, dtype=torch.int64).to(self.dv)
        kk = ii + jj * self.n_mix
        recon = x.reshape(self.n_mix * self.bsz, -1)
        rr = recon[kk]
        if (self.feats and not self.feats_back):
            rrm=torch.min(rr,dim=1)[0].unsqueeze(dim=1)
            rrM=torch.max(rr,dim=1)[0].unsqueeze(dim=1)
            rr=(rr-rrm)/(rrM-rrm)
        return rr



def get_scheduler(args,model):
    scheduler=None
    if args.wd:
        l2 = lambda epoch: pow((1.-1. * epoch/args.nepoch),0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(model.optimizer, lr_lambda=l2)

    return scheduler

