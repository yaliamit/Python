import torch
from torch import nn, optim
import numpy as np
import models_mix_by_class
import models_opt_mix
import contextlib
@contextlib.contextmanager
def dummy_context_mgr():
    yield None

class STVAE_OPT_mix_by_class(models_mix_by_class.STVAE_mix_by_class):

    def __init__(self, x_h, x_w, device, args):
        super(STVAE_OPT_mix_by_class, self).__init__(x_h, x_w, device, args)

        self.n_class=args.n_class
        self.n_mix_perclass=np.int32(self.n_mix/self.n_class)

        self.mu_lr = args.mu_lr
        self.eyy = torch.eye(self.n_mix).to(self.dv)
        if (args.optimizer == 'Adam'):
            self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        elif (args.optimizer == 'Adadelta'):
            self.optimizer = optim.Adadelta(self.parameters())

    def update_s(self, mu, logvar, pi, epoch):
        mu_lr = self.mu_lr[0]
        if epoch > 200:
            mu_lr = self.mu_lr[1]
        self.mu = torch.autograd.Variable(mu.to(self.dv), requires_grad=True)
        self.logvar = torch.autograd.Variable(logvar.to(self.dv), requires_grad=True)
        self.pi = torch.autograd.Variable(pi.to(self.dv), requires_grad=True)
        self.optimizer_s = optim.Adam([self.mu, self.logvar, self.pi], mu_lr)


    def forward(self,data,targ):

        pit=torch.softmax(self.pi, dim=1)
        return self.get_loss(data,targ, self.mu, self.logvar, pit)

    def compute_loss_and_grad(self,data, targ, type, optim, opt='par'):

        optim.zero_grad()

        recon_loss, tot= self.forward(data,targ)

        loss = recon_loss + tot


        if (type == 'train' or opt=='mu'):
            loss.backward()
            optim.step()

        ls=loss.item()
        rcs=recon_loss.item()

        return rcs, ls


    def run_epoch(self, train, epoch,num_mu_iter, MU, LOGVAR,PI, type='test',fout=None):

        if (type=='train'):
            self.train()
        tr_recon_loss = 0;tr_full_loss = 0
        ii = np.arange(0, train[0].shape[0], 1)
        # if (type=='train'):
        #   np.random.shuffle(ii)
        tr = train[0][ii].transpose(0, 3, 1, 2)
        y = np.argmax(train[1][ii],axis=1)
        mu = MU[ii]
        logvar = LOGVAR[ii]
        pi = PI[ii]
        batch_size = self.bsz
        for j in np.arange(0, len(y), self.bsz):
            data = torch.from_numpy(tr[j:j + self.bsz]).float().to(self.dv)
            target = torch.from_numpy(y[j:j + self.bsz]).float().to(self.dv)
            self.update_s(mu[j:j + batch_size, :], logvar[j:j + batch_size, :], pi[j:j + batch_size], epoch)
            for it in range(num_mu_iter):
                self.compute_loss_and_grad(data, target, type, self.optimizer_s, opt='mu')
            with torch.no_grad() if (type != 'train') else dummy_context_mgr():
                recon_loss, loss = self.compute_loss_and_grad(data, target, type, self.optimizer)

            tr_recon_loss += recon_loss
            tr_full_loss += loss


        fout.write('====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}\n'.format(type,
        epoch, tr_recon_loss / len(tr), tr_full_loss/len(tr)))

        return MU, LOGVAR, PI

    def run_epoch_classify(self, train, epoch, num_mu_iter, fout=None):

        self.eval()
        ii = np.arange(0, train[0].shape[0], 1)
        # if (type=='train'):
        #   np.random.shuffle(ii)
        tr = train[0][ii].transpose(0, 3, 1, 2)
        y = np.argmax(train[1][ii],axis=1)
        acc=0
        for j in np.arange(0, len(y), self.bsz):

            data = torch.from_numpy(tr[j:j + self.bsz]).float().to(self.dv)
            target = torch.from_numpy(y[j:j + self.bsz]).float().to(self.dv)
            for it in range(num_mu_iter):
                models_opt_mix.compute_loss_and_grad(data, type, self.optimizer_s, opt='mu')

            s_mu = self.mu.view(-1, self.n_mix, self.s_dim)
            recon_batch = self.decoder_and_trans(s_mu)
            b = self.mixed_loss_pre(recon_batch, data, self.pi.shape[1])
            vy, by= torch.min(b,1)
            by=np.int32(by.detach().cpu().numpy()/self.n_mix_perclass)

            acc+=np.sum(np.equal(by,y[j:j+self.bsz]))

        fout.write('====> Epoch {}: {} Accuracy: {:.4f}\n'.format(type,
        epoch, acc/ len(tr)))




    def recon(self,input,num_mu_iter,cl):

        num_inp=input.shape[0]
        self.setup_id(num_inp)
        inp = input.to(self.dv)

        mu, logvar, pi = self.initialize_mus(input, True)
        self.update_s(mu, logvar, pi, 0)
        for it in range(num_mu_iter):
            models_opt_mix.compute_loss_and_grad(input, type, self.optimizer_s, opt='mu')
        s_mu = self.mu.view(-1, self.n_mix, self.s_dim)
        pi = self.pi.view(-1,self.n_class,self.n_mix_perclass)
        pi= pi[:,cl,:]
        recon_batch = self.decoder_and_trans(s_mu)
        recon_batch=recon_batch.reshape(-1,self.n_class,self.n_mix_perclass,recon_batch.shape[-1])
        ii = torch.argmax(pi, dim=1)
        recon_batch=recon_batch[:,cl,:,:]
        jj = torch.arange(0,num_inp,dtype=torch.int64).to(self.dv)
        kk = ii+jj*self.n_mix_perclass
        recon=recon_batch.reshape(self.n_mix_perclass*num_inp,-1)
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

