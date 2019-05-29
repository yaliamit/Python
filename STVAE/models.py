import torch
import torch.nn.functional as F
from torch import nn, optim
from tps import TPSGridGen

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
        if self.type!='vae':
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

        self.x2h = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.ReLU()
        )

        self.h2he = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
        )

        self.h2hd = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim)
        )

        self.h2x=nn.Linear(self.h_dim, self.x_dim)

        self.h2smu = nn.Linear(self.h_dim, self.s_dim)
        self.h2svar = nn.Linear(self.h_dim, self.s_dim)
        if (self.type=='stvae' or self.type=='vae'):
            self.s2s = nn.Linear(self.s_dim, self.s_dim)
        elif (self.type=='tvae'):
            self.u2u = nn.Linear(self.u_dim, self.u_dim)
        self.z2h = nn.Linear(self.z_dim, self.h_dim)

        self.optimizer = optim.Adadelta(self.parameters())


    def forward_encoder(self, inputs):
        h=self.x2h(inputs)
        for i in range(self.num_hlayers):
            h=self.h2he(h)
        s_mu = self.h2smu(h)
        s_var = F.threshold(self.h2svar(h), -6, -6)
        return s_mu, s_var

    def sample_s(self, mu, logvar):
        eps = torch.randn(self.bsz, self.s_dim).to(self.dv)
        return mu + torch.exp(logvar/2) * eps

    def forward_decoder(self, z):
        h=self.z2h(z)
        for i in range(self.num_hlayers):
            h=self.h2hd(h)
        x=F.sigmoid(self.h2x(h))
        return x

    def forward(self, inputs):
        s_mu, s_var = self.forward_encoder(inputs.view(-1, self.x_dim))
        s = self.sample_s(s_mu, s_var)
        # Apply linear transformation to entire sampled vector.
        if (self.type=='stvae' or self.type=='vae'):
            s = self.s2s(s)
        # Extract non-transformation part
        z = s.narrow(1,self.u_dim,self.z_dim)
        # Create image from it.
        x = self.forward_decoder(z)
        if self.type !=  'vae':
            u = F.tanh(s.narrow(1, 0, self.u_dim))
            # Apply linear only to dedicated transformation part of sampled vector.
            if (self.type=='tvae'):
                u=self.u2u(u)
            if self.tf == 'aff':
                self.theta = u.view(-1, 2, 3) + self.id
                grid = F.affine_grid(self.theta, x.view(-1,self.h,self.w).unsqueeze(1).size())
            elif self.tf=='tps':
                self.theta = u + self.id
                grid = self.gridGen(self.theta)
            x = F.grid_sample(x.view(-1,self.h,self.w).unsqueeze(1), grid, padding_mode='border')
        x = x.clamp(1e-6, 1-1e-6)
        return x, s_mu, s_var

    def sample_from_z_prior(self,theta=None):
        s = torch.randn(self.bsz, self.s_dim).to(self.dv)
        #theta=theta.to(self.dv)
        if (self.type=='stvae' or self.type=='vae'):
            s=self.s2s(s)
        z = s.narrow(1, self.u_dim, self.z_dim)
        x = self.forward_decoder(z)
        if self.type != 'vae':
            u = F.tanh(s.narrow(1, 0, self.u_dim))
            if self.type=='tvae':
                if theta is not None:
                    u=self.u2u(theta)
                else:
                    u=self.u2u(u)
            if self.tf == 'aff':
                self.theta = u.view(-1, 2, 3) + self.id
                grid = F.affine_grid(self.theta, x.view(-1, self.h, self.w).unsqueeze(1).size())
            else:
                self.theta=u+self.id
                grid = self.gridGen(self.theta+self.id)
            x = F.grid_sample(x.view(-1,self.h,self.w).unsqueeze(1), grid).detach()
        else:
            x=x.view(-1,self.h,self.w).detach()
        return x

    def loss_V(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.squeeze().view(-1, self.x_dim), x.view(-1, self.x_dim), size_average=False)
        KLD1 = -0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar))  # z
        return BCE, KLD1

    def test(self,te,epoch):
        self.eval()
        test_recon_loss = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(te):
                data = data.to(self.dv)
                target = target.to(self.dv)
                recon_batch, smu, slogvar = self(data)
                recon_loss, kl = self.loss_V(recon_batch, data, smu, slogvar)
                loss = recon_loss + kl
                test_recon_loss += recon_loss.item()

        test_recon_loss /= (len(te) * self.bsz)
        print('====> Epoch:{} Test reconstruction loss: {:.4f}'.format(epoch, test_recon_loss))

    def train_epoch(self, tr, epoch):
        self.train()
        tr_recon_loss = 0

        for _, (data, target) in enumerate(tr):
            data = data.to(self.dv)
            target = target.to(self.dv)
            self.optimizer.zero_grad()
            recon_batch, smu, slogvar = self(data)
            recon_loss, kl = self.loss_V(recon_batch, data, smu, slogvar)

            loss = recon_loss + kl
            loss.backward()
            tr_recon_loss += recon_loss.item()
            self.optimizer.step()

        print('====> Epoch: {} Reconstruction loss: {:.4f}'.format(
            epoch, tr_recon_loss / (len(tr) * self.bsz)))
