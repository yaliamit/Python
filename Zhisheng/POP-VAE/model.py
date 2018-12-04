import torch
#import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable

'''
Generate 2d rotation and translation matrix
dgtM: size batch*3: dim 1--degree of rotation; dim 2,3---translation
'''
def rotTrMatrix(dgtM,device):
    dgM = dgtM[:,0].view(-1,1)
    r1 = torch.cos(dgM)
    r2 = torch.sin(dgM)
    return torch.cat((torch.cat((r1, -r2 , r2 ,r1),dim=1).view(-1,2,2),dgtM[:,(1,2)].view(-1,2,1)),dim=2).to(device)

def anchor_move(h,w,num_parts_h,num_parts_w):
    k=num_parts_h*num_parts_w
    
    W =  torch.cat((torch.eye(2),torch.zeros(2).unsqueeze(1)),dim=1)
    W = W.repeat(np.int(k),1,1)
    x = (torch.arange(start=0, end=num_parts_w, step=1, out=None, dtype=torch.float, layout=torch.strided,  requires_grad=False)*(w/num_parts_w) + w/num_parts_w/2)/w - 0.5
    y = (torch.arange(start=0, end=num_parts_h, step=1, out=None, dtype=torch.float, layout=torch.strided,  requires_grad=False)*(h/num_parts_h) + h/num_parts_h/2)/h - 0.5
    
    
    x = x.repeat(1,np.int(num_parts_h))
    y = y.repeat(np.int(num_parts_w),1)
    y= y.t().contiguous().view(1,np.int(k))
    
    W[:,0,2] = x
    W[:,1,2] = y
    
    return W.to('cuda')

class POPVAE(nn.Module):

    def __init__(self, x_h, x_w, h_dim, z_dim, u_dim, mb_size, nc,device,num_parts,num_parts_h,num_parts_w,part_h, part_w,t,M):
        super(POPVAE, self).__init__()

        self.x_dim = x_h * x_w # height * width
        self.part_h = part_h
        self.part_w = part_w
        self.part_dim = part_h*part_w # size of each part
        self.h = x_h
        self.w = x_w
        self.h_dim = h_dim # hidden layer
        
        self.z_large_dim = z_dim*num_parts #here let z_large_dim = 40
        self.u_large_dim = u_dim*num_parts
        
        self.num_parts_h = num_parts_h
        self.num_parts_w = num_parts_w
        
        self.z_dim = z_dim # generic latent variable, here is 1
        self.u1_dim = u_dim 
        self.bsz = mb_size #batch size
        self.num_parts = num_parts#num of parts in POP model
        self.dv = device
        self.nc=nc #num of channels
        
        """
        encoder: two fc layers
        """
        self.x2h = nn.Sequential(
            nn.Linear(self.x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
            )
        
        
        self.tf = t

        
        
        
        self.h2umu = nn.Linear(h_dim,self.u_large_dim)
        nn.init.zeros_(self.h2umu.weight)
        nn.init.zeros_(self.h2umu.bias)
        self.h2uvar = nn.Linear(h_dim,self.u_large_dim)
       
        #Layer for global transformation
        self.h2gumu = nn.Linear(h_dim,self.u1_dim)
        self.h2guvar = nn.Linear(h_dim,self.u1_dim)
       
        #use the same u' to u layer for every part 
        
        '''separate generate u
        '''
        self.u2u = nn.Linear(self.u1_dim,self.u1_dim)
        
        
        """
        decoder: two fc layers
        same network for diff parts
        """
        self.z2parts = nn.Sequential(
                    nn.Linear(self.z_dim,self.num_parts*self.part_dim,bias=False),
                    nn.Sigmoid()
                    )
        
        
        self.M = M


    def sample_z(self, mu, var):
       
        eps = Variable(mu.data.new(mu.size()).normal_()).to(self.dv)
        return mu + var * eps

    def sample_u(self, mu, var):
        
        eps = Variable(mu.data.new(mu.size()).normal_()).to(self.dv)
        return mu + var * eps
    
    '''
    generate kernel matrix
    bs is batch size; nc is num of channels
    typ = 'loc', 'gaussian'
    For 'loc', s is the size of the nonzero area
    For 'gaussian', s is the the parameter in the exponential
    '''
    
    def forward(self, inputs):
        h = self.x2h(inputs.view(-1, self.x_dim))
        
        '''
        distribution for global transf
        '''
        globalu_mu = torch.tanh(self.h2gumu(h))
        globalu_var = torch.exp(F.threshold(self.h2guvar(h),-6,6))
        global_u = self.sample_u(globalu_mu,globalu_var)
    
        '''
        distribution of z for each part
        '''
        
        '''
        distribution of transf for each part
        '''
        u_large_mu = torch.tanh(self.h2umu(h))
        u_large_logvar = F.threshold(self.h2uvar(h), -6, -6)
        u_large_var = torch.exp(u_large_logvar)
        
        
       
        u_large = self.sample_u(u_large_mu, u_large_var).view(-1,self.num_parts,self.u1_dim)
        
        
        
        parts = self.z2parts(torch.ones(self.bsz,self.z_dim).to(self.dv)).view(self.bsz,self.num_parts,self.part_dim)
        '''
        padding parts to the size of full image
        '''
        pad_h1 = int((self.h-self.part_h)/2)
        pad_h2 = self.h-self.part_h-pad_h1
        pad_w1 = int((self.w-self.part_w)/2)
        pad_w2 = self.w-self.part_w-pad_w1
        parts = F.pad(parts.view(-1,self.num_parts,self.part_h,self.part_w),(pad_h1,pad_h2,pad_w1,pad_w2))
        
        '''
        anchor move for parts
        '''
        Place = anchor_move(self.h,self.w,self.num_parts_h,self.num_parts_w)
        
        Place = Place.repeat(inputs.size()[0],1,1)
        
        anchor_place = F.affine_grid(Place, parts.view(-1,self.h,self.w).unsqueeze(1).size())
        parts = F.grid_sample(parts.view(-1,self.h,self.w).unsqueeze(1), anchor_place, padding_mode='zeros').view(-1,self.num_parts,self.h,self.w)
        '''
        anchor move for kernal
        '''
        KM = self.M
        KM = KM[0:inputs.size()[0]*self.num_parts,:,:,:]
        KM = F.grid_sample(KM.view(-1,self.h,self.w).unsqueeze(1), anchor_place, padding_mode='zeros').view(-1,1,self.h,self.w)
        
        '''
        Do transformation to parts and kernal
        '''
       
        u = u_large.view(-1,self.u1_dim)
        theta = rotTrMatrix(u,self.dv) #+ self.idty
        
        grid = F.affine_grid(theta, parts.view(-1,self.h,self.w).unsqueeze(1).size())
            
        theta_K = torch.cat((torch.eye(2).expand(inputs.shape[0]*self.num_parts,2,2).to(self.dv),theta[:,:,2].unsqueeze(2)),dim=2).view(-1,2,3)
        grid_K = F.affine_grid(theta_K, parts.view(-1,self.h,self.w).unsqueeze(1).size())
        KM = F.grid_sample(KM, grid_K,padding_mode='zeros').view(-1,self.num_parts,self.h,self.w)
       
        ###transf x after padding
        x = F.grid_sample(parts.view(-1,self.h,self.w).unsqueeze(1), grid, padding_mode='zeros').view(-1,self.num_parts,self.h,self.w)
       
            
        '''
        add parts together
        '''
        
        x=torch.sum(x*KM,1)
        y = torch.sum(KM,1)
        y[y==0]=1
        x = x/y
        '''
        global transf
        '''
       
        theta_g = rotTrMatrix(global_u,self.dv)
        grid_g = F.affine_grid(theta_g, x.view(-1,self.h,self.w).unsqueeze(1).size())
        x = F.grid_sample(x.view(-1,self.h,self.w).unsqueeze(1), grid_g, padding_mode='zeros')
        global_U = {'mu':globalu_mu,'var':globalu_var,'u':global_u}
        '''
        shrink toward 1/2
        '''
       
        
        return x, u_large_mu, u_large_var, u, global_U,parts, KM,theta
