# A very straightforward implementation of TPS transformation
# taken from https://github.com/ignacio-rocco/cnngeometric_pytorch/blob/master/geotnf/transformation.py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TPSGridGen(nn.Module):
    def __init__(self, out_h=28, out_w=28, grid_size=3, device="cpu"):
        super(TPSGridGen, self).__init__()
        self.h = out_h
        self.w = out_w
        self.dv = device

        # sampling grid
        self.gridX, self.gridY = np.meshgrid(np.linspace(-1,1,out_w),np.linspace(-1,1,out_h))
        # grid_X, grid_Y: H * W

        #DONE: reshape the tensor, now 1 * H * W * 1
        self.gridX = torch.FloatTensor(self.gridX).unsqueeze(0).unsqueeze(3).to(self.dv)
        self.gridY = torch.FloatTensor(self.gridY).unsqueeze(0).unsqueeze(3).to(self.dv)

        # use regular grid
        axis_coords = np.linspace(-1, 1, grid_size)
        self.N = grid_size * grid_size
        PX, PY = np.meshgrid(axis_coords, axis_coords)
        PX = torch.Tensor(np.reshape(PX, (-1,1))) # (N,1)
        PY = torch.Tensor(np.reshape(PY, (-1,1))) # (N,1)


        self.Li = self.compute_L_inverse(PX,PY).unsqueeze(0) # 1 * (N + 3) * (N + 3)
        self.PX = PX.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4).to(self.dv) # 1 * 1 * 1 * 1 * N
        self.PY = PY.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0,4).to(self.dv)

        self.U = self.compute_U()

    def forward(self, theta):
        warped_grid = self.applytf(theta, torch.cat((self.gridX, self.gridY),3))
        return warped_grid

    def compute_L_inverse(self, X, Y):
        N = X.size()[0]
        Xmat = X.expand(N,N)
        Ymat = Y.expand(N,N)
        D = torch.pow(Xmat-Xmat.t(),2) + torch.pow(Ymat-Ymat.t(),2)
        D[D==0] = 1
        K = torch.mul(D, torch.log(D))

        P = torch.cat((torch.FloatTensor(N,1).fill_(1),X,Y),1)
        O = torch.FloatTensor(3,3).fill_(0)
        L = torch.cat((torch.cat((K,P),1),torch.cat((P.t(),O),1)),0)
        Li = torch.inverse(L).to(self.dv)
        return Li

    def compute_U(self):
        Px = self.PX.expand((1,self.h,self.w,1,self.N))
        Py = self.PY.expand((1,self.h,self.w,1,self.N))
        px = self.gridX.unsqueeze(4).expand(self.gridX.size()+(self.N,))
        py = self.gridY.unsqueeze(4).expand(self.gridY.size()+(self.N,))
        dx = px - Px
        dy = py - Py
        dist = torch.pow(dx, 2) + torch.pow(dy, 2)
        dist[dist==0] = 1
        return torch.mul(dist, torch.log(dist))

    def applytf(self,theta,points):
        # points of size (1,H,W,2)
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        bsz = theta.size(0)
        Qx = theta[:,:self.N,:,:].squeeze(3)
        Qy = theta[:,self.N:,:,:].squeeze(3)

        Wx = torch.bmm(self.Li[:,:self.N,:self.N].expand((bsz,self.N,self.N)),Qx)
        Wy = torch.bmm(self.Li[:,:self.N,:self.N].expand((bsz,self.N,self.N)),Qy)

        Wx = Wx.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,self.h,self.w,1,1)
        Wy = Wy.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,self.h,self.w,1,1)

        Ax = torch.bmm(self.Li[:,self.N:,:self.N].expand((bsz,3,self.N)),Qx)
        Ay = torch.bmm(self.Li[:,self.N:,:self.N].expand((bsz,3,self.N)),Qy)

        Ax = Ax.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,self.h,self.w,1,1)
        Ay = Ay.unsqueeze(3).unsqueeze(4).transpose(1,4).repeat(1,self.h,self.w,1,1)
        
        pxb = self.gridX.expand((bsz,)+self.gridX.size()[1:])
        pyb = self.gridY.expand((bsz,)+self.gridY.size()[1:])

        fx = Ax[:,:,:,:,0] + \
             torch.mul(Ax[:,:,:,:,1],pxb) + \
             torch.mul(Ax[:,:,:,:,2],pyb) + \
             torch.sum(torch.mul(Wx,self.U.expand_as(Wx)),4)
        fy = Ay[:,:,:,:,0] + \
             torch.mul(Ay[:,:,:,:,1],pxb) + \
             torch.mul(Ay[:,:,:,:,2],pyb) + \
             torch.sum(torch.mul(Wy,self.U.expand_as(Wy)),4)

        return torch.cat((fx,fy),3)
