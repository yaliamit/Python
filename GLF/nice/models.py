import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from .layers import AdditiveCouplingLayer, AffineCouplingLayer,SigmoidCouplingLayer

def _build_relu_network(in_dim, hidden_dim, num_layers,out_dim,cp = 0,bn = True):
    """Helper function to construct a ReLU network of varying number of layers."""
    _modules = [ nn.Linear(in_dim, hidden_dim) ]
    for _ in range(num_layers):
        _modules.append( nn.Linear(hidden_dim, hidden_dim) )
        _modules.append( nn.ReLU() )
        if bn:
            _modules.append( nn.BatchNorm1d(hidden_dim) )
    if cp > 0:
        MUL = 2
    else:
        MUL = 1   
    _modules.append( nn.Linear(hidden_dim, MUL*out_dim))
    if not bn:
        _modules.append(nn.Tanh())
    return nn.Sequential( *_modules )
    

class NICEModel(nn.Module):
    """
    FLOW model, modified from NICE MODEL https://github.com/paultsw/nice_pytorch
    
    Input: input_dim --- input dimension
           hidden_dim --- dimension of intermediate layer of each flow block
           num_layers --- number of FC-layers for each flow block
           num_block --- number of flow blocks in the flow model
           coupling --- which kind of coupling layer to be used: 0=additive, 1=affine, 2=sigmoid
           num_class --- dimension of one-hot label when using class-related model
    """
    def __init__(self, input_dim, hidden_dim, num_layers,num_block,coupling = 1,num_class = 0):
        super(NICEModel, self).__init__()
        assert (input_dim % 2 == 0), "[NICEModel] only even input dimensions supported for now"
        assert (num_layers > 2), "[NICEModel] num_layers must be at least 3"
        self.input_dim = input_dim
        self.num_block = num_block
        mask_list = []
        for _ in range(num_block):
            mask_list.append('odd')
            mask_list.append('even')
        half_dim = int(input_dim / 2)
        if coupling == 2:
            self.layers = nn.ModuleList([SigmoidCouplingLayer(input_dim, mask, _build_relu_network(half_dim, hidden_dim, num_layers,
                                                                                    half_dim,coupling)) for mask in mask_list])
        elif coupling == 1:
            self.layers = nn.ModuleList([AffineCouplingLayer(input_dim, mask, _build_relu_network(half_dim, hidden_dim, num_layers,
                                                                                    half_dim,coupling)) for mask in mask_list])
        else:
            self.layers = nn.ModuleList([AdditiveCouplingLayer(input_dim, mask, _build_relu_network(half_dim, hidden_dim, num_layers,
                                                                                    half_dim,coupling)) for mask in mask_list])
        self.scaling_diag = nn.Parameter(torch.ones(input_dim))
        # W-matrices parameters for invertible transf 
        w_shape = [input_dim,input_dim]
        self.permutes = nn.ParameterList([])
        for _ in range(num_block):
            w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')
            self.permutes.append(nn.Parameter(torch.Tensor(w_init)))
        # randomly initialize weights:
        for ly in self.layers:
            for p in ly.parameters():
                if len(p.shape) > 1:
                    init.kaiming_uniform_(p, nonlinearity='relu')
                else:
                    init.normal_(p, mean=0., std=0.001)
        # class-related network
        self.num_cl = num_class
        if num_class>1:
            self.cl_fc_layers = nn.ModuleList([_build_relu_network(num_class, hidden_dim, 2,input_dim,cp = 0,
                                                                   bn = False) for _ in range(num_block)])
        
    #Compute log determinant of W-matrix
    def logd_inv(self,idx):
        return torch.det(self.permutes[idx]).abs().log()
    
    def forward(self, xs,label = 0):
        """
        Forward pass through all invertible coupling layers and W-transf
        
        Args:
        * xs: float tensor of shape (B,dim).

        Returns:
        * ys: float tensor of shape (B,dim).
        """
        if self.num_cl > 1:
            assert (label.shape[1] == self.num_cl), "wrong one-hot dimension when using class-related model"
        for i in range(self.num_block):
            if i==0:
                ys, logd = self.layers[2*i](xs)
                ys, dlogd = self.layers[2*i+1](ys)
                logd += dlogd
            else:
                ys, dlogd = self.layers[2*i](ys)
                logd += dlogd
                ys, dlogd = self.layers[2*i+1](ys)
                logd += dlogd
            ys = F.linear(ys,self.permutes[i])
            logd += self.logd_inv(i)
            if self.num_cl > 1:
                ys += self.cl_fc_layers[i](label)
        ys = torch.matmul(ys, torch.diag(torch.exp(self.scaling_diag)))
        return ys, logd


    def inverse(self, ys,label = 0):
        """Inverse Transform"""
        if self.num_cl > 1:
            assert (label.shape[1] == self.num_cl), "wrong one-hot dimension when using class-related model"
        with torch.no_grad():
            xs = torch.matmul(ys, torch.diag(torch.reciprocal(torch.exp(self.scaling_diag))))
            for i in range(self.num_block):
                j = self.num_block - 1 - i
                if self.num_cl > 1:
                    xs -= self.cl_fc_layers[j](label)
                xs = F.linear(xs,self.permutes[j].inverse())
                xs = self.layers[2*j+1].inverse(xs)
                xs = self.layers[2*j].inverse(xs)
        return xs
