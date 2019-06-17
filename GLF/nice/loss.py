"""
Implementation of NICE log-likelihood loss.
"""
import torch
import torch.nn as nn
import numpy as np

# ===== ===== Loss Function Implementations ===== =====
"""
We assume that we final output of the network are components of a multivariate distribution that
factorizes, i.e. the output is (y1,y2,...,yK) ~ p(Y) s.t. p(Y) = p_1(Y1) * p_2(Y2) * ... * p_K(YK),
with each individual component's prior distribution coming from a standardized family of
distributions, i.e. p_i == Gaussian(mu,sigma) for all i in 1..K, or p_i == Logistic(mu,scale).
"""
def gaussian_nice_loglkhd(h, diag):
    """
    Definition of log-likelihood function with a Gaussian prior, as in the paper.
    
    Args:
    * h: float tensor of shape (N,D). First dimension is batch dim, second dim consists of components
      of a factorized probability distribution.
    * diag: scaling diagonal of shape (D,).

    Returns:
    * loss: torch float tensor of shape (N,).
    """
    # \sum^D_i s_{ii} - { (1/2) * \sum^D_i  h_i**2) + (D/2) * log(2\pi) }
    return torch.sum(diag) - (0.5*torch.sum(torch.pow(h,2),dim=1) + h.size(1)*0.5*torch.log(torch.tensor(2*np.pi).cuda()))

def logistic_nice_loglkhd(h, diag):
    """
    Definition of log-likelihood function with a Logistic prior.
    
    Same arguments/returns as gaussian_nice_loglkhd.
    """
    # \sum^D_i s_{ii} - { \sum^D_i log(exp(h)+1) + torch.log(exp(-h)+1) }
    return (torch.sum(diag) - (torch.sum(torch.log1p(torch.exp(h)) + torch.log1p(torch.exp(-h)), dim=1)))

# wrap above loss functions in Modules:
class GaussianPriorNICELoss(nn.Module):
    def __init__(self, size_average=True):
        super(GaussianPriorNICELoss, self).__init__()
        self.size_average = size_average

    def forward(self, fx, diag):
        if self.size_average:
            return torch.mean(-gaussian_nice_loglkhd(fx, diag))
        else:
            return torch.sum(-gaussian_nice_loglkhd(fx, diag))

class LogisticPriorNICELoss(nn.Module):
    def __init__(self, size_average=True):
        super(LogisticPriorNICELoss, self).__init__()
        self.size_average = size_average

    def forward(self, fx, diag):
        if self.size_average:
            return torch.mean(-logistic_nice_loglkhd(fx, diag))
        else:
            return torch.sum(-logistic_nice_loglkhd(fx, diag))
