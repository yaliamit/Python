# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:44:42 2019

@author: Qing Yan
"""
import torch
import torch.nn.functional as F

def step_forward(model, xs,label):
    """
    Forward pass through all invertible coupling layers and W-transf
    
    Args:
    * xs: float tensor of shape (B,dim).

    Returns:
    * ys: float tensor of shape (B,dim).
    """
    step = []
    for i in range(model.num_block):
        if i==0:
            ys, logd = model.layers[2*i](xs)
            ys, dlogd = model.layers[2*i+1](ys)
            logd += dlogd
        else:
            ys, dlogd = model.layers[2*i](ys)
            logd += dlogd
            ys, dlogd = model.layers[2*i+1](ys)
            logd += dlogd
        ys = F.linear(ys,model.permutes[i])
        if model.num_cl > 1:
            ys += model.cl_fc_layers[i](label)
        step.append(ys.detach().cpu().numpy())
        logd += model.logd_inv(i)
    ys = torch.matmul(ys, torch.diag(torch.exp(model.scaling_diag)))
    step.append(ys.detach().cpu().numpy())
    return step