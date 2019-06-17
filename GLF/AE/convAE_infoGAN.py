"""
Auto-Encoder 
the same generator architecture of InfoGAN(https://arxiv.org/abs/1606.03657) is used as decoder
the encoder is symmetric to the decode.
"""
from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import os
import random
import torch.utils.data
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import numpy as np

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

            
class ConvAE(nn.Module):
    def __init__(self, args):
        super(ConvAE, self).__init__()
        if args.dataset == 'mnist' or args.dataset == 'Fashion-mnist':
            self.nc = 1
        else:
            self.nc = 3
        self.args = args
        self.have_cuda = True
        self.nz = args.num_latent
        self.input_size = args.image_size
        self.conv = nn.Sequential(
            nn.Conv2d(self.nc, 64, 4, 2, 1),
            #nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        
        self.encode_fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, self.nz),
        )
        
        self.decode_fc = nn.Sequential(
            nn.Linear(self.nz, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, self.nc, 4, 2, 1),
            nn.Tanh()
        )
        
        initialize_weights(self)

    def encode(self, x):
        conv = self.conv(x)
        h1 = self.encode_fc(conv.view(-1, 128*(self.input_size//4)*(self.input_size//4)))
        return h1
    
    def decode(self, z):
        deconv_input = self.decode_fc(z)
        deconv_input = deconv_input.view(-1,128, self.input_size//4, self.input_size//4)
        output = self.deconv(deconv_input)
        if self.args.dataset == 'mnist' or self.args.dataset == 'Fashion-mnist':
            if not self.args.loss_type == 'Perceptual':
                output = output*0.5 + 0.5
        return output

    def forward(self, x):
        z = self.encode(x)
        decoded = self.decode(z)
        return decoded, z