import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
from get_net_text import get_network



class enc_dec_conv2(nn.Module):
    def __init__(self,device,inp_f,out_f,filt_s,pool_w,pool_s,opt,x_hw,non_lin=None):

        super(enc_dec_conv2,self).__init__()
        x_h=x_hw[0]
        x_w=x_hw[1]
        #pp = np.int32(np.floor(filt_s / 2))
        pp = np.int32(np.floor(filt_s / 2)/pool_s)

        self.feats=out_f
        self.inp_f=inp_f
        self.x_dim = np.int32((x_h / pool_s) * (x_w / pool_s) * out_f)
        self.x_h=x_h
        self.x_hf = np.int32(x_h / pool_s)
        self.x_wf = np.int32(x_w / pool_s)
        self.x_hwf=[self.x_hf,self.x_wf]
        self.dv=device
        #self.conv = torch.nn.Conv2d(inp_f, out_f, filt_s, stride=1, bias=False,
        #                            padding=pp).to(self.dv)
        if not opt:
            self.conv = torch.nn.Conv2d(inp_f, out_f, filt_s, stride=pool_s, bias=False,
                                    padding=pp).to(self.dv)
            self.drop_enc=torch.nn.Dropout(.5)
            self.bn=torch.nn.Identity() #BatchNorm2d(out_f)
        #self.deconv = torch.nn.ConvTranspose2d(out_f, inp_f, filt_s, stride=pool_s,
        #                                       padding=pp, output_padding=1, bias=False).to(self.dv)
        self.deconv = torch.nn.ConvTranspose2d(out_f, inp_f, filt_s, stride=pool_s,
                                               padding=pp, output_padding=0, bias=False).to(self.dv)
        self.drop_dec=torch.nn.Dropout(.5)
        #self.deconv.weight.data = self.conv.weight.data
        self.dbn=torch.nn.Identity() #BatchNorm2d(inp_f)

        # if (np.mod(pool_w, 2) == 1):
        #     pad = np.int32(pool_w / 2)
        # else:
        #     pad = np.int32((pool_w - 1) / 2)
        # self.pool = nn.MaxPool2d(pool_w, stride=pool_s, padding=(pad, pad))
        self.pool=nn.Identity().to(self.dv)
        if non_lin is None:
            self.nonl = nn.Identity().to(self.dv)
        elif non_lin == 'relu':
            self.nonl = nn.ReLU().to(self.dv)


    def fwd(self,xx):
            xx=self.conv(xx)
            xx=self.bn(xx)
            xx=self.nonl(self.pool(xx))
            #xx=self.drop_enc(xx)
            return xx


    def bkwd(self,xx):
        xx=self.deconv(xx.reshape(-1, self.feats, self.x_hf, self.x_wf))
        if (self.inp_f>3):
            xx = self.dbn(xx)
            xx = self.nonl(xx)
            #xx = self.drop_dec(xx)
        return xx


class ENC_DEC(nn.Module):
    def __init__(self, sh, device, args):
        super(ENC_DEC, self).__init__()


        _,layers = get_network(args.enc_layers)
        self.layers_text = layers
        self.dv=device
        self.x_hw = sh
        self.OPT = args.OPT
        self.setup(sh)


    def setup(self,sh):

        enc_hw=sh[1:3]
        enc_inp_f=sh[0]
        self.layers=nn.ModuleList()
        for i,ll in enumerate(self.layers_text):
            if 'conv' in ll['name']:
                non_l=None
                if 'non_linearity' in ll:
                    non_l=ll['non_linearity']
                self.layers.append(enc_dec_conv2(self.dv,enc_inp_f,ll['num_filters'],ll['filter_size'],
                                       ll['pool_size'],ll['pool_stride'],self.OPT,enc_hw,non_l))

                enc_hw=self.layers[-1].x_hwf
                print(i,enc_hw)
                enc_inp_f=ll['num_filters']
        self.x_dim=enc_hw*ll['num_filters']


    def forw(self,input):

        out=input
        for l in self.layers:
            out=l.fwd(out)

        return(out)

    def bkwd(self,input):
        out=input
        for l in reversed(self.layers):
            out=l.bkwd(out)

        return(out)


