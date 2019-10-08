import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import os
import sys
import argparse
import time
import aux

# Network module
class CLEAN(nn.Module):
    def __init__(self, device, x_dim, y_dim, args):
        super(CLEAN, self).__init__()

        self.first=True
        self.lenc=args.lenc # Maximal number of characters in string. All strings padded with spaces to that length
        self.bsz=args.bsz # Batch size - gets multiplied by number of shifts so needs to be quite small.
        self.x_dim=x_dim # Dimensions of all images.
        self.y_dim=y_dim
        self.full_dim=x_dim*y_dim
        self.dv=device
        self.ll=args.ll # Number of possible character labels.
        self.weights=torch.ones(self.ll).to(device)
        self.weights[0]=1.
        self.pools = args.pools # List of pooling at each level of network
        self.drops=args.drops # Drop fraction at each level of network
        ff=len(args.filts) # Number of filters = number of conv layers.
        # Create list of convolution layers with the appropriate filter size, number of features etc.
        self.convs = nn.ModuleList([torch.nn.Conv2d(args.feats[i], args.feats[i+1],
                                                    args.filts[i],stride=1,padding=np.int32(np.floor(args.filts[i]/2)))
                                    for i in range(ff)])

        # The loss function
        self.criterion=nn.CrossEntropyLoss()
        self.criterion_shift=nn.CrossEntropyLoss(reduce=False)
        if (args.optimizer == 'Adam'):
            self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=args.lr)

    # Apply sequence of conv layers up to the final one that will be determined later.
    def forward_pre(self,input):

        out=input
        if (self.first):
            self.pool_layers=[]
        for i, cc in enumerate(self.convs):
            if (self.first):
                print(out.shape)
            out=cc(out)
            pp=torch.fmod(torch.tensor(out.shape),self.pools[i])
            if (self.pools[i]>1):
                if (self.first):
                    self.pool_layers+=[nn.MaxPool2d(self.pools[i],padding=tuple(pp[2:4]))]
                out=self.pool_layers[i](out)
            else:
                if (self.first):
                    self.pool_layers+=[None]
            if (self.drops[i]<1.):
                out=nn.functional.dropout(out,self.drops[i])
            # Relu non-linearity at each level.
            out=F.relu(out)
        return(out)

    # Full network
    def forward(self,input):

        out=self.forward_pre(input)

        # During first pass check if dropout required.
        if (self.drops[-1]<1.):
            if (self.first):
                self.dr2d=nn.Dropout2d(self.drops[-1])
            out=self.dr2d(out)
        # If first time running through setup last layer.
        if (self.first):
            # Get dimensions of current output
            self.sh=out.shape
            # Create x-size of new filter, y-size is full y-dimension
            self.sh2a = np.int32(np.floor(self.sh[3] / self.lenc))
            # Compute padding to the end of the array
            self.pad = self.sh2a * (self.lenc)+1 - self.sh[3]
            print('pre final shape',out.shape,self.sh[2],self.sh2a, self.pad, self.lenc)
        # Concatenate the padding
        out=torch.cat((out,torch.zeros(out.shape[0],self.sh[1],self.sh[2],self.pad).to(self.dv)),dim=3)
        if (self.first):
            # Define last conv layer that has as many output features as labels - this is the vector of
            # of outputs that go to the softmax to define label probs.
            self.l_out=torch.nn.Conv2d(out.shape[1],self.ll,[self.sh[2],self.sh2a+1],stride=[1,self.sh2a]).to(self.dv)
        # Apply last layer
        out=self.l_out(out)
        if (self.first):
            print('final shape',out.shape)
            self.first=False

        return(out)

    # Get loss for optimal shift (same as other loss)
    def loss_shift(self,out,targ=None):


        outl=out.permute(1, 0, 2, 3).reshape([self.ll, -1]).transpose(0, 1)
        # poutl=torch.softmax(outl,dim=1)
        # v, mx = torch.max(poutl, dim=1)
        # PMX = torch.zeros_like(poutl)
        # PMX.scatter_(1, mx.reshape(-1, 1), 1)
        # poutl2=poutl-PMX*poutl
        # v2, mx2 = torch.max(poutl2,dim=1)
        # vv=v-v2
        # vv=torch.sum(vv.reshape(-1,self.lenc),dim=1)
        # vv=vv.reshape(-1,self.lst)
        # u,lossm=torch.max(vv,1)
        # MX = mx.reshape(-1, self.lenc)
        # ii = torch.arange(0, len(MX), self.lst, dtype=torch.int64).to(self.dv) + lossm
        # MSMX=MX[ii]
        # tot_loss=torch.tensor(0)
        #if (targ is None):
        v, mx=torch.max(outl,dim=1)
        MX=torch.zeros_like(outl)
        MX.scatter_(1,mx.reshape(-1,1),1)
        MX=MX.reshape(-1,self.lst,self.lenc,self.ll)
        SMX=torch.sum(MX,dim=1)
        VSMX, MSMX=torch.max(SMX,dim=2)
        spMX=MSMX[:,0]==0
        MSMX[spMX,0:self.lenc-1]=MSMX[spMX,1:self.lenc]
        MSMX[spMX,self.lenc-1]=0
        hhr = MSMX.repeat_interleave(self.lst, dim=0)
        loss = self.criterion_shift(outl, hhr.view(-1))
        slossa = torch.sum(loss.reshape(-1, self.lenc), dim=1).reshape(-1, self.lst)
        v, lossm = torch.min(slossa, 1)

        tot_loss=torch.mean(v)
        return lossm, tot_loss, MSMX





    # Find optimal shift/scale for each image
    def get_loss_shift(self,input_shift,target_shift, epoch, fout, type):
        self.eval()
        num_tr=len(input_shift)
        num_tro=num_tr/self.lst
        rmx = []
        # Loop over batches of training data each lst of them are transformation of same image.
        OUT=[]
        TS = (torch.from_numpy(target_shift)).type(torch.int64).to(self.dv)
        for j in np.arange(0, num_tr, self.bsz):
            # Data is stored as uint8 to save space. So transfer to float for gpu.
            sinput = (torch.from_numpy(input_shift[j:j + self.bsz]).float()/255.).to(self.dv)
            # Apply network
            out = self.forward(sinput)
            OUT+=[out]

        OUT=torch.cat(OUT,dim=0)

        # CHoose shift/scale based on labels obtained from a vote at each location.
        lossm=[]
        shift_loss=0
        jump=self.bsz*self.lst
        MSMX=[]
        for j in np.arange(0,num_tr,jump):
            lossmb, s_l, msmx=self.loss_shift(OUT[j:j+jump],target_shift[j:j+jump])
            lossm+=[lossmb]
            shift_loss+=s_l.item()
            MSMX+=[msmx]
        shift_loss/=(num_tr/jump)
        lossm=torch.cat(lossm,dim=0)
        MSMX=torch.cat(MSMX,dim=0).detach().cpu().numpy()
        ii=torch.arange(0,len(OUT),self.lst,dtype=torch.int64)+lossm.detach().cpu()

        outs=OUT[ii]
        stargs=TS[ii]

        # Get accuracy for chosen shift/scales
        outsp=outs.permute(1,0,2,3).reshape([self.ll,-1]).transpose(0,1).to(self.dv)
        target = (stargs.reshape(-1)).to(self.dv)
        loss, acc, acca, numa, accc, mx =self.get_acc_and_loss(outsp,target)

        # Extract best version of each image for the network training stage.
        train_choice_shift=(input_shift[ii])
        rmx += [mx.cpu().detach().numpy()]

        fout.write('====> {}: {} Full loss: {:.4F}\n'.format(type + '_shift', epoch,
                                                             shift_loss))
        fout.write(
            '====> Epoch {}: {} Full loss: {:.4F}, Full acc: {:.4F}, Non space acc: {:.4F}, case insensitive acc {:.4F}\n'.format(
                type, epoch,
                loss.item(), acc.item() / (num_tro * self.lenc), acca.item() / numa,
                accc.item() / (num_tro * self.lenc)))

        return train_choice_shift, rmx, MSMX

        # Get loss and accuracy (all characters and non-space characters).
    def get_acc_and_loss(self, out, targ):
            v, mx = torch.max(out, 1)
            # Non-space characters
            targa = targ[targ > 0]
            mxa = mx[targ > 0]
            numa = targa.shape[0]
            # Total loss
            loss = self.criterion(out, targ)
            # total accuracy
            acc = torch.sum(mx.eq(targ))
            # Accuracy on case insensitive
            mxc = 1 + torch.fmod((mx - 1), 26)
            targc = 1 + torch.fmod((targ - 1), 26)
            accc = torch.sum(mxc.eq(targc))
            # Accuracy on non-space
            acca = torch.sum(mxa.eq(targa))

            return loss, acc, acca, numa, accc, mx

    # GRADIENT STEP
    def loss_and_grad(self, input, target, type='train'):

        # Get output of network
        out=self.forward(input)

        if (type == 'train'):
            self.optimizer.zero_grad()
        # Compute loss and accuracy
        loss, acc, acca, numa, accc, mx=self.get_acc_and_loss(out.permute(1,0,2,3).reshape([self.ll,-1]).transpose(0,1),target.reshape(-1))

        # Perform optimizer step using loss as criterion
        if (type == 'train'):
            loss.backward()
            self.optimizer.step()

        return loss, acc, acca, numa, accc, mx

    # Epoch of network training
    def run_epoch(self, train, text, epoch, fout, type):

        if (type=='train'):
            self.train()
        else:
            self.eval()
        num_tr=train.shape[0]
        #ii = np.arange(0, num_tr, 1)
        #if (type=='train'):
        #   np.random.shuffle(ii)
        trin=train#[ii]
        targ=text#[ii]

        full_loss=0; full_acc=0; full_acca=0; full_numa=0; full_accc=0
        rmx=[]
        # Loop over batches.
        jump=self.bsz
        for j in np.arange(0, num_tr, jump):
            data = (torch.from_numpy(trin[j:j + jump]).float()/255.).to(self.dv)
            target = torch.from_numpy(targ[j:j + jump]).to(self.dv)
            target=target.type(torch.int64)

            loss, acc, acca, numa, accc, mx= self.loss_and_grad(data, target, type)
            full_loss += loss.item()
            full_acc += acc.item()
            full_acca+=acca.item()
            full_accc += accc.item()
            full_numa+=numa
            rmx+=[mx.cpu().detach().numpy()]
        fout.write('====> Epoch {}: {} Full loss: {:.4F}, Full acc: {:.4F}, Non space acc: {:.4F}, case insensitive acc {:.4F}\n'.format(type,epoch,
                    full_loss /(num_tr/self.bsz), full_acc/(num_tr*self.lenc), full_acca/full_numa, full_accc / (num_tr * self.lenc)))

        return(rmx)


    def get_scheduler(self,args):
        scheduler = None
        if args.wd:
            l2 = lambda epoch: pow((1. - 1. * epoch / args.nepoch), 0.9)
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=l2)

        return scheduler



