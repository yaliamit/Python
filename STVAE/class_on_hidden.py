import torch
import sys
from torch import nn, optim
import torch.nn.functional as F

import numpy as np
import time


class NET(nn.Module):

    def __init__(self, hdim, ncl,args,device):
        super(NET, self).__init__()

        self.hdim=hdim
        self.bsz=args.mb_size
        self.dv=device
        self.lamda=args.lamda
        ihdim=hdim
        if (args.hid_hid>0):
            self.intermediate=nn.Linear(hdim,args.hid_hid)
            ihdim=args.hid_hid
        else:
            self.intermediate=nn.Identity()
        self.dropp = nn.Dropout(p=args.hid_prob)
        self.final = nn.Linear(ihdim, ncl)
        self.optimizer = optim.Adam(self.parameters(),args.lr)
        self.loss = nn.CrossEntropyLoss(reduction='sum')




    def l2_loss(self):
        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.sum(param.pow(2))

        reg_loss

        return reg_loss

    def forward(self,input):

        out=self.dropp(F.relu(self.intermediate(input)))
        logits=self.final(out)

        return logits

    def compute_loss_and_grad(self,data,target,d_type):

        if (d_type == 'train'):
            self.optimizer.zero_grad()
        logits = self(data)
        l2l=self.l2_loss()
        like = self.loss(logits, target)
        cost=like+self.lamda*l2l
        acc=torch.sum(torch.eq(torch.argmax(logits,dim=1),target))
        if (d_type == 'train'):
            cost.backward()
            self.optimizer.step()

        return like, acc, l2l

    def run_epoch(self, trainX, trainY,epoch, d_type='test',fout=None):

        if (d_type=='train'):
            self.train()
        else:
            self.eval()
        tr_like = 0; tr_acc=0; tr_l2=0
        ii = np.arange(0, trainX.shape[0], 1)
        # if (type=='train'):
        #   np.random.shuffle(ii)
        tr = trainX[ii]
        y = trainY[ii]

        for j in np.arange(0, len(y), self.bsz):
            data = torch.from_numpy(tr[j:j + self.bsz]).float().to(self.dv)
            target = torch.from_numpy(y[j:j + self.bsz]).long().to(self.dv)
            like, acc, l2l=self.compute_loss_and_grad(data,target,d_type)
            tr_like += like
            tr_acc+= acc
            tr_l2 = l2l

        tr_l2=tr_l2*self.lamda/self.bsz
        tr_acc=np.float(tr_acc)/len(tr)
        tr_like/=len(tr)
        if (np.mod(epoch, 10) == 9 or epoch==0):
            fout.write('====> Epoch {}: {} loss: {:.4f}, accuracy:{:.4f} \n'.format(d_type,
            epoch, tr_like+self.lamda*tr_l2,tr_acc))


def train_new(model,args,train,test,device):

    fout=sys.stdout
    print("In from hidden number of training",train[0].shape[0])
    trX=train[0]
    trY=train[1]
    print('In train new:')
    print(str(args))
    val = None
    ncl=np.int32(np.max(trY)+1)
    net=NET(trX.shape[1],ncl,args,device).to(device)
    tot_pars = 0
    for keys, vals in net.state_dict().items():
        fout.write(keys + ',' + str(np.array(vals.shape)) + '\n')
        tot_pars += np.prod(np.array(vals.shape))
    fout.write('tot_pars for fc,' + str(tot_pars) + '\n')
    scheduler=None
    for epoch in range(args.nepoch):
        if (scheduler is not None):
            scheduler.step()
        t1=time.time()
        net.run_epoch(trX,trY,epoch, d_type='train',fout=fout)
        if (val is not None):
                net.run_epoch(val,epoch, type='val',fout=fout)
        if (np.mod(epoch,10)==9 or epoch==0):
            fout.write('epoch: {0} in {1:5.3f} seconds\n'.format(epoch,time.time()-t1))
            fout.flush()

    teX=test[0]
    teY=test[1]
    net.run_epoch(teX, teY, 0, d_type='test', fout=fout)
    fout.flush()




