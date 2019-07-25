import torch
import sys
import models_opt_mix
from torch import nn, optim
import numpy as np
import time


class NET(nn.Module):

    def __init__(self, hdim, ncl,args,device):
        super(NET, self).__init__()

        self.hdim=hdim
        self.bsz=args.mb_size
        self.dv=device
        self.final = nn.Linear(hdim, ncl)
        self.optimizer = optim.Adam(self.parameters(),.01)



    def compute_loss_and_grad(self,data,target,type):

        if (type == 'train'):
            self.optimizer.zero_grad()
        logits = self.final(data)
        loss = nn.CrossEntropyLoss()
        cost = loss(logits, target)
        acc=torch.sum(torch.eq(torch.argmax(logits,dim=1),target))
        if (type == 'train'):
            cost.backward()
            self.optimizer.step()

        return cost, acc

    def run_epoch(self, trainX, trainY,epoch, type='test',fout=None):

        if (type=='train'):
            self.train()
        tr_loss = 0; tr_acc=0
        ii = np.arange(0, trainX.shape[0], 1)
        # if (type=='train'):
        #   np.random.shuffle(ii)
        tr = trainX[ii]
        y = trainY[ii]

        for j in np.arange(0, len(y), self.bsz):
            data = (tr[j:j + self.bsz]).float().to(self.dv)
            target = torch.from_numpy(y[j:j + self.bsz]).long().to(self.dv)
            cost, acc=self.compute_loss_and_grad(data,target,type)
            tr_loss += cost
            tr_acc+= acc
        fout.write('====> Epoch {}: {} loss: {:.4f}: accuracy:{:.4f}\n'.format(type,
        epoch, tr_loss / len(tr), np.float(tr_acc)/len(tr)))

def prepare_new(model,args,train,fout):


    trainMU, trainLOGVAR, trPI = model.initialize_mus(train[0], args.OPT)
    trainMU, trainLOGVAR, trPI = model.run_epoch(train, 0, args.nti, trainMU, trainLOGVAR, trPI,
                                                  type='trest',fout=fout)
    trmu=trainMU.detach().cpu().numpy()
    trlogvar=trainLOGVAR.detach().cpu().numpy()
    trpi=trPI.detach().cpu().numpy()
    trX=torch.cat([torch.tensor(trmu),torch.tensor(trlogvar),torch.tensor(trpi)],dim=1)
    trY=np.argmax(train[1],axis=1)

    return trX, trY


def train_new(model,args,train,test,device):

    fout=sys.stdout
    trX, trY=prepare_new(model,args,train,fout)

    val = None
    ncl=np.max(trY)+1
    net=NET(trX.shape[1],ncl,args,device).to(device)
    scheduler=None
    for epoch in range(args.nepoch):
        if (scheduler is not None):
            scheduler.step()
        t1=time.time()
        net.run_epoch(trX,trY,epoch, type='train',fout=fout)
        if (val is not None):
                net.run_epoch(val,epoch, type='val',fout=fout)

        fout.write('epoch: {0} in {1:5.3f} seconds\n'.format(epoch,time.time()-t1))
        fout.flush()

    teX,teY=prepare_new(model,args,test,fout)
    net.run_epoch(teX, teY, 0, type='test', fout=fout)
    fout.flush()




