import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

import os
import sys
import argparse
import time
import aux

class CLEAN(nn.Module):
    def __init__(self, device, x_dim, y_dim, args):
        super(CLEAN, self).__init__()

        self.first=True
        self.lenc=args.lenc
        self.numc=args.num_char
        self.bsz=args.bsz
        self.x_dim=x_dim
        self.y_dim=y_dim
        self.full_dim=x_dim*y_dim
        self.dv=device
        self.ll=args.ll
        self.weights=torch.ones(self.ll).to(device)
        self.weights[0]=1.
        self.pools = args.pools
        self.drops=args.drops
        ll=len(args.filts)
        self.convs = nn.ModuleList([torch.nn.Conv2d(args.feats[i], args.feats[i+1],args.filts[i],stride=1,padding=np.int32(np.floor(args.filts[i]/2))) for i in range(ll)])
        self.criterion=nn.CrossEntropyLoss(weight=self.weights)
        self.criterion_shift=nn.CrossEntropyLoss(reduce=False)
        if (args.optimizer == 'Adam'):
            self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=args.lr)

    def forward_pre(self,input):

        out=input
        for i, cc in enumerate(self.convs):
            if (self.first):
                print(out.shape)
            out=cc(out)
            pp=torch.fmod(torch.tensor(out.shape),self.pools[i])
            if (self.pools[i]>1):
                pool=nn.MaxPool2d(self.pools[i],padding=tuple(pp[2:4]))
                out=pool(out)
            if (self.drops[i]<1.):
                out=nn.functional.dropout(out,self.drops[i])
            out=F.relu(out)
        return(out)

    def forward(self,input):

        out=self.forward_pre(input)
        if (self.drops[-1]<1.):
            if (self.first):
                self.dr2d=nn.Dropout2d(self.drops[-1])
            out=self.dr2d(out)
        if (self.first):
            self.sh=out.shape
            self.sh2a = np.int32(np.ceil(self.sh[3] / self.lenc))
            self.pad = self.sh2a * (self.lenc)+1 - self.sh[3]
            print('pre final shape',out.shape,self.sh[2],self.sh2a)
        out=torch.cat((out,torch.zeros(out.shape[0],self.sh[1],self.sh[2],self.pad).to(self.dv)),dim=3)
        if (self.first):
            self.l_out=torch.nn.Conv2d(out.shape[1],args.ll,[self.sh[2],self.sh2a+1],stride=[1,self.sh2a]).to(self.dv)
        out=self.l_out(out)
        if (self.first):
            print('final shape',out.shape)
            self.first=False

        return(out)



    def get_acc_and_loss(self,out,targ):

        v,mx=torch.max(out,1)
        targa=targ[targ>0]
        mxa=mx[targ>0]
        numa = targa.shape[0]
        loss=self.criterion(out,targ)
        acc=torch.sum(mx.eq(targ))
        acca=torch.sum(mxa.eq(targa))

        return loss, acc, acca, numa, mx

    def get_loss_shift(self,input_shift,target_shift, lst, fout, type):
        self.eval()
        num_tr=len(input_shift)
        num_tro=num_tr/lst
        full_loss=0; full_acc=0; full_acca=0; full_numa=0
        sh=np.array(input_shift.shape)
        sh[0]/=lst
        train_choice_shift=np.zeros(sh)
        rmx = []
        for j in np.arange(0, num_tr, self.bsz*lst):
            jo=np.int32(j/lst)
            sinput = torch.from_numpy(input_shift[j:j + self.bsz*lst]).float().to(self.dv)
            starg = torch.from_numpy(target_shift[j:j + self.bsz*lst]).to(self.dv)
            starg = starg.type(torch.int64)
            out = self.forward(sinput)
            loss=self.criterion_shift(out.permute(1,0,2,3).reshape([self.ll,-1]).transpose(0,1),starg.reshape(-1))

            sloss=torch.sum(loss.view(-1,self.lenc),dim=1).view(-1,lst)

            v,lossm=torch.min(sloss,1)
            ii=torch.arange(0,len(sinput),lst,dtype=torch.long).to(self.dv)+lossm
            outs=out[ii]
            stargs=starg[ii]
            loss, acc, acca, numa, mx =self.get_acc_and_loss(outs.permute(1,0,2,3).reshape([self.ll,-1]).transpose(0,1),stargs.reshape(-1))
            train_choice_shift[jo:jo+self.bsz]=sinput[ii].cpu().detach().numpy()
            full_loss += loss.item()
            full_acc += acc.item()
            full_acca += acca.item()
            full_numa += numa
            rmx += [mx.cpu().detach().numpy()]
        fout.write('====> Epoch {}: {} Full loss: {:.4F}, Full acc: {:.4F}, Non space acc: {:.4F}\n'.format(type+'_shift', epoch,
                        full_loss / (num_tro / self.bsz),full_acc / (num_tro * model.numc), full_acca / full_numa))

        return train_choice_shift, rmx


    def loss_and_grad(self, input, target, type='train'):

        out=self.forward(input)

        if (type == 'train'):
            self.optimizer.zero_grad()
        loss, acc, acca, numa, mx=self.get_acc_and_loss(out.permute(1,0,2,3).reshape([self.ll,-1]).transpose(0,1),target.reshape(-1))
        if (type == 'train'):
            loss.backward()
            self.optimizer.step()

        return loss, acc, acca, numa, mx

    def run_epoch(self, train, text, epoch, fout, type):

        if (type=='train'):
            self.train()
        else:
            self.eval()
        num_tr=train.shape[0]
        ii = np.arange(0, num_tr, 1)
        #if (type=='train'):
        #   np.random.shuffle(ii)
        trin=train[ii]
        targ=text[ii]

        full_loss=0
        full_acc=0
        full_acca=0
        full_numa=0
        rmx=[]
        for j in np.arange(0, num_tr, self.bsz):
            data = torch.from_numpy(trin[j:j + self.bsz]).float().to(self.dv)
            target = torch.from_numpy(targ[j:j + self.bsz]).to(self.dv)
            target=target.type(torch.int64)
            #target_boxes = torch.from_numpy(train_boxes[j:j+self.bsz]).float().to(self.dv)

            loss, acc, acca, numa, mx= self.loss_and_grad(data, target, type)
            full_loss += loss.item()
            full_acc += acc.item()
            full_acca+=acca.item()
            full_numa+=numa
            rmx+=[mx.cpu().detach().numpy()]
        fout.write('====> Epoch {}: {} Full loss: {:.4F}, Full acc: {:.4F}, Non space acc: {:.4F}\n'.format(type,epoch,
                    full_loss /(num_tr/self.bsz), full_acc/(num_tr*model.numc), full_acca/full_numa))

        return(rmx)


    def get_scheduler(self,args):
        scheduler = None
        if args.wd:
            l2 = lambda epoch: pow((1. - 1. * epoch / args.nepoch), 0.9)
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=l2)

        return scheduler



os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
    description='Variational Autoencoder with Spatial Transformation'
)


args=aux.process_args(parser)

use_gpu = args.gpu and torch.cuda.is_available()
if (use_gpu and not args.CONS):
    fout=open('OUT.txt','w')
else:
    args.CONS=True
    fout=sys.stdout

fout.write(str(args)+'\n')
args.fout=fout
fout.flush()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda:1" if use_gpu else "cpu")
fout.write('Device,'+str(device)+'\n')
fout.write('USE_GPU,'+str(use_gpu)+'\n')

ll=0
train_data, train_data_boxes, train_text, test_data, test_data_boxes, test_text = aux.get_data(args)



fout.write('num train '+str(train_data.shape[0])+'\n')
fout.write('num test '+str(test_data.shape[0])+'\n')

x_dim=np.int32(train_data[0].shape[1]/2)
y_dim=train_data[0].shape[2]
train_data=train_data[:, :, 0:x_dim, :]
test_data=test_data[:, :, 0:x_dim, :]
S = [0, 2, 4, 6]
T = [0, 4]
lst=len(S)*len(T)
train_data_shift=aux.add_shifts_new(train_data,S,T)
test_data_shift=aux.add_shifts_new(test_data,S,T)
train_text_shift=np.repeat(train_text,lst,axis=0)
test_text_shift=np.repeat(test_text,lst,axis=0)


model=CLEAN(device,x_dim, y_dim, args).to(device)
model.run_epoch(train_data[0:model.bsz], train_text, 0, fout, 'test')

tot_pars=0
for keys, vals in model.state_dict().items():
    fout.write(keys+','+str(np.array(vals.shape))+'\n')
    tot_pars+=np.prod(np.array(vals.shape))
fout.write('tot_pars,'+str(tot_pars)+'\n')

scheduler=model.get_scheduler(args)

for epoch in range(args.nepoch):
    if (scheduler is not None):
            scheduler.step()
    t1=time.time()
    train_data_choice_shift, _=model.get_loss_shift(train_data_shift,train_text_shift,lst,fout,'train')
    t2=time.time()
    fout.write('shift: in {:5.3f} seconds\n'.format(t2-t1))
    model.run_epoch(train_data_choice_shift, train_text, epoch,fout, 'train')
    t3=time.time()
    fout.write('epoch:  in {:5.3f} seconds\n'.format(t3-t2))
    model.get_loss_shift(test_data_shift, test_text_shift,lst,fout,'test')
    fout.write('test: in {:5.3f} seconds\n'.format(time.time()-t3))
    fout.write('epoch: {0} in {1:5.3f} seconds\n'.format(epoch,time.time()-t1))
    fout.flush()

test_data_choice_shift,rx=model.get_loss_shift(test_data_shift, test_text_shift,lst, fout,'test')
rxx=np.int32(np.array(rx)).ravel()
tt=np.array([args.aa[i] for i in rxx]).reshape(len(test_text),args.lenc)
aux.create_image(test_data,tt,model.x_dim,'try')

#model.show_recon(test_data[0:model.bsz],'test')
ex_file='MM'
if not os.path.isfile('_output'):
    os.system('mkdir _output')
torch.save(model.state_dict(),'_output/'+ex_file+'.pt')
fout.write("DONE\n")
fout.flush()

if (not args.CONS):
    fout.close()