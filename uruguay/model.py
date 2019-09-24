import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import h5py
import os
import sys
import argparse
import time
import aux

class CLEAN(nn.Module):
    def __init__(self, device, x_dim, y_dim, args):
        super(CLEAN, self).__init__()

        self.numc=args.num_char
        self.bsz=args.bsz
        self.x_dim=x_dim
        self.y_dim=y_dim
        self.full_dim=x_dim*y_dim
        self.dv=device
        self.ll=args.ll
        ll=len(args.filts)
        self.convs = nn.ModuleList([torch.nn.Conv2d(args.feats[i], args.feats[i+1],args.filts[i],stride=1,padding=np.int32(np.floor(args.filts[i]/2))) for i in range(ll)])
        self.l_out=None
        self.criterion=nn.CrossEntropyLoss(reduce=None)
        if (args.optimizer == 'Adam'):
            self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=args.lr)

    def forward_pre(self,input):

        out=input
        for cc in self.convs:
            out=cc(out)
            pp=torch.fmod(torch.tensor(out.shape),2)
            pool=nn.MaxPool2d(2,padding=tuple(pp[2:4]))
            out=pool(out)
            out=F.relu(out)
        return(out)

    def forward(self,input):
        first=False
        if (self.l_out is None):
            first=True
        out=self.forward_pre(input)

        if (first):
            sh2 = out.shape[3]
            sh1 = out.shape[2]
            sh2a = np.int32(np.ceil(sh2 / 5))
            pad = sh2a * 5 - sh2
            print('pre final shape',out.shape,sh1,sh2a)
            self.l_out=torch.nn.Conv2d(out.shape[1],args.ll,[sh1,sh2a],stride=[1,sh2a],padding=[0,pad]).to(self.dv)
        out=self.l_out(out)
        if (first):
            print('final shape',out.shape)

        return(out)

    def get_acc(self,out,targ):

        v,mx=torch.max(out,1)
        targa=targ[targ>0]
        mxa=mx[targ>0]
        acc=torch.sum(mx.eq(targ))
        acca=torch.sum(mxa.eq(targa))
        numa=mxa.shape[0]
        return acc, mx, acca, numa

    def loss_and_grad(self, input, target, type='train'):

        out=self.forward(input)

        if (type == 'train'):
            self.optimizer.zero_grad()
        loss=self.criterion(out.permute(1,0,2,3).reshape([self.ll,-1]).transpose(0,1),target.reshape(-1))
        acc,mx, acca, numa=self.get_acc(out.permute(1,0,2,3).reshape([self.ll,-1]).transpose(0,1),target.reshape(-1))
        if (type == 'train'):
            loss.backward()
            self.optimizer.step()

        return loss, acc, mx, acca, numa

    def run_epoch(self, train, text, epoch, fout, type):

        if (type=='train'):
            self.train()
        num_tr=train.shape[0]
        ii = np.arange(0, num_tr, 1)
        if (type=='train'):
           np.random.shuffle(ii)
        trin=train[ii,:,0:self.x_dim,:]
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

            loss, acc, mx, acca, numa= self.loss_and_grad(data, target, type)
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

def make_boxes(bx,td):
    standard_size = (35, 150)
    boxes=[]
    for b,tr in zip(bx,td):
        a=np.zeros(standard_size)
        a[0:np.int32(b[1]),0:np.int32(b[0])]=1
        #a[tr[0,standard_size[0]:,:]<.3]=2
        boxes+=[a]
    boxes=np.array(boxes)
    return boxes


def get_data(args):
    with h5py.File('pairs.hdf5', 'r') as f:
        #key = list(f.keys())[0]
        # Get the data
        pairs = f['PAIRS']
        print('tr', pairs.shape)
        all_pairs=np.float32(pairs)/255.
        all_pairs=all_pairs[0:args.num_train]
        all_pairs=all_pairs.reshape(-1,1,all_pairs.shape[1],all_pairs.shape[2])
        lltr=np.int32(np.ceil(.8*len(all_pairs))//args.bsz *args.bsz)
        llte=np.int32((len(all_pairs)-lltr)//args.bsz * args.bsz)
        ii=np.array(range(lltr+llte))
        np.random.shuffle(ii)
        bx=np.float32(f['BOXES'])
        boxes=make_boxes(bx,all_pairs)
        train_data = all_pairs[ii[0:lltr]]
        train_data_boxes=boxes[ii[0:lltr]]
        test_data=all_pairs[ii[lltr:lltr+llte]]
        test_data_boxes=boxes[ii[lltr:lltr+llte]]
    with open('texts.txt','r') as f:
        TEXT = [line.rstrip() for line in f.readlines()]
        aa=sorted(set(' '.join(TEXT)))
        print(aa)
        global ll
        if (' ' in aa):
            ll=len(aa)
            spa=0
        else:
            ll=len(aa)+1
            spa=ll-1
        train_t=[TEXT[j] for j in ii[0:lltr]]
        test_t=[TEXT[j] for j in ii[lltr:lltr+llte]]
        train_text=np.ones((len(train_t),5))*spa
        for j,tt in enumerate(train_t):
            for i,ss in enumerate(tt):
                train_text[j,i]=aa.index(ss)
        test_text=np.ones((len(test_t),5))*spa
        for j,tt in enumerate(test_t):
            for i,ss in enumerate(tt):
                test_text[j,i]=aa.index(ss)
        train_text=np.int32(train_text)
        test_text=np.int32(test_text)
        print("hello")
        args.aa=aa

    return train_data, train_data_boxes, train_text, test_data, test_data_boxes, test_text

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
train_data, train_data_boxes, train_text, test_data, test_data_boxes, test_text = get_data(args)
args.ll=ll
fout.write('num train '+str(train_data.shape[0])+'\n')
fout.write('num test '+str(test_data.shape[0])+'\n')

x_dim=np.int32(train_data[0].shape[1]/2)
y_dim=train_data[0].shape[2]

model=CLEAN(device,x_dim, y_dim, args).to(device)
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
    model.run_epoch(train_data, train_text, epoch,fout, 'train')
    model.run_epoch(test_data, test_text, epoch,fout, 'test')

    fout.write('epoch: {0} in {1:5.3f} seconds\n'.format(epoch,time.time()-t1))
    fout.flush()

rx=model.run_epoch(test_data, test_text, epoch,fout, 'test')
rxx=np.int32(np.array(rx)).ravel()
tt=np.array([args.aa[i] for i in rxx]).reshape(len(test_text),5)
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