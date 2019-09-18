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


        self.bsz=args.bsz
        self.x_dim=x_dim
        self.y_dim=y_dim
        self.full_dim=x_dim*y_dim
        self.dv=device
        ll=len(args.filts)
        self.convs = nn.ModuleList([torch.nn.Conv2d(args.feats[i], args.feats[i+1],args.filts[i],stride=1,padding=np.int32(np.floor(args.filts[i]/2))) for i in range(ll)])
        self.l_out=torch.nn.Conv2d(args.feats[-1],1,args.filt_size_out,stride=1,padding=np.int32(np.floor(args.filt_size_out/2)))
        if (args.optimizer == 'Adam'):
            self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=args.lr)

    def forward(self,input):

        out=input
        for cc in self.convs:
            out=cc(out)
            out=F.relu(out)
        out=self.l_out(out)
        out=torch.sigmoid(out)
        return(out)

    def loss_and_grad(self, input, target, target_boxes, type='train'):

        out=self.forward(input)

        if (type == 'train'):
            self.optimizer.zero_grad()
        loss = F.binary_cross_entropy(out.squeeze().view(-1, self.full_dim), target.view(-1, self.full_dim), weight=target_boxes.view(-1,self.full_dim),reduction='mean')

        if (type == 'train'):
            loss.backward()
            self.optimizer.step()

        return loss, out

    def run_epoch(self, train, train_boxes, epoch, fout, type):

        if (type=='train'):
            self.train()
        num_tr=train.shape[0]
        ii = np.arange(0, num_tr, 1)
        if (type=='train'):
           np.random.shuffle(ii)
        trin=train[ii,:,0:self.x_dim,:]
        trat=train[ii,:,self.x_dim:,:]
        full_loss=0
        for j in np.arange(0, num_tr, self.bsz):
            data = torch.from_numpy(trin[j:j + self.bsz]).float().to(self.dv)
            target = torch.from_numpy(trat[j:j + self.bsz]).float().to(self.dv)
            target_boxes = torch.from_numpy(train_boxes[j:j+self.bsz]).float().to(self.dv)

            loss, out= self.loss_and_grad(data, target, target_boxes, type)
            full_loss += loss.item()

        fout.write('====> Epoch {}: {} Full loss: {:.4F}\n'.format(type,epoch,
                    full_loss /(num_tr/self.bsz)))

    def show_recon(self, train,type):

        num_tr = train.shape[0]
        trin = train[:, :, 0:self.x_dim, :]
        trat = train[:, :, self.x_dim:, :]
        full_loss = 0
        OUT=[]
        for j in np.arange(0, num_tr, self.bsz):
            data = torch.from_numpy(trin[j:j + self.bsz]).float().to(self.dv)
            target = torch.from_numpy(trat[j:j + self.bsz]).float().to(self.dv)

            out = self.forward(data)
            OUT=OUT+[out.detach().cpu().numpy()]

        OUTA=np.concatenate(OUT,axis=0).squeeze()
        aux.create_image(trin.squeeze(),trat.squeeze(),OUTA,'recon'+type)

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

def get_data():
    with h5py.File('pairs.hdf5', 'r') as f:
        #key = list(f.keys())[0]
        # Get the data

        pairs = f['PAIRS']
        print('tr', pairs.shape)
        all_pairs=np.float32(pairs)/255.
        all_pairs=all_pairs.reshape(-1,1,all_pairs.shape[1],all_pairs.shape[2])
        lltr=np.int32(np.ceil(.8*len(all_pairs))//args.bsz *args.bsz)
        llte=np.int32((len(all_pairs)-lltr)//args.bsz * args.bsz)
        ii=np.array(range(lltr+llte))
        #np.random.shuffle(ii)
        bx=np.float32(f['BOXES'])
        boxes=make_boxes(bx,all_pairs)
        train_data = all_pairs[ii[0:lltr]]
        train_data_boxes=boxes[ii[0:lltr]]
        test_data=all_pairs[ii[lltr:lltr+llte]]
        test_data_boxes=boxes[ii[lltr:lltr+llte]]

    return train_data, train_data_boxes, test_data, test_data_boxes

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

train_data, train_data_boxes, test_data, test_data_boxes = get_data()


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
    model.run_epoch(train_data, train_data_boxes, epoch,fout, 'train')
    model.run_epoch(test_data, test_data_boxes, epoch,fout, 'test')

    fout.write('epoch: {0} in {1:5.3f} seconds\n'.format(epoch,time.time()-t1))
    fout.flush()

model.show_recon(train_data[0:model.bsz],'train')

model.show_recon(test_data[0:model.bsz],'test')
ex_file='MM'
if not os.path.isfile('_output'):
    os.system('mkdir _output')
torch.save(model.state_dict(),'_output/'+ex_file+'.pt')
fout.write("DONE\n")
fout.flush()

if (not args.CONS):
    fout.close()