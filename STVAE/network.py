import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from Conv_data import rotate_dataset_rand
import contextlib
from aux import create_img
@contextlib.contextmanager
def dummy_context_mgr():
    yield None

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, dv, stride=1,pd=0):
        super(residual_block, self).__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.a=False
        self.conv1=torch.nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=pd).to(dv)
        self.bn1=torch.nn.BatchNorm2d(out_channels).to(dv)
        self.conv2=torch.nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0).to(dv)
        self.bn2=torch.nn.BatchNorm2d(out_channels).to(dv)

        if in_channels!=out_channels:
            self.conv1a=torch.nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=pd).to(dv)
            self.a=True

    def forward(self,inp):

        out=self.conv1(inp)
        out=self.bn1(out)
        out=F.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=F.relu(out)

        if self.a:
            inp=self.conv1a(inp)

        out+=inp

        return out

class residual_block_small(nn.Module):
    def __init__(self, in_channels, out_channels, dv, stride=1,pd=0):
        super(residual_block_small, self).__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.conv1=torch.nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=pd).to(dv)
        self.conv2=torch.nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0).to(dv)


    def forward(self,inp):

        out1=self.conv1(inp)
        out=self.conv2(out1)
        out+=out1

        return out




# Network module
class network(nn.Module):
    def __init__(self, device,  args, layers, lnti):
        super(network, self).__init__()

        self.wd=args.wd
        self.embedd=args.embedd
        self.del_last=args.del_last
        self.first=True
        self.bsz=args.mb_size # Batch size - gets multiplied by number of shifts so needs to be quite small.
        #self.full_dim=args.full_dim
        self.dv=device
        self.n_class=args.n_class
        #self.pools = args.pools # List of pooling at each level of network
        #self.drops=args.drops # Drop fraction at each level of network
        self.optimizer_type=args.optimizer
        self.lr=args.lr
        self.layer_text=layers
        self.lnti=lnti

        # The loss function
        self.criterion=nn.CrossEntropyLoss()
        self.criterion_shift=nn.CrossEntropyLoss()

    def do_nonlinearity(self,ll,out):

        if ('non_linearity' not in ll):
            return(out)
        elif ('tanh' in ll['non_linearity']):
            return(F.tanh(out))
        elif ('relu' in ll['non_linearity']):
            return(F.relu(out))

    def forward(self,input):
        out = input
        in_dims=[]
        if (self.first):
            self.layers = nn.ModuleList()
        OUTS=[]

        for i,ll in enumerate(self.layer_text):
                inp_ind = i - 1
                if ('parent' in ll):
                    pp=ll['parent']
                    # over ride default inp_feats
                    if len(pp)==1:
                        inp_ind=self.lnti[pp[0]]
                        inp_feats=self.layer_text[inp_ind]['num_filters']
                        in_dim=in_dims[inp_ind]
                    else:
                        inp_feats=[]
                        loc_in_dims=[]
                        inp_ind=[]
                        for p in pp:
                            inp_ind += [self.lnti[p]]
                            inp_feats+=[self.layer_text[inp_ind[-1]]['num_filters']]
                            loc_in_dims+=[in_dims[inp_ind[-1]]]
                if ('input' in ll['name']):
                    OUTS+=[input]

                if ('conv' in ll['name']):
                    if self.first:
                        pd=tuple(np.int32(np.floor(np.array(ll['filter_size'])/2)))
                        self.layers.add_module(ll['name'],torch.nn.Conv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=1,padding=pd).to(self.dv))
                        #self.layers+=[torch.nn.Conv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=1,padding=pd).to(self.dv)]
                    out=getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS += [self.do_nonlinearity(ll,out)]

                if ('pool' in ll['name']):
                    if self.first:
                        pp = np.mod(out.shape, ll['pool_size'])
                        self.layers.add_module(ll['name'],nn.MaxPool2d(ll['pool_size'], padding=tuple(pp[2:4])).to(self.dv))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS += [out]

                if ('drop' in ll['name']):
                    if self.first:
                        self.layers.add_module(ll['name'],torch.nn.Dropout(p=ll['drop'], inplace=False).to(self.dv))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS += [out]
                    #OUTS += [self.layers[i-1](OUTS[inp_ind])]

                if ('dense' in ll['name']):
                    if self.first:
                        out_dim=ll['num_units']
                        bis=True
                        if ('nb' in ll):
                            bis=False
                        self.layers.add_module(ll['name'],nn.Linear(in_dim,out_dim,bias=bis).to(self.dv))
                    out=OUTS[inp_ind]
                    out = out.reshape(out.shape[0], -1)
                    out = getattr(self.layers, ll['name'])(out)
                    OUTS += [self.do_nonlinearity(ll, out)]
                if ('norm') in ll['name']:
                    if self.first:
                        if len(OUTS[-1].shape)==4:
                            self.layers.add_module(ll['name'],torch.nn.BatchNorm2d(OUTS[-1].shape[1]).to(self.dv))
                        else:
                            self.layers.add_module(ll['name'],torch.nn.BatchNorm1d(OUTS[-1].shape[1]).to(self.dv))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS += [out]
                if ('res' in ll['name']):
                    if self.first:
                        pd=tuple(np.int32(np.floor(np.array(ll['filter_size'])/2)))
                        self.layers.add_module(ll['name'],residual_block(inp_feats,ll['num_filters'],self.dv,stride=1,pd=pd))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS += [self.do_nonlinearity(ll, out)]
                if ('opr' in ll['name']):
                    if 'add' in ll['name']:
                        #self.layers.add_module += [torch.nn.Identity()]
                        out = OUTS[inp_ind[0]]+OUTS[inp_ind[1]]
                        #out = self.layers[i-1](out)
                        OUTS+=[out]
                        inp_feats=out.shape[1]
                if ('num_filters' in ll):
                    inp_feats = ll['num_filters']
                if self.first:
                    print(ll['name'],OUTS[-1].shape)
                in_dim = np.prod(OUTS[-1].shape[1:])
                in_dims+=[in_dim]

        out = OUTS[-1]
        if self.first:
            tot_pars = 0
            KEYS=[]
            for keys, vals in self.state_dict().items():
                print(keys + ',' + str(np.array(vals.shape)))
                if 'running' not in keys and 'tracked' not in keys:
                    KEYS+=[keys]
                tot_pars += np.prod(np.array(vals.shape))
            print('tot_pars,' + str(tot_pars))
            self.first = False
            # TEMPORARY
            pp=[]
            for k,p in zip(KEYS,self.parameters()):
                 if ('final' not in k or not self.del_last):
                 #if ('conv2' in k or 'dense1' in k):
                     print('TO optimizer',k,p.shape)
                     pp+=[p]
                 else:
                     p.requires_grad=False

            if (self.optimizer_type == 'Adam'):
                print('Optimizer Adam')
                self.optimizer = optim.Adam(pp, lr=self.lr,weight_decay=self.wd)
            else:
                print('Optimizer SGD')
                self.optimizer = optim.SGD(self.parameters(), lr=self.lr,weight_decay=self.wd)

        return(out,OUTS[-2])



        # Get loss and accuracy (all characters and non-space characters).
    def get_acc_and_loss(self, out, targ):
            v, mx = torch.max(out, 1)
            # Non-space characters
            # Total loss
            loss = self.criterion(out, targ)
            # total accuracy
            acc = torch.sum(mx.eq(targ))
            return loss, acc

    def get_embedd_loss(self,out0,out1,targ):

        out0 #-=torch.mean(out0,dim=1).reshape(-1,1)
        out1 #-=torch.mean(out1,dim=1).reshape(-1,1)
        sd0=torch.sqrt(torch.sum(out0*out0,dim=1)).reshape(-1,1)
        sd1=torch.sqrt(torch.sum(out1*out1,dim=1)).reshape(-1,1)
        cors=.01*torch.sum(out0*out1/(sd0*sd1),dim=1)
        ll=torch.log(1.+torch.exp(cors))
        tcors=targ.type(torch.float32)*cors
        loss=torch.sum(-tcors+ll)
        #cors=targ.type(torch.float32)*torch.sum(out0*out1,dim=1)
        #loss=torch.sum(F.relu(1-cors))
        #loss=torch.sum(torch.log(1+torch.exp(-2*cors)))
        acc=torch.sum((cors>0) & (targ>0))+torch.sum((cors<0) & (targ<=0))
        return loss, acc

    # GRADIENT STEP
    def loss_and_grad(self, input, target, d_type='train'):

        if (d_type == 'train'):
            self.optimizer.zero_grad()

        # Get output of network
        if type(input) is list:
            out0,_=self.forward(input[0])
            out1,_=self.forward(input[1])

            loss, acc = self.get_embedd_loss(out0,out1,target)
            # WW=0
            # for p in self.optimizer.param_groups[0]['params']:
            #     WW+=torch.sum(p*p)
            # print(loss,WW)
        else:
            out,_=self.forward(input)
            # Compute loss and accuracy
            loss, acc=self.get_acc_and_loss(out,target)

        # Perform optimizer step using loss as criterion
        if (d_type == 'train'):
            loss.backward()
            self.optimizer.step()

        return loss, acc

    # Epoch of network training
    def run_epoch(self, train, epoch, num_mu_iter=None, trainMU=None, trainLOGVAR=None, trPI=None, d_type='train', fout='OUT'):

        if (d_type=='train'):
            self.train()
        else:
            self.eval()
        num_tr=train[0].shape[0]
        ii = np.arange(0, num_tr, 1)
        #if (type=='train'):
        #   np.random.shuffle(ii)
        jump = self.bsz
        trin = train[0][ii]
        targ = train[2][ii]
        self.n_class = np.max(targ) + 1
        if (self.embedd):
            np.random.shuffle(ii)
            jumpd=np.int32(num_tr/2)
            trin_def=rotate_dataset_rand(trin.transpose(0,2,3,1),0,0).transpose(0,3,1,2)
            train_new_a=np.zeros_like(trin)
            train_new_b=np.zeros_like(trin)
            train_new_a[0:jumpd]=trin[ii[0:jumpd]]
            train_new_b[0:jumpd]=trin_def[ii[0:jumpd]]
            jj=ii[jumpd:].copy()
            np.random.shuffle(jj)
            train_new_a[jumpd:]=trin[ii[jumpd:]]
            train_new_b[jumpd:]=trin_def[jj]
            #imga = create_img(train_new_a, 1, 28, 28, 10, 10)
            #imgb = create_img(train_new_b, 1, 28, 28, 10, 10)
            targ=np.zeros(num_tr)
            targ[0:jumpd]=np.ones(jumpd)
            kk = np.arange(0, num_tr, 1)
            np.random.shuffle(kk)
            train_new_a=train_new_a[kk]
            train_new_b=train_new_b[kk]
            targ=targ[kk]
            #imga=create_img(train_new_a,1,28,28,10,10)
            #imgb=create_img(train_new_b,1,28,28,10,10)

        full_loss=0; full_acc=0;
        # Loop over batches.

        targ_in=targ

        for j in np.arange(0, num_tr, jump,dtype=np.int32):
            if self.embedd:
                data=[(torch.from_numpy(train_new_a[j:j+jump]).float()).to(self.dv),(torch.from_numpy(train_new_b[j:j+jump]).float()).to(self.dv)]
            else:
                data = (torch.from_numpy(trin[j:j + jump]).float()).to(self.dv)

            target = torch.from_numpy(targ_in[j:j + jump]).to(self.dv)
            target=target.type(torch.int64)
            with torch.no_grad() if (d_type!='train') else dummy_context_mgr():
                loss, acc= self.loss_and_grad(data, target, d_type)
            full_loss += loss.item()
            full_acc += acc.item()
        if (True): #np.mod(epoch,10)==9 or epoch<=10):
            fout.write('\n ====> Ep {}: {} Full loss: {:.4F}, Full acc: {:.4F} \n'.format(d_type,epoch,
                    full_loss /(num_tr/jump), full_acc/(num_tr)))

        return trainMU, trainLOGVAR, trPI

    def get_embedding(self, train):

        trin = train[0]
        jump = self.bsz
        num_tr = train[0].shape[0]
        self.eval()
        OUT=[]
        for j in np.arange(0, num_tr, jump, dtype=np.int32):
            data = (torch.from_numpy(trin[j:j + jump]).float()).to(self.dv)

            with torch.no_grad():
                OUT+=[self.forward(data)[0]]

        OUTA=torch.cat(OUT,dim=0)

        return OUTA

    def get_scheduler(self,args):
        scheduler = None
        if args.wd:
            l2 = lambda epoch: pow((1. - 1. * epoch / args.nepoch), 0.9)
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=l2)

        return scheduler



