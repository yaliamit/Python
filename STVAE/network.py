import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

import contextlib
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
                        self.layers+=[torch.nn.Conv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=1,padding=pd).to(self.dv)]
                    out = self.layers[i-1](OUTS[inp_ind])

                    OUTS += [self.do_nonlinearity(ll,out)]

                if ('pool' in ll['name']):
                    if self.first:
                        pp = np.mod(out.shape, ll['pool_size'])
                        self.layers += [nn.MaxPool2d(ll['pool_size'], padding=tuple(pp[2:4])).to(self.dv)]
                    OUTS += [self.layers[i-1](OUTS[inp_ind])]

                if ('drop' in ll['name']):
                    if self.first:
                        self.layers += [torch.nn.Dropout(p=ll['drop'], inplace=False).to(self.dv)]
                    OUTS += [self.layers[i-1](OUTS[inp_ind])]

                if ('dense' in ll['name']):
                    if self.first:
                        out_dim=ll['num_units']
                        self.layers+=[nn.Linear(in_dim,out_dim).to(self.dv)]
                    out=OUTS[inp_ind]
                    out = out.reshape(out.shape[0], -1)
                    out = self.layers[i-1](out)
                    OUTS += [self.do_nonlinearity(ll, out)]
                if ('norm') in ll['name']:
                    if self.first:
                        if len(OUTS[-1].shape)==4:
                            self.layers+=[torch.nn.BatchNorm2d(OUTS[-1].shape[1]).to(self.dv)]
                        else:
                            self.layers += [torch.nn.BatchNorm1d(OUTS[-1].shape[1]).to(self.dv)]
                    OUTS+=[self.layers[i-1](OUTS[inp_ind])]
                if ('res' in ll['name']):
                    if self.first:
                        pd=tuple(np.int32(np.floor(np.array(ll['filter_size'])/2)))
                        # reslay=nn.Sequential(
                        #         torch.nn.Conv2d(inp_feats, ll['num_filters'], ll['filter_size'], stride=1, padding=pd).to(self.dv),
                        #         torch.nn.Conv2d(ll['num_filters'], ll['num_filters'], 1, stride=1, padding=0).to(self.dv)
                        # )
                        self.layers+=[residual_block_small(inp_feats,ll['num_filters'],self.dv,stride=1,pd=pd)]
                    out=self.layers[i-1](OUTS[inp_ind])
                    #out_temp=self.layers[i-1][0](OUTS[inp_ind])
                    #out_temp1=self.layers[i-1][1](out_temp)
                    #out=out_temp+out_temp1
                    OUTS += [self.do_nonlinearity(ll, out)]
                if ('opr' in ll['name']):
                    if 'add' in ll['name']:
                        self.layers += [torch.nn.Identity()]
                        out = OUTS[inp_ind[0]]+OUTS[inp_ind[1]]
                        out = self.layers[i-1](out)
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
            for keys, vals in self.state_dict().items():
                print(keys + ',' + str(np.array(vals.shape)))
                tot_pars += np.prod(np.array(vals.shape))
            print('tot_pars,' + str(tot_pars))
            self.first = False
            # TEMPORARY
            pp=[]
            for p in self.parameters():
                pp+=[p]
            if self.del_last:
                del pp[-2:]
            if (self.optimizer_type == 'Adam'):
                self.optimizer = optim.Adam(pp, lr=self.lr)
            else:
                self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

        return(out)



        # Get loss and accuracy (all characters and non-space characters).
    def get_acc_and_loss(self, out, targ):
            v, mx = torch.max(out, 1)
            # Non-space characters
            # Total loss
            loss = self.criterion(out, targ)
            # total accuracy
            acc = torch.sum(mx.eq(targ))


            return loss, acc

    # GRADIENT STEP
    def loss_and_grad(self, input, target, d_type='train'):

        # Get output of network
        out=self.forward(input)

        if (d_type == 'train'):
            self.optimizer.zero_grad()
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
        trin = train[0][ii]
        targ = train[2][ii]
        self.n_class=np.max(targ)+1
        full_loss=0; full_acc=0;
        # Loop over batches.
        jump=self.bsz
        targ_in=targ

        for j in np.arange(0, num_tr, jump,dtype=np.int32):
            data = (torch.from_numpy(trin[j:j + jump]).float()).to(self.dv)
            target = torch.from_numpy(targ_in[j:j + jump]).to(self.dv)
            target=target.type(torch.int64)
            with torch.no_grad() if (d_type!='train') else dummy_context_mgr():
                loss, acc= self.loss_and_grad(data, target, d_type)
            full_loss += loss.item()
            full_acc += acc.item()
        if (np.mod(epoch,10)==9 or epoch==0):
            fout.write('\n ====> Ep {}: {} Full loss: {:.4F}, Full acc: {:.4F} \n'.format(d_type,epoch,
                    full_loss /(num_tr/jump), full_acc/(num_tr)))

        return trainMU, trainLOGVAR, trPI

    def get_scheduler(self,args):
        scheduler = None
        if args.wd:
            l2 = lambda epoch: pow((1. - 1. * epoch / args.nepoch), 0.9)
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=l2)

        return scheduler



