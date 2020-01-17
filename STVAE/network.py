import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim


# Network module
class network(nn.Module):
    def __init__(self, device, x_dim, y_dim, args, layers):
        super(network, self).__init__()

        self.first=True
        self.bsz=args.mb_size # Batch size - gets multiplied by number of shifts so needs to be quite small.
        self.x_dim=x_dim # Dimensions of all images.
        self.y_dim=y_dim
        #self.full_dim=args.full_dim
        self.dv=device
        self.n_class=args.n_class
        #self.pools = args.pools # List of pooling at each level of network
        #self.drops=args.drops # Drop fraction at each level of network
        self.optimizer_type=args.optimizer
        self.lr=args.lr
        self.layer_text=layers


        # The loss function
        self.criterion=nn.CrossEntropyLoss()
        self.criterion_shift=nn.CrossEntropyLoss(reduce=False)

    def do_nonlinearity(self,ll,out):

        if ('non_linearity' not in ll):
            return(out)
        elif ('tanh' in ll['non_linearity']):
            return(F.tanh(out))
        elif ('relu' in ll['non_linearity']):
            return(F.relu(out))

    def forward(self,input):
        out = input
        if (self.first):
            self.layers = nn.ModuleList()
        if (len(out.shape)==4):
            inp_feats=input.shape[1]
        for i,ll in enumerate(self.layer_text):
                if ('conv' in ll['name']):
                    if self.first:
                        pd=tuple(np.int32(np.floor(np.array(ll['filter_size'])/2)))
                        self.layers+=[torch.nn.Conv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=1,padding=pd).to(self.dv)]
                        inp_feats=ll['num_filters']
                    out = self.layers[i](out)
                    out = self.do_nonlinearity(ll,out)
                if ('pool' in ll['name']):
                    if self.first:
                        pp = np.mod(out.shape, ll['pool_size'])
                        self.layers += [nn.MaxPool2d(ll['pool_size'], padding=tuple(pp[2:4])).to(self.dv)]
                    out = self.layers[i](out)
                if ('drop' in ll['name']):
                    if self.first:
                        self.layers += [torch.nn.Dropout(p=ll['drop'], inplace=False).to(self.dv)]
                    out = self.layers[i](out)
                if ('dense' in ll['name']):
                    if self.first:
                        in_dim=np.prod(out.shape[1:])
                        out_dim=ll['num_units']
                        self.layers+=[nn.Linear(in_dim,out_dim).to(self.dv)]
                    out = out.reshape(out.shape[0], -1)
                    out = self.layers[i](out)
                    out = self.do_nonlinearity(ll, out)
                if ('res' in ll['name']):
                    if self.first:
                        pd = np.int32np.floor(ll['filtersize'] / 2)
                        self.layers += [torch.nn.Conv2d(inp_feats, ll['num_filters'], ll['filter_size'], stride=1, padding=pd).to(self.dv)]
                    out_temp=self.layers[i](out)
                    out+=out_temp
                    out = self.do_nonlinearity(ll, out)
                    inp_feats = ll['num_filters']

        if self.first:
            self.first = False
            if (self.optimizer_type == 'Adam'):
                self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
            else:
                self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

        return(out)

    # Apply sequence of conv layers up to the final one that will be determined later.
    def forward_old(self,input):

        out=input
        if (self.first):
            self.pool_layers=[]
            #self.drop_layers=[]
        dr_i=0
        if (self.first):
            print(out.shape)
        for i, cc in enumerate(self.convs):

            out=cc(out)

            if (self.pools[i]>1):
                if (self.first):
                    pp = torch.fmod(torch.tensor(out.shape), self.pools[i])
                    self.pool_layers+=[nn.MaxPool2d(self.pools[i],padding=tuple(pp[2:4])).to(self.dv)]
                out=self.pool_layers[i](out)
            else:
                if (self.first):
                    self.pool_layers+=[None]
            out = F.relu(out)
            if (self.first):
                print(out.shape)
            if (self.first):
                    self.drop_layers+=[torch.nn.Dropout(p=self.drops[dr_i], inplace=False)]
            out=self.drop_layers[dr_i](out)

            dr_i+=1
            # Relu non-linearity at each level.


        if self.first:
            in_dim=out.shape[1]*out.shape[2]*out.shape[3]
            out_dim=self.full_dim
            self.pre_l_out=nn.Linear(in_dim,out_dim).to(self.dv)

        out = out.reshape(out.shape[0],-1)
        out=self.pre_l_out(out)
        out=F.relu(out)
        if self.first:
                self.drop_layers += [torch.nn.Dropout(p=self.drops[dr_i], inplace=False)]
        out = self.drop_layers[dr_i](out)

        dr_i += 1
        if self.first:
            self.l_out=nn.Linear(out_dim,self.n_class).to(self.dv)
            self.first=False
            if (self.optimizer_type == 'Adam'):
                self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
            else:
                self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        out=self.l_out(out)
        # Concatenate the padding
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
        full_loss=0; full_acc=0; full_acca=0; full_numa=0; full_accc=0
        rmx=[]
        # Loop over batches.
        jump=self.bsz
        targ_in=targ

        for j in np.arange(0, num_tr, jump,dtype=np.int32):
            data = (torch.from_numpy(trin[j:j + jump]).float()).to(self.dv)
            target = torch.from_numpy(targ_in[j:j + jump]).to(self.dv)
            target=target.type(torch.int64)

            loss, acc= self.loss_and_grad(data, target, d_type)
            full_loss += loss.item()
            full_acc += acc.item()
        if (np.mod(epoch,10)==9 or epoch==0):
            fout.write('\n ====> Epoch {}: {} Full loss: {:.4F}, Full acc: {:.4F} \n'.format(d_type,epoch,
                    full_loss /(num_tr/jump), full_acc/(num_tr)))

        return trainMU, trainLOGVAR, trPI

    def get_scheduler(self,args):
        scheduler = None
        if args.wd:
            l2 = lambda epoch: pow((1. - 1. * epoch / args.nepoch), 0.9)
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=l2)

        return scheduler



