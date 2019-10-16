import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import Shift


# Network module
class CLEAN(nn.Module):
    def __init__(self, device, x_dim, y_dim, lst, args):
        super(CLEAN, self).__init__()

        self.first=True
        self.lenc=args.lenc # Maximal number of characters in string. All strings padded with spaces to that length
        self.bsz=args.bsz # Batch size - gets multiplied by number of shifts so needs to be quite small.
        self.x_dim=x_dim # Dimensions of all images.
        self.y_dim=y_dim
        self.full_dim=x_dim*y_dim
        self.dv=device
        self.ll=args.ll # Number of possible character labels.
        self.lst=lst
        self.weights=torch.ones(self.ll).to(device)
        self.weights[0]=1.
        self.select=args.select
        self.pools = args.pools # List of pooling at each level of network
        self.drops=args.drops # Drop fraction at each level of network
        ff=len(args.filts) # Number of filters = number of conv layers.
        # Create list of convolution layers with the appropriate filter size, number of features etc.
        self.convs = nn.ModuleList([torch.nn.Conv2d(args.feats[i], args.feats[i+1],
                                                    args.filts[i],stride=1,padding=np.int32(np.floor(args.filts[i]/2)))
                                    for i in range(ff)])
        if (self.select):
            self.sel_len=self.lst*x_dim*y_dim
            self.seln=nn.Linear(self.sel_len,self.lst)

        # The loss function
        self.criterion=nn.CrossEntropyLoss()
        self.criterion_shift=nn.CrossEntropyLoss(reduce=False)
        if (args.optimizer == 'Adam'):
            self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=args.lr)

    # Apply sequence of conv layers up to the final one that will be determined later.
    def forward_pre(self,input):

        weights=None
        out=input
        if (self.first):
            self.pool_layers=[]

        # if (self.select):
        #     #ind = torch.range(0, input.shape[0] - 1, self.lst, dtype=torch.long).to(self.dv)
        #     tmp=input.view(-1,self.sel_len)
        #     tmp = nn.functional.dropout(tmp, .5)
        #     tmp = self.seln(tmp)
        #    weights=torch.softmax(tmp.view(-1,self.lst),dim=1)
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
            if (i==(self.select-1)):
                if (self.first):
                    self.seln = nn.Linear(torch.prod(torch.tensor(out.shape[1:4]))*self.lst, self.lst).to(dv)
                tmp=out.reshape(-1,self.lst,out.shape[1],out.shape[2],out.shape[3])
                tmp=tmp.reshape(tmp.shape[0],-1)
                tmp=nn.functional.dropout(tmp,.5)
                weights=torch.softmax(self.seln(tmp),dim=1)
        return(out,weights)

    # Full network
    def forward(self,input):

        out, weights=self.forward_pre(input)

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

        if (self.select):
            out_t=out.reshape(-1,self.lst,out.shape[1]*out.shape[2]*out.shape[3])
            out = torch.sum(weights.unsqueeze(2) * out_t, dim=1).reshape(-1, out.shape[1], out.shape[2], out.shape[3])
        return(out)



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
        jump_tr=jump
        if self.select:
            jump_tr *= self.lst
            targ_in=targ[np.arange(0,targ.shape[0],self.lst,dtype=np.int32)]
            num_tr/=self.lst
        else:
            targ_in=targ
        for j in np.arange(0, num_tr, jump,dtype=np.int32):
            j_tr=np.int32(j);
            if (self.select):
                j_tr*=self.lst
            data = (torch.from_numpy(trin[j_tr:j_tr + jump_tr]).float()/255.).to(self.dv)
            target = torch.from_numpy(targ_in[j:j + jump]).to(self.dv)
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



