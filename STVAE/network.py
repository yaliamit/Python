import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from Conv_data import rotate_dataset_rand
import contextlib
from torch_edges import Edge
from aux import create_img
import time
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
        self.edges=args.edges
        self.n_class=args.n_class
        #self.pools = args.pools # List of pooling at each level of network
        #self.drops=args.drops # Drop fraction at each level of network
        self.optimizer_type=args.optimizer
        self.lr=args.lr
        self.layer_text=layers
        self.lnti=lnti
        self.ed = Edge(self.dv, dtr=.03).to(self.dv)
        # The loss function
        self.criterion=nn.CrossEntropyLoss()
        self.criterion_shift=nn.CrossEntropyLoss()
        if (hasattr(args,'perturb')):
            self.perturb=args.perturb
        self.u_dim = 6
        self.idty = torch.cat((torch.eye(2), torch.zeros(2).unsqueeze(1)), dim=1)
        self.id = self.idty.expand((self.bsz,) + self.idty.size()).to(self.dv)

    def do_nonlinearity(self,ll,out):

        if ('non_linearity' not in ll):
            return(out)
        elif ('tanh' in ll['non_linearity']):
            return(F.tanh(out))
        elif ('relu' in ll['non_linearity']):
            return(F.relu(out))

    def forward(self,input,everything=False):

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
                        stride = ll['pool_size']
                        if ('stride' in ll):
                            stride = ll['stride']
                        self.layers.add_module(ll['name'],nn.MaxPool2d(ll['pool_size'], stride=stride, padding=tuple(pp[2:4])).to(self.dv))
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
                 if ('final' in k or not self.del_last):
                 #if ('conv2' in k or 'dense1' in k):
                     print('TO optimizer',k,p.shape)
                     pp+=[p]
                 else:
                     p.requires_grad=False

            if (self.optimizer_type == 'Adam'):
                print('Optimizer Adam',self.lr)
                self.optimizer = optim.Adam(pp, lr=self.lr,weight_decay=self.wd)
            else:
                print('Optimizer SGD',self.lr)
                self.optimizer = optim.SGD(self.parameters(), lr=self.lr,weight_decay=self.wd)
        out1=[]

        if(everything):
            out1=OUTS
        elif (len(OUTS) > 3):
            out1 = OUTS[-3]
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

    def standardize(self,out):

        outa=out.reshape(out.shape[0],-1)#-torch.mean(out,dim=1).reshape(-1,1)
        sd = torch.sqrt(torch.sum(outa * outa, dim=1)).reshape(-1, 1)
        out_a = outa/(sd+.01)

        return out_a

    def get_embedd_loss(self,out0,out1,targ):

        out0a=self.standardize(out0)
        out1a=self.standardize(out1)
        COV=torch.mm(out0a,out1a.transpose(0,1))
        v = torch.diag(COV)
        lecov=torch.logsumexp(COV-torch.diag(v),dim=1)-v
        #COV = torch.mm(out1a, out1a.transpose(0, 1))
        #v = torch.diag(COV)
        #lecov += torch.logsumexp(COV - torch.diag(v), dim=1)
        #lecov=F.relu(1.-v)+torch.sum(F.relu(1+(COV-torch.diag(v))),dim=1)
        loss=torch.sum(lecov)
        ID=2.*torch.eye(out0.shape[0]).to(self.dv)-1.
        icov=ID*COV
        # ll=torch.log(1.+torch.exp(icov))
        # loss=torch.sum(-icov+ll)
        acc=torch.sum(icov>0)
        return loss,acc



    # GRADIENT STEP
    def loss_and_grad(self, input, target, d_type='train'):

        if (d_type == 'train'):
            self.optimizer.zero_grad()

        # Get output of network
        if type(input) is list:
            out0,ot0=self.forward(input[0])
            out1,ot1=self.forward(input[1])

            loss, acc = self.get_embedd_loss(out0,out1,target)
            # WW=0
            # for p in self.optimizer.param_groups[0]['params']:
            #     WW+=torch.sum(p*p)
            # print(loss,WW)
        else:
            out=self.forward(input)
            # Compute loss and accuracy
            loss, acc=self.get_acc_and_loss(out,target)

        # Perform optimizer step using loss as criterion
        if (d_type == 'train'):
            loss.backward()
            self.optimizer.step()

        return loss, acc

    def deform_data(self,x_in):
        h=x_in.shape[2]
        w=x_in.shape[3]
        nn=x_in.shape[0]
        u=(torch.rand(nn,6)*self.perturb).to(self.dv)
        self.theta = u.view(-1, 2, 3) + self.id
        grid = F.affine_grid(self.theta, x_in[:,0,:,:].view(-1, h, w).unsqueeze(1).size())
        x_out=F.grid_sample(x_in,grid,padding_mode='reflection')
        #oo=x_out.reshape(self.bsz,-1)
        #print('MIN',torch.min(torch.sqrt(torch.sum(oo*oo,dim=1)/oo.shape[1])))
        return x_out


    def get_binary_signature(self,inp1, inp2=None, level=1):

        num_tr1=inp1[0].shape[0]
        OT1=[];
        with torch.no_grad():
            for j in np.arange(0, num_tr1, self.bsz, dtype=np.int32):
                data=(torch.from_numpy(inp1[0][j:j + self.bsz]).float()).to(self.dv)
                out,ot1=self.forward(data,everything=True)
                OT1+=[ot1[level]]
            OT1 = torch.cat(OT1)
            qq1=2*(OT1.reshape(num_tr1,-1)>0).type(torch.float32)-1.

            if inp2 is not None:
                OT2 = []
                num_tr2=inp2[0].shape[0]
                for j in np.arange(0, num_tr2, self.bsz, dtype=np.int32):
                    data = (torch.from_numpy(inp2[0][j:j + self.bsz]).float()).to(self.dv)
                    out, ot2 = self.forward(data, everything=True)
                    OT2 += [ot2[level]]
                OT2=torch.cat(OT2)
                qq2=2*(OT2.reshape(num_tr2,-1)>0).type(torch.float32)-1.
            else:
                qq2=qq1

            cc=torch.mm(qq1,qq2.transpose(0,1))/qq1.shape[1]
            if inp2 is None:
                cc-=torch.diag(torch.diag(cc))
            ##print('CC',torch.sum(cc==1.).type(torch.float32)/num_tr1)
            return cc

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
        # if (self.embedd):
        #     np.random.shuffle(ii)
        #     trin_def=self.deform_data(trin[0:self.bsz])
        #     #trin_def=rotate_dataset_rand(trin.transpose(0,2,3,1),20,0.5).transpose(0,3,1,2)
        #     train_new_a=trin
        #     train_new_b=trin_def
        #     targ=np.zeros(num_tr)


        full_loss=0; full_acc=0;
        # Loop over batches.

        targ_in=targ

        for j in np.arange(0, num_tr, jump,dtype=np.int32):
            if self.embedd:

                with torch.no_grad():
                    data_in=(torch.from_numpy(trin[j:j + jump]).float()).to(self.dv)
                    data_out1=self.deform_data(data_in)
                    #data_out2=self.deform_data(data_in)
                    #print('DIFF',torch.max(torch.abs(data_out1-data_out2)))
                    data=[data_in,data_out1]

                #data=[(torch.from_numpy(train_new_a[j:j+jump]).float()).to(self.dv),(torch.from_numpy(train_new_b[j:j+jump]).float()).to(self.dv)]
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

                OUT+=[self.forward(data)[1]]

        OUTA=torch.cat(OUT,dim=0)

        return OUTA

    def get_scheduler(self,args):
        scheduler = None
        if args.sched>0.:
            l2 = lambda epoch: pow((1. - 1. * epoch / args.nepoch), args.sched)
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=l2)

        return scheduler



