import torch
import numpy as np

# Get loss for optimal shift (same as other loss)
def loss_shift(self,out,targ=None):


        outl=out.permute(1, 0, 2, 3).reshape([self.ll, -1]).transpose(0, 1)
        # poutl=torch.softmax(outl,dim=1)
        # v, mx = torch.max(poutl, dim=1)
        # PMX = torch.zeros_like(poutl)
        # PMX.scatter_(1, mx.reshape(-1, 1), 1)
        # poutl2=poutl-PMX*poutl
        # v2, mx2 = torch.max(poutl2,dim=1)
        # vv=v-v2
        # vv=torch.sum(vv.reshape(-1,self.lenc),dim=1)
        # vv=vv.reshape(-1,self.lst)
        # u,lossm=torch.max(vv,1)
        # MX = mx.reshape(-1, self.lenc)
        # ii = torch.arange(0, len(MX), self.lst, dtype=torch.int64).to(self.dv) + lossm
        # MSMX=MX[ii]
        # tot_loss=torch.tensor(0)
        #if (targ is None):
        v, mx=torch.max(outl,dim=1)
        MX=torch.zeros_like(outl)
        MX.scatter_(1,mx.reshape(-1,1),1)
        MX=MX.reshape(-1,self.lst,self.lenc,self.ll)
        SMX=torch.sum(MX,dim=1)
        VSMX, MSMX=torch.max(SMX,dim=2)
        spMX=MSMX[:,0]==0
        print(torch.sum(spMX))
        MSMX[spMX,0:self.lenc-1]=MSMX[spMX,1:self.lenc]
        MSMX[spMX,self.lenc-1]=0
        hhr = MSMX.repeat_interleave(self.lst, dim=0)
        loss = self.criterion_shift(outl, hhr.view(-1))
        slossa = torch.sum(loss.reshape(-1, self.lenc), dim=1).reshape(-1, self.lst)
        v, lossm = torch.min(slossa, 1)

        tot_loss=torch.mean(v)
        return lossm, tot_loss, MSMX





# Find optimal shift/scale for each image
def get_loss_shift(self,input_shift,target_shift, epoch, fout, type):
        self.eval()
        num_tr=len(input_shift)
        num_tro=num_tr/self.lst
        rmx = []
        # Loop over batches of training data each lst of them are transformation of same image.
        OUT=[]
        TS = (torch.from_numpy(target_shift)).type(torch.int64).to(self.dv)
        for j in np.arange(0, num_tr, self.bsz):
            # Data is stored as uint8 to save space. So transfer to float for gpu.
            sinput = (torch.from_numpy(input_shift[j:j + self.bsz]).float()/255.).to(self.dv)
            # Apply network
            out = self.forward(sinput)
            OUT+=[out]

        OUT=torch.cat(OUT,dim=0)

        # CHoose shift/scale based on labels obtained from a vote at each location.
        lossm=[]
        shift_loss=0
        jump=self.bsz*self.lst
        MSMX=[]
        for j in np.arange(0,num_tr,jump):
            lossmb, s_l, msmx=self.loss_shift(OUT[j:j+jump],target_shift[j:j+jump])
            lossm+=[lossmb]
            shift_loss+=s_l.item()
            MSMX+=[msmx]
        shift_loss/=(num_tr/jump)
        lossm=torch.cat(lossm,dim=0)
        MSMX=torch.cat(MSMX,dim=0).detach().cpu().numpy()
        ii=torch.arange(0,len(OUT),self.lst,dtype=torch.int64)+lossm.detach().cpu()

        outs=OUT[ii]
        stargs=TS[ii]

        # Get accuracy for chosen shift/scales
        outsp=outs.permute(1,0,2,3).reshape([self.ll,-1]).transpose(0,1).to(self.dv)
        target = (stargs.reshape(-1)).to(self.dv)
        loss, acc, acca, numa, accc, mx =self.get_acc_and_loss(outsp,target)

        # Extract best version of each image for the network training stage.
        train_choice_shift=(input_shift[ii])
        rmx += [mx.cpu().detach().numpy()]

        fout.write('====> {}: {} Full loss: {:.4F}\n'.format(type + '_shift', epoch,
                                                             shift_loss))
        fout.write(
            '====> Epoch {}: {} Full loss: {:.4F}, Full acc: {:.4F}, Non space acc: {:.4F}, case insensitive acc {:.4F}\n'.format(
                type, epoch,
                loss.item(), acc.item() / (num_tro * self.lenc), acca.item() / numa,
                accc.item() / (num_tro * self.lenc)))

        return train_choice_shift, rmx, MSMX