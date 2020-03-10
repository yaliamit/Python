import torch
from Conv_data import get_data
import eagerpy as ep

class Edge(torch.nn.Module):
    def __init__(self, device, ntr=4, dtr=0):
        super(Edge, self).__init__()
        self.ntr = ntr
        self.dtr = dtr
        self.dv = device

    def forward(self, x):
        x = self.pre_edges(x).to(self.dv)
        return x


    def pre_edges(self, im):

        EDGES=[]
        for k in range(im.shape[1]):
            EDGES+=[self.get_edges(im[:,k,:,:])]

        ED=torch.cat(EDGES,dim=1)

        return ED

    def get_edges(self,im):

        sh=im.shape
        delta=3
        im_b=torch.ones((sh[0],sh[1]+2*delta,sh[2]+2*delta)).to(self.dv)

        im_b[:,delta:delta+sh[1],delta:delta+sh[2]]=im

        diff_11 = torch.roll(im_b,(1,1),dims=(1,2))-im_b
        diff_nn11 = torch.roll(im_b, (-1, -1) ,dims=(1,2)) - im_b

        diff_01 = torch.roll(im_b,(0,1), dims=(1,2))-im_b
        diff_n01 = torch.roll(im_b,(0,-1),dims=(1,2))-im_b
        diff_10 = torch.roll(im_b,(1,0), dims=(1,2))-im_b
        diff_n10 = torch.roll(im_b,(-1,0),dims=(1,2))-im_b
        diff_n11 = torch.roll(im_b,(-1,1),dims=(1,2))-im_b
        diff_1n1 = torch.roll(im_b,(1,-1),dims=(1,2))-im_b

        thresh=self.ntr
        dtr=self.dtr
        ad_10=torch.abs(diff_10)
        ad_10=ad_10*(ad_10>dtr).float()
        e10a=torch.gt(ad_10,torch.abs(diff_01)).type(torch.uint8)\
             + torch.gt(ad_10,torch.abs(diff_n01)).type(torch.uint8) + torch.gt(ad_10,torch.abs(diff_n10)).type(torch.uint8)
        e10b=torch.gt(ad_10,torch.abs(torch.roll(diff_01,(1,0),dims=(1,2)))).type(torch.uint8)+\
                    torch.gt(ad_10, torch.abs(torch.roll(diff_n01, (1, 0), dims=(1, 2)))).type(torch.uint8)+\
                            torch.gt(ad_10,torch.abs(torch.roll(diff_01, (1, 0), dims=(1, 2)))).type(torch.uint8)
        e10 = torch.gt(e10a+e10b,thresh) & (diff_10>0)
        e10n =torch.gt(e10a+e10b,thresh) & (diff_10<0)

        ad_01 = torch.abs(diff_01)
        ad_01 = ad_01*(ad_01>dtr).float()
        e01a = torch.gt(ad_01, torch.abs(diff_10)).type(torch.uint8) \
               + torch.gt(ad_01, torch.abs(diff_n10)).type(torch.uint8) + torch.gt(ad_01, torch.abs(diff_n01)).type(torch.uint8)
        e01b = torch.gt(ad_01, torch.abs(torch.roll(diff_10, (0, 1), dims=(1, 2)))).type(torch.uint8) + \
                torch.gt(ad_01, torch.abs(torch.roll(diff_n10, (0, 1), dims=(1, 2)))).type(torch.uint8) +\
                    torch.gt(ad_01, torch.abs(torch.roll(diff_01, (0, 1), dims=(1, 2)))).type(torch.uint8)
        e01 = torch.gt(e01a + e01b, thresh) & (diff_01 > 0)
        e01n = torch.gt(e01a + e01b, thresh) & (diff_01 < 0)

        ad_11 = torch.abs(diff_11)
        ad_11 = ad_11*(ad_11>dtr).float()
        e11a = torch.gt(ad_11, torch.abs(diff_n11)).type(torch.uint8) \
               + torch.gt(ad_11, torch.abs(diff_1n1)).type(torch.uint8) + torch.gt(ad_11, torch.abs(diff_nn11)).type(torch.uint8)
        e11b = torch.gt(ad_11, torch.abs(torch.roll(diff_n11, (1, 1), dims=(1, 2)))).type(torch.uint8) + \
                torch.gt(ad_11, torch.abs(torch.roll(diff_1n1, (1, 1), dims=(1, 2)))).type(torch.uint8) + \
                    torch.gt(ad_11, torch.abs(torch.roll(diff_11, (1, 1), dims=(1, 2)))).type(torch.uint8)
        e11 = torch.gt(e11a + e11b, thresh) & (diff_11 > 0)
        e11n = torch.gt(e11a + e11b , thresh) & (diff_11 < 0)

        ad_n11 = torch.abs(diff_n11)
        ad_n11 = ad_n11*(ad_n11>dtr).float()
        en11a = torch.gt(ad_n11, torch.abs(diff_11)).type(torch.uint8) \
               + torch.gt(ad_n11, torch.abs(diff_1n1)).type(torch.uint8) + torch.gt(ad_n11, torch.abs(diff_nn11)).type(torch.uint8)
        en11b = torch.gt(ad_n11, torch.abs(torch.roll(diff_11, (-1, 1), dims=(1, 2)))).type(torch.uint8) + \
               torch.gt(ad_n11, torch.abs(torch.roll(diff_n11, (-1, 1), dims=(1, 2)))).type(torch.uint8) + \
               torch.gt(ad_n11, torch.abs(torch.roll(diff_n11, (-1, 1), dims=(1, 2)))).type(torch.uint8)
        en11 = torch.gt(en11a + en11b, thresh) & (diff_n11 > 0)
        en11n = torch.gt(en11a + en11b, thresh) & (diff_n11 < 0)

        edges=torch.zeros((im.shape[0],8,im.shape[1],im.shape[2])).to(self.dv)
        edges[:,0,2:sh[1],0:sh[2]]=e10[:,delta+2:delta+sh[1],delta:delta+sh[2]]
        edges[:,1,0:sh[1]-2,0:sh[2]]=e10n[:,delta:delta+sh[1]-2,delta:delta+sh[2]]
        edges[:,2,0:sh[1], 2:sh[2]] = e01[:, delta:delta + sh[1], delta+2:delta + sh[2]]
        edges[:,3,0:sh[1], 0:sh[2]-2] = e01n[:, delta:delta + sh[1], delta:delta + sh[2]-2]
        edges[:,4,2:sh[1], 2:sh[2]] = e11[:, delta + 2:delta + sh[1], delta+2:delta + sh[2]]
        edges[:,5,0:sh[1] - 2, 0:sh[2]-2] = e11n[:, delta:delta + sh[1] - 2, delta:delta + sh[2]-2]
        edges[:,6,0:sh[1]-2, 2:sh[2]] = en11[:, delta:delta + sh[1]-2, delta+2:delta + sh[2]]
        edges[:,7,2:sh[1], 0:sh[2]-2] = en11n[:, delta+2:delta + sh[1], delta:delta + sh[2]-2]

        return(edges)

