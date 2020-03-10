import numpy as np
from Conv_data import get_data

def pre_edges(im,ntr=4,dtr=0):

    EDGES=[]
    for k in range(im.shape[3]):
        EDGES+=[get_edges(im[:,:,:,k],ntr,dtr)]

    ED=np.concatenate(EDGES,axis=3)
    ED=ED.to(self.dv)
    return ED

def get_edges(im,ntr=4,dtr=0):

    sh=im.shape
    delta=3
    im_b=np.ones((sh[0],sh[1]+2*delta,sh[2]+2*delta))

    im_b[:,delta:delta+sh[1],delta:delta+sh[2]]=im

    diff_11 = np.roll(im_b,(1,1),axis=(1,2))-im_b
    diff_nn11 = np.roll(im_b, (-1, -1) ,axis=(1,2)) - im_b

    diff_01 = np.roll(im_b,(0,1), axis=(1,2))-im_b
    diff_n01 = np.roll(im_b,(0,-1),axis=(1,2))-im_b
    diff_10 = np.roll(im_b,(1,0), axis=(1,2))-im_b
    diff_n10 = np.roll(im_b,(-1,0),axis=(1,2))-im_b
    diff_n11 = np.roll(im_b,(-1,1),axis=(1,2))-im_b
    diff_1n1 = np.roll(im_b,(1,-1),axis=(1,2))-im_b

    thresh=ntr
    ad_10=np.abs(diff_10)
    ad_10=ad_10*(ad_10>dtr)
    e10a=np.uint8(np.greater(ad_10,np.abs(diff_01)))\
         + np.uint8(np.greater(ad_10,np.abs(diff_n01))) + np.uint8(np.greater(ad_10,np.abs(diff_n10)))
    e10b=np.uint8(np.greater(ad_10,np.abs(np.roll(diff_01,(1,0),axis=(1,2)))))+\
                np.uint8(np.greater(ad_10, np.abs(np.roll(diff_n01, (1, 0), axis=(1, 2)))))+\
                        np.uint8(np.greater(ad_10,np.abs(np.roll(diff_01, (1, 0), axis=(1, 2)))))
    e10 = np.logical_and(np.greater(e10a+e10b,thresh),  diff_10>0)
    e10n =np.logical_and(e10a+e10b > thresh,  diff_10<0)

    ad_01 = np.abs(diff_01)
    ad_01 = ad_01*(ad_01>dtr)
    e01a = np.uint8(np.greater(ad_01, np.abs(diff_10))) \
           + np.uint8(np.greater(ad_01, np.abs(diff_n10))) + np.uint8(np.greater(ad_01, np.abs(diff_n01)))
    e01b = np.uint8(np.greater(ad_01, np.abs(np.roll(diff_10, (0, 1), axis=(1, 2))))) + \
           np.uint8(np.greater(ad_01, np.abs(np.roll(diff_n10, (0, 1), axis=(1, 2))))) + \
           np.uint8(np.greater(ad_01, np.abs(np.roll(diff_01, (0, 1), axis=(1, 2)))))
    e01 = np.logical_and(np.greater(e01a + e01b, thresh), diff_01 > 0)
    e01n = np.logical_and(e01a + e01b > thresh, diff_01 < 0)

    ad_11 = np.abs(diff_11)
    ad_11 = ad_11*(ad_11>dtr)
    e11a = np.uint8(np.greater(ad_11, np.abs(diff_n11))) \
           + np.uint8(np.greater(ad_11, np.abs(diff_1n1))) + np.uint8(np.greater(ad_11, np.abs(diff_nn11)))
    e11b = np.uint8(np.greater(ad_11, np.abs(np.roll(diff_n11, (1, 1), axis=(1, 2))))) + \
           np.uint8(np.greater(ad_11, np.abs(np.roll(diff_1n1, (1, 1), axis=(1, 2))))) + \
           np.uint8(np.greater(ad_11, np.abs(np.roll(diff_11, (1, 1), axis=(1, 2)))))
    e11 = np.logical_and(np.greater(e11a + e11b, thresh), diff_11 > 0)
    e11n = np.logical_and(e11a + e11b > thresh, diff_11 < 0)

    ad_n11 = np.abs(diff_n11)
    ad_n11 = ad_n11*(ad_n11>dtr)
    en11a = np.uint8(np.greater(ad_n11, np.abs(diff_11))) \
           + np.uint8(np.greater(ad_n11, np.abs(diff_1n1))) + np.uint8(np.greater(ad_n11, np.abs(diff_nn11)))
    en11b = np.uint8(np.greater(ad_n11, np.abs(np.roll(diff_11, (-1, 1), axis=(1, 2))))) + \
           np.uint8(np.greater(ad_n11, np.abs(np.roll(diff_n11, (-1, 1), axis=(1, 2))))) + \
           np.uint8(np.greater(ad_n11, np.abs(np.roll(diff_n11, (-1, 1), axis=(1, 2)))))
    en11 = np.logical_and(np.greater(en11a + en11b, thresh), diff_n11 > 0)
    en11n = np.logical_and(en11a + en11b > thresh, diff_n11 < 0)

    edges=np.zeros((im.shape[0],im.shape[1],im.shape[2],8))
    edges[:,2:sh[1],0:sh[2],0]=e10[:,delta+2:delta+sh[1],delta:delta+sh[2]]
    edges[:,0:sh[1]-2,0:sh[2],1]=e10n[:,delta:delta+sh[1]-2,delta:delta+sh[2]]
    edges[:, 0:sh[1], 2:sh[2], 2] = e01[:, delta:delta + sh[1], delta+2:delta + sh[2]]
    edges[:, 0:sh[1], 0:sh[2]-2, 3] = e01n[:, delta:delta + sh[1], delta:delta + sh[2]-2]
    edges[:, 2:sh[1], 2:sh[2], 4] = e11[:, delta + 2:delta + sh[1], delta+2:delta + sh[2]]
    edges[:, 0:sh[1] - 2, 0:sh[2]-2, 5] = e11n[:, delta:delta + sh[1] - 2, delta:delta + sh[2]-2]
    edges[:, 0:sh[1]-2, 2:sh[2], 6] = en11[:, delta:delta + sh[1]-2, delta+2:delta + sh[2]]
    edges[:, 2:sh[1], 0:sh[2]-2, 7] = en11n[:, delta+2:delta + sh[1], delta:delta + sh[2]-2]

    return(edges)

