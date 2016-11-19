from __future__ import print_function

import numpy as np
import os
import deepdish as dd
import mnist
import cifar
import scipy.ndimage
import pylab as py
import matplotlib.colors as col
import amitgroup as ag



def rotate_dataset_rand(X,angle=0,scale=0,shift=0,gr=0,flip=False,blur=False,saturation=False, spl=None):
    # angle=NETPARS['trans']['angle']
    # scale=NETPARS['trans']['scale']
    # #shear=NETPARS['trans']['shear']
    # shift=NETPARS['trans']['shift']
    s=np.shape(X)
    Xr=np.zeros(s)
    cent=np.array(s[2:4])/2
    angles=np.random.rand(Xr.shape[0])*angle-angle/2.
    SX=np.exp(np.random.rand(Xr.shape[0],2)*scale-scale/2.)
    SH=np.int32(np.round(np.random.rand(Xr.shape[0],2)*shift)-shift/2)
    FL=np.zeros(Xr.shape[0])
    BL=np.zeros(Xr.shape[0])
    if (flip):
        FL=(np.random.rand(Xr.shape[0])>.5)
    if (blur):
        BL=(np.random.rand(Xr.shape[0])>.5)
    if (saturation):
        HS=(np.power(2,np.random.rand(Xr.shape[0])*4-2))
        HU=((np.random.rand(Xr.shape[0]))-.5)*.2
    #SHR=np.random.rand(Xr.shape[0])*shear-shear/2.
    for i in range(Xr.shape[0]):
        if (np.mod(i,1000)==0):
            print(i,end=" ")
        mat=np.eye(2)
        #mat[1,0]=SHR[i]
        mat[0,0]=SX[i,0]
        mat[1,1]=SX[i,0]
        rmat=np.eye(2)
        a=angles[i]*np.pi/180.
        rmat[0,0]=np.cos(a)
        rmat[0,1]=-np.sin(a)
        rmat[1,0]=np.sin(a)
        rmat[1,1]=np.cos(a)
        mat=mat.dot(rmat)
        offset=cent-mat.dot(cent)+SH[i]
        for d in range(X.shape[1]):
            Xt=scipy.ndimage.interpolation.affine_transform(X[i,d],mat, offset=offset, mode='reflect')
            Xt=np.minimum(Xt,.99)
            if (FL[i]):
                Xt=np.fliplr(Xt)
            if (BL[i]):
                Xt=scipy.ndimage.gaussian_filter(Xt,sigma=.5)
            Xr[i,d,]=Xt
        if (HS[i]):

            y=col.rgb_to_hsv(Xr[i].transpose(1,2,0))
            y[:,:,1]=np.minimum(y[:,:,1]*HS[i],1)
            y[:,:,0]=np.mod(y[:,:,0]+HU[i],1.)
            z=col.hsv_to_rgb(y)
            Xr[i]=z.transpose(2,0,1)

    if (gr):
        fig1=py.figure(1)
        fig2=py.figure(2)
        ii=range(X.shape[0])
        np.random.shuffle(ii)
        nr=12
        nr2=nr*nr
        for j in range(nr2):
            #print(angles[i],SX[i],SH[i],FL[i],BL[i])
            py.figure(fig1.number)
            py.subplot(nr,nr,j+1)
            py.imshow(X[ii[j]].transpose(1,2,0))
            py.axis('off')
            py.figure(fig2.number)
            py.subplot(nr,nr,j+1)
            py.imshow(Xr[ii[j]].transpose(1,2,0))
            py.axis('off')
        py.show()
    print(end="\n")
    # if (spl is not None):
    #     xs=np.arange(0,s[2]-spl[1]+1,spl[0])
    #     ys=np.arange(0,s[3]-spl[1]+1,spl[0])
    #     XX=np.copy(Xr) #np.zeros(s)
    #     ii=np.ones(s)
    #     for x in xs:
    #         for y in ys:
    #             ss=(slice(None),slice(None),slice(x,x+spl[1]),slice(y,y+spl[1]))
    #             Xspl=Xr[ss]
    #             Xs=rotate_dataset_rand(Xspl,angle=0,scale=0,shift=shift/2,gr=gr)
    #             XX[ss]=XX[ss]+Xs
    #             ii[ss]=ii[ss]+1
    #     Xrr=XX/ii
    #     if (gr):
    #         for n in range(Xr.shape[0]):
    #             py.subplot(1,2,1)
    #             py.imshow(Xr[n].transpose(1,2,0))
    #             py.subplot(1,2,2)
    #             py.imshow(Xrr[n].transpose(1,2,0))
    #             py.show()
    return(np.float32(Xr))

def rotate_dataset(X,NETPARS):
    angle=NETPARS['trans']['angle']
    angle_step=NETPARS['trans']['angle_step']
    scale=NETPARS['trans']['scale']
    if (scale>0):
        scale_step=NETPARS['trans']['scale_step']
    sh=np.shape(X)
    cent=np.array(sh[2:4])/2
    if (angle>0):
        angles=np.arange(-angle,angle+1,angle_step)
    else:
        angles=(0,)
    if (scale>0):
        scales=np.exp(np.arange(-scale,scale+.01,scale_step))
    else:
        scales=(1,)
    Xr=np.zeros((sh[0],len(angles)*len(scales))+sh[1:])
    mats=[]
    offsets=[]
    for da in angles:
            for s in scales:
                mat=np.eye(2)
                rmat=np.eye(2)
                # mat[0,0]=s
                # mat[1,1]=s
                a=da*np.pi/180.
                rmat[0,0]=np.cos(a)
                rmat[0,1]=-np.sin(a)
                rmat[1,0]=np.sin(a)
                rmat[1,1]=np.cos(a)
                mat=mat.dot(rmat)
                offset=cent-mat.dot(cent)
                offsets.append(offset)
                mats.append(mat)
    for i in range(Xr.shape[0]):
        if (np.mod(i,1000)==0):
            print(i,end=" ")
        for t,mat in enumerate(mats):
                for c in range(sh[1]):
                    Xt=scipy.ndimage.interpolation.affine_transform(X[i,c],mat, offset=offsets[t], mode='reflect')
                    Xr[i,t,c]=np.minimum(Xt,.99)
                if (NETPARS['trans']['gr']):
                    print(a,s)
                    py.subplot(1,2,1)
                    py.imshow(X[i].transpose(1,2,0))
                    py.subplot(1,2,2)
                    py.imshow(Xr[i,t].transpose(1,2,0))
                    py.show()

    print(end="\n")
    return(Xr)


def  do_read_det(ss,num_train):
            Xtr=[]
            if (num_train==0):
                num_files=100
            else:
                num_files=np.int32(np.ceil(num_train/1111))
            tot=0
            for j in range(num_files+1):

                if (os.path.isfile('/Users/amit/Desktop/AMNEW/_CIFAR10/rot/cifar_10'+ss+'.rotate'+str(j)+'.h5')):
                    print(j, end=" ")
                    X=dd.io.load('/Users/amit/Desktop/AMNEW/_CIFAR10/rot/cifar_10'+ss+'.rotate'+str(j)+'.h5')
                    tot+=X.shape[0]
                    end=X.shape[0]
                    if (tot>num_train):
                        end=X.shape[0]+num_train-tot
                    Xtr.append(X[0:end])
            X_out=np.concatenate(Xtr,axis=0)
            X_out=np.float32(X_out)
            X_out=np.transpose(X_out,(1,0,2,3,4))
            X_out=list(X_out)
            return(X_out)

def do_rands(x,NETPARS,insert=False):
        Xtr=[]
        num_rand=1
        if 'num_rand' in NETPARS['trans']:
            num_rand=NETPARS['trans']['num_rand']
        for i in range(num_rand):
            Xtr.append(rotate_dataset_rand(x,angle=NETPARS['trans']['angle'],
                           scale=NETPARS['trans']['scale'],
                           shift=NETPARS['trans']['shift'],flip=NETPARS['trans']['flip'],blur=NETPARS['trans']['blur'],
                           saturation=NETPARS['trans']['saturation'],gr=NETPARS['trans']['gr'],spl=NETPARS['trans']['spl']))
        if (insert):
            Xtr.insert(np.int32(np.floor(num_rand/2)),x)
        if (len(Xtr)>1):
            X=Xtr
        else:
            X=np.float32(Xtr[0])

        return(X)

def do_det(x,NETPARS,ss):
        X=rotate_dataset(x,NETPARS)
        X=np.transpose(X,(1,0,2,3,4))
        X=np.float32(X)
        X=list(X)
        # num=np.int32(np.ceil(X.shape[0]*X.shape[1]/10000))
        # inc=np.int32(np.floor(10000/X.shape[1]))
        # for j in range(num+1):
        #     beg=j*inc
        #     end=np.minimum((j+1)*inc,X.shape[0])
        #     dd.io.save('cifar_10_'+ss+'.rotate'+str(j)+'.h5',X[beg:end])
        return(X)

def load_rotated_dataset(NETPARS,x_train,x_val,x_test, num_train=0):

    X_train=None
    X_val=None
    if (NETPARS['trans']['angle']==-1):
        X_train=do_read_det('train',NETPARS, num_train)
        X_val=do_read_det('val',NETPARS, num_train)
        X_test=do_read_det('test',NETPARS, num_train)
        return(X_train,X_val,X_test)
    else:
        angle_step=None
        if (num_train==0):
                num_train=x_train.shape[0]
        if ('angle_step' in NETPARS['trans']):
            angle_step=NETPARS['trans']['angle_step']
        # This is a deterministic range of perturbations write it to a file.
        if (angle_step is not None):
            if (x_train is not None):
                X_train=do_det(x_train[0:num_train],NETPARS,'train')
                X_val=x_val[0:np.minimum(num_train,x_val.shape[0])]
            X_test=do_det(x_test,NETPARS,'test')
            return(X_train,X_val,X_test)
        # Random perturbations
        else:
            insert=False
            if ('insert' in NETPARS['trans']):
                insert=True
            if (x_train is not None):
                X_train=do_rands(x_train[0:num_train],NETPARS,insert=insert)
                X_val=x_val[0:np.minimum(num_train,x_val.shape[0])] #do_rands(x_val[0:np.minimum(num_train,x_val.shape[0])],NETPARS)
            X_test=do_rands(x_test,NETPARS,insert=True)

            return(X_train,X_val,X_test)

def get_train(NETPARS):

    if (NETPARS['Mnist']=='mnist'):
        pad=0
        if ('data_pad' in NETPARS):
            pad=NETPARS['data_pad']
        X_train, y_train, X_val, y_val, X_test, y_test = mnist.load_dataset(pad=pad, nval=10000)
    else:
        num_val=5000
        if ('num_val' in NETPARS):
            num_val=NETPARS['num_val']
        X_train, y_train, X_val, y_val, X_test, y_test = cifar.load_dataset(NETPARS['Mnist'],Train=NETPARS['train'],white=False,num_val=num_val)

    num_train=NETPARS['num_train']
    if (num_train==0):
        num_train=np.shape(y_train)[0]
    if (y_train is not None):
        y_train=np.int32(y_train[0:num_train])
    y_test=np.int32(y_test)
    if ('trans' in NETPARS):
        if (NETPARS['trans']['angle']>0 or NETPARS['trans']['scale']>0 or
                    NETPARS['trans']['shift']>0 or NETPARS['trans']['flip'] or
                    NETPARS['trans']['saturation']):
            X_train, X_val, X_test = load_rotated_dataset(NETPARS,X_train,X_val,X_test,num_train)
            # To save time num_train data were processed also from val and test in case num_train is less than original length
            if (y_val is not None):
                y_val=np.int32(y_val[0:np.minimum(num_train,len(y_val))])
    else:
        if (y_val is not None):
            y_val=np.int32(y_val)

    if (X_train is not None):
        if (type(X_train) is list):
            #if (NETPARS['simple_augmentation']):
            ll=len(X_train)
            NETPARS['simple_augmentation']=ll
            X_train=np.concatenate(X_train,axis=0)
            y_train=np.tile(y_train,ll)
            # else:
            #     for i, X_t in enumerate(X_train):
            #         X_train[i]=X_t[0:num_train]
        else:
            X_train=X_train[0:num_train]
    if (type(X_test) is list):
        ll=len(X_test)
        NETPARS['simple_augmentation']=ll
        X_test=np.concatenate(X_test,axis=0)
        y_test=np.tile(y_test,ll)
    if ('edges' not in NETPARS or not NETPARS['edges']):
        pass
    else:
        if (X_train is not None):
            X_train=get_edges(X_train)
            X_val=get_edges(X_val)
        X_test=get_edges(X_test)


    return(X_train, y_train, X_val, y_val, X_test, y_test)


def get_edges(X):

    Xe=np.zeros((X.shape[0],24,X.shape[2],X.shape[3]),dtype=np.float32)
    for i in range(X.shape[1]):
        XX=np.float64(X[:,i,:,:])
        Xee=np.float32(ag.features.bedges(XX,minimum_contrast=.05))
        Xee=Xee.transpose(0,3,1,2)
        Xe[:,i*8:(i+1)*8,:,:]=Xee
    return(Xe)

def create_paired_data_set(NETPARS,X,y,num,cls=[],reps=1):

    if (NETPARS['simple']):
        if (len(cls)):
            ii=[]
            for c in cls:
                ii.append(np.where(y==c)[0])
            ii=np.hstack(ii)
        else:
            ii=range(X.shape[0])
        if (2*num> len(ii)):
            num=len(ii)/2

        np.random.shuffle(ii)
        ii1=ii[0:num]
        ii2=ii[num:(num+num)]
        ytr=y[ii1]==y[ii2]
        Xtr=X[ii1,]
        Xtr_comp=X[ii2,]
    else:

        X=X[:num]
        y=y[:num]
        num_class=np.int32(np.max(np.unique(y))+1)
        Xtr=[]
        Xtr_comp=[]
        ytr=[]
        for c in range(num_class):
            ii=np.where(y==c)[0]
            jj=np.where(y!=c)[0]

            iii=np.repeat(ii,reps)
            jjj=np.int32(np.floor(np.random.rand(len(iii))*len(jj)))
            Xtr.append(X[iii])
            Xtr.append(X[iii])
            ytr.append(np.ones(len(iii)))
            np.random.shuffle(iii)
            Xtr_comp.append(X[iii])
            Xtr_comp.append(X[jjj])
            ytr.append(np.zeros(len(iii)))

        Xtr=np.vstack(Xtr)
        Xtr_comp=np.vstack(Xtr_comp)
        ytr=np.hstack(ytr)

    return np.float32(Xtr), np.float32(Xtr_comp), np.float32(ytr)



def create_paired_data_set_with_fonts(X,y,num):
    import crop
    Xfont=crop.get_fonts()
    if (num> X.shape[0]):
        num=X.shape[0]
    print(num)
    ii=range(X.shape[0])
    np.random.shuffle(ii)
    ii1=ii[0:num]

    Xtr=np.repeat(X[ii1,],10,axis=0)
    yy=np.repeat(y[ii1],10)
    yyy=np.tile(range(10),num)
    Xtr_comp=Xfont[yyy,]
    ytr=(yy==yyy)
    ylab=y[ii1]
    Xtr_comp=np.expand_dims(Xtr_comp,axis=1)
    return np.float32(Xtr), np.float32(Xtr_comp), np.float32(ytr), np.int32(ylab)