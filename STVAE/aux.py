import numpy as np
import pylab as py
import torch
from imageio import imsave
import os
from scipy import ndimage
import scipy
import matplotlib.colors as col

def process_args(parser):
    parser.add_argument('--full_dim', type=int, default=256, help='fully connected layer size')
    parser.add_argument('--hid_hid', type=int, default=256, help='fully connected layer size')
    parser.add_argument('--hid_prob', type=float, default=0., help='dropout')
    parser.add_argument('--transformation', default='aff', help='type of transformation: aff or tps')
    parser.add_argument('--feats', type=int, default=0, help='Number of features in case data preprocessed')
    parser.add_argument('--feats_back', action='store_true',help='reconstruct image from features')
    parser.add_argument('--filts', type=int, default=3, help='Filter size')
    parser.add_argument('--pool', type=int, default=2, help='Pooling size')
    parser.add_argument('--pool_stride', type=int, default=2, help='Pooling stride')
    parser.add_argument('--input_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--type', default='vae', help='type of transformation: aff or tps')
    parser.add_argument('--dataset', default='mnist', help='which data set')
    parser.add_argument('--layers',  nargs="*", default=None, help='layer')
    parser.add_argument('--hid_layers',  nargs="*", default=None, help='layer')
    parser.add_argument('--tps_num', type=int, default=3, help='dimension of s')
    parser.add_argument('--sdim', type=int, default=26, help='dimension of s')
    parser.add_argument('--hdim', type=int, default=256, help='dimension of h')
    parser.add_argument('--hdim_dec',type=int, default=None, help='dims of decoder')
    parser.add_argument('--num_hlayers', type=int, default=0, help='number of hlayers')
    parser.add_argument('--nepoch', type=int, default=40, help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=2, help='whether to run in the GPU')
    parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
    parser.add_argument('--num_train', type=int, default=60000, help='num train (default: 60000)')
    parser.add_argument('--network_num_train', type=int, default=60000, help='num train (default: 60000)')
    parser.add_argument('--num_test', type=int, default=0, help='num test (default: 10000)')
    parser.add_argument('--nval', type=int, default=1000, help='num train (default: 1000)')
    parser.add_argument('--mb_size', type=int, default=100, help='mb_size (default: 500)')
    parser.add_argument('--n_class', type=int, default=0, help='number of classes')
    parser.add_argument('--model', default=None, nargs="*", help='model (default: base)')
    parser.add_argument('--optimizer', default='Adam', help='Type of optimiser')
    parser.add_argument('--lr', type=float, default=.001, help='Learning rate (default: .001)')
    parser.add_argument('--conf', type=float, default=0, help='confidence level')
    parser.add_argument('--ortho_lr', type=float, default=.0, help='Learning rate (default: .000001)')
    parser.add_argument('--mu_lr', type=float, default=[.05,.01], nargs=2,help='Learning rate (default: .05)')
    parser.add_argument('--lamda', type=float, default=1.0, help='weight decay')
    parser.add_argument('--lamda1', type=float, default=1.0, help='penalty on conv matrix')
    parser.add_argument('--scale', type=float, default=None, help='range of bias term for decoder templates')
    parser.add_argument('--lim', type=int, default=0, help='penalty on conv matrix')
    parser.add_argument('--num_mu_iter', type=int, default=10, help='Learning rate (default: .05)')
    parser.add_argument('--wd', action='store_true', help='Use weight decay')
    parser.add_argument('--cl', type=int, default=None, help='class (default: None)')
    parser.add_argument('--run_existing', action='store_true', help='Use existing model')
    parser.add_argument('--nti', type=int, default=500, help='num test iterations (default: 100)')
    parser.add_argument('--nvi', type=int, default=20, help='num val iterations (default: 20)')
    parser.add_argument('--n_mix', type=int, default=0, help='num mixtures (default: 0)')
    parser.add_argument('--clust', type=int, default=None, help='which cluster to shoe')
    parser.add_argument('--n_parts', type=int, default=0, help='number of parts per location')
    parser.add_argument('--n_part_locs', type=int, default=0, help='number of part locations (a^2)')
    parser.add_argument('--part_dim', type=int, default=None, help='dimension of part')
    parser.add_argument('--MM', action='store_true', help='Use max max')
    parser.add_argument('--OPT', action='store_true', help='Optimization instead of encoding')
    parser.add_argument('--network', action='store_true', help='classification network')
    parser.add_argument('--CONS', action='store_true', help='Output to consol')
    parser.add_argument('--sample', action='store_true', help='sample from distribution')
    parser.add_argument('--classify', action='store_true', help='Output to consol')
    parser.add_argument('--Diag', action='store_true', help='Output to consol')
    parser.add_argument('--output_cont', action='store_true', help='cont data')
    parser.add_argument('--erode', action='store_true', help='cont data')
    parser.add_argument('--rerun', action='store_true', help='cont data')

    parser.add_argument('--sep', action='store_true', help='Output to consol')
    parser.add_argument('--reinit', action='store_true', help='reinitialize part of trained model')
    parser.add_argument('--only_pi', action='store_true', help='only optimize over pi')

    parser.add_argument('--output_prefix', default='', help='path to model')

    #parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    args = parser.parse_args()

    return (args)

def make_images(test,model,ex_file,args):

    if (True):
        old_bsz=model.bsz
        model.bsz = 100
        model.setup_id(model.bsz)
        num_mu_iter=None
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        if args.n_mix>0:
            for clust in range(args.n_mix):
                show_sampled_images(model,ex_file,clust)
        else:
            show_sampled_images(model, ex_file)

        if (args.n_class):
            for c in range(model.n_class):
                ind=(test[1]==c)
                show_reconstructed_images([test[0][ind]],model,ex_file,args.nti,c, args.erode)
        else:
            show_reconstructed_images(test,model,ex_file,args.nti,None, args.erode)



        model.bsz=old_bsz
        model.setup_id(old_bsz)

def create_image(XX, model, ex_file):
    mat = []
    t = 0
    for i in range(10):
        line = []
        for j in range(10):
            if (t<len(XX)):
                line += [XX[t].reshape((model.input_channels,model.h,model.w)).transpose(1,2,0)]
            else:
                line += [np.zeros((model.input_channels,model.h,model.w)).transpose(1,2,0)]
            t+=1
        mat+=[np.concatenate(line,axis=0)]
    manifold = np.concatenate(mat, axis=1)
    if (model.input_channels==1):
        img = np.concatenate([manifold, manifold, manifold], axis=2)
    else:
        img=manifold

    if not os.path.isdir('_Images'):
        os.system('mkdir _Images')
    imsave('_Images/'+ex_file+'.png', np.uint8(img*255))

    #print("Saved the sampled images")

def show_sampled_images(model,ex_file,clust=None):
    theta = torch.zeros(model.bsz, model.u_dim)
    X=model.sample_from_z_prior(theta,clust)
    XX=X.cpu().detach().numpy()
    if clust is not None:
        ex_file=ex_file+'_'+str(clust)
    create_image(XX, model, ex_file)


def show_reconstructed_images(test,model,ex_file,num_iter=None, cl=None, erd=False):

    inp=torch.from_numpy(erode(erd,test[0][0:100]))


    if (cl is not None):
        X=model.recon(inp,num_iter,cl)
    else:
        X,_ = model.recon(inp, num_iter)
    X = X.cpu().detach().numpy().reshape(inp.shape)
    XX=np.concatenate([inp[0:50],X[0:50]])
    if (cl is not None):
        create_image(XX,model, ex_file+'_recon'+'_'+str(cl))
    else:
        create_image(XX, model, ex_file + '_recon')

def add_occlusion(recon_data):
    recon_data[0][0:20,0:13,:,:]=0
    return recon_data

def add_clutter(recon_data):

    block_size=3
    num_clutter=2
    dim=np.zeros((1,2))
    dim[0,0]=recon_data[0].shape[1]-block_size
    dim[0,1]=recon_data[0].shape[2]-block_size
    qq=np.random.rand(recon_data.shape[0],num_clutter,2)
    rr=np.int32(qq*dim)
    for  rrr,im in zip(rr,recon_data):
        for k in range(num_clutter):
            x=rrr[k,0]
            y=rrr[k,1]
            im[0,x:x+block_size,y:y+block_size]=np.ones((block_size,block_size))

    return recon_data

def prepare_recons(model, DATA, args):
    dat = []
    HV=[]
    for k in range(3):
        if (DATA[k][0] is not None):
            INP = torch.from_numpy(DATA[k][0])
            if k==0:
                INP = INP[0:args.network_num_train]
            RR = []
            HVARS=[]
            for j in np.arange(0, INP.shape[0], 500):
                inp = INP[j:j + 500]
                rr, h_vars = model.recon(inp, args.nti)
                RR += [rr.detach().cpu().numpy()]
                HVARS += [h_vars.detach().cpu().numpy()]
            RR = np.concatenate(RR)
            HVARS = np.concatenate(HVARS)
            tr = RR.reshape(-1, 1, 28, 28)
            dat += [[tr, DATA[k][1][0:INP.shape[0]]]]
            HV+=[[HVARS,DATA[k][1][0:INP.shape[0]]]]
        else:
            dat += [DATA[k]]
            HV += [DATA[k]]
    print("Hello")

    return dat, HV


def erode(do_er,data):

    rdata=rotate_dataset_rand(data) #,angle=40,scale=.2)
    if (do_er):
        el=np.zeros((3,3))
        el[0,1]=el[1,0]=el[1,2]=el[2,1]=el[1,1]=1
        rr=np.random.rand(len(data))<.5
        ndata=np.zeros_like(data)
        for r,ndd,dd in zip(rr,ndata,rdata):
            if (r):
                dda=ndimage.binary_erosion(dd[0,:,:]>0,el).astype(dd.dtype)
            else:
                dda=ndimage.binary_dilation(dd[0,:,:]>.9,el).astype(dd.dtype)
            ndd[0,:,:]=dda
    else:
        ndata=rdata

    return ndata

def rotate_dataset_rand(X,angle=0,scale=0,shift=0,gr=0,flip=False,blur=False,saturation=False, spl=None):
    # angle=NETPARS['trans']['angle']
    # scale=NETPARS['trans']['scale']
    # #shear=NETPARS['trans']['shear']
    # shift=NETPARS['trans']['shift']
    if angle==0 and scale==0 and shift==0 and not flip and not blur and not saturation:
        return X
    s=np.shape(X)
    Xr=np.zeros(s)
    cent=np.array(s[2:4])/2
    angles=np.random.rand(Xr.shape[0])*angle-angle/2.
    #aa=np.random.rand(Xr.shape[0])*.25
    #aa[np.int32(len(aa)/2):]=aa[np.int32(len(aa)/2):]+.75
    #angles=aa*angle-angle/2
    SX=np.exp(np.random.rand(Xr.shape[0],2)*scale-scale/2.)
    SH=np.int32(np.round(np.random.rand(Xr.shape[0],2)*shift)-shift/2)
    FL=np.zeros(Xr.shape[0])
    BL=np.zeros(Xr.shape[0])
    HS=np.zeros(Xr.shape[0])
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
            print(i," ")
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
        ii=np.arange(0,X.shape[0],1)
        np.random.shuffle(ii)
        nr=12
        nr2=nr*nr
        for j in range(nr2):
            print(angles[ii[j]]) #,SX[i],SH[i],FL[i],BL[i])
            py.figure(fig1.number)
            py.subplot(nr,nr,j+1)
            py.imshow(X[ii[j]].transpose(1,2,0))
            py.axis('off')
            py.figure(fig2.number)
            py.subplot(nr,nr,j+1)
            py.imshow(Xr[ii[j]].transpose(1,2,0))
            py.axis('off')
        py.show()


    return(np.float32(Xr))



