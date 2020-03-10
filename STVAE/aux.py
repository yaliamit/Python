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
    parser.add_argument('--perturb', type=float, default=0, help='Learning rate (default: .001)')
    parser.add_argument('--hid_lr', type=float, default=.001, help='Learning rate (default: .001)')
    parser.add_argument('--binary_thresh', type=float, default=1e-6, help='threshold for bernoulli probs')
    parser.add_argument('--conf', type=float, default=0, help='confidence level')
    parser.add_argument('--ortho_lr', type=float, default=.0, help='Learning rate (default: .000001)')
    parser.add_argument('--mu_lr', type=float, default=[.05,.01], nargs=2,help='Learning rate (default: .05)')
    parser.add_argument('--lamda', type=float, default=1.0, help='weight decay')
    parser.add_argument('--lamda1', type=float, default=1.0, help='penalty on conv matrix')
    parser.add_argument('--scale', type=float, default=None, help='range of bias term for decoder templates')
    parser.add_argument('--lim', type=int, default=0, help='penalty on conv matrix')
    parser.add_argument('--num_mu_iter', type=int, default=10, help='Learning rate (default: .05)')
    parser.add_argument('--wd', type=float, default=0, help='Use weight decay')
    parser.add_argument('--sched', type=float, default=0, help='time step change')
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
    parser.add_argument('--del_last', action='store_true', help='dont update classifier weights')
    parser.add_argument('--sep', action='store_true', help='Output to consol')
    parser.add_argument('--embedd', action='store_true', help='embedding training')
    parser.add_argument('--reinit', action='store_true', help='reinitialize part of trained model')
    parser.add_argument('--only_pi', action='store_true', help='only optimize over pi')
    parser.add_argument('--edges', action='store_true', help='compute edges')
    parser.add_argument('--edge_dtr', type=float, default=0., help='difference minimum for edge')

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

def create_img(XX,c,h,w,ri=10,rj=10,sep=0):
    mat = []
    t = 0
    for i in range(ri):
        line = []
        for j in range(rj):
            if (t < len(XX)):
                line += [XX[t].reshape((c, h, w)).transpose(1, 2, 0)]
            else:
                line += [np.zeros((c,h,w)).transpose(1, 2, 0)]
            line+=[np.ones((c,sep,w)).transpose(1,2,0)]
            t += 1
        mat += [np.concatenate(line, axis=0)]
    manifold = np.concatenate(mat, axis=1)
    if (c==1):
        img = np.concatenate([manifold, manifold, manifold], axis=2)
    else:
        img=manifold
    return img

def create_image(XX, model, ex_file):

    img=create_img(XX,model.input_channels,model.h,model.w)


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

    #rdata=rotate_dataset_rand(data) #,angle=40,scale=.2)
    rdata=data
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


