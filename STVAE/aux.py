import numpy as np
import pylab as py
import torch
from imageio import imsave
import os


def process_args(parser):
    parser.add_argument('--transformation', default='aff', help='type of transformation: aff or tps')
    parser.add_argument('--feats', type=int, default=0, help='Number of features in case data preprocessed')
    parser.add_argument('--filts', type=int, default=3, help='Number of features in case data preprocessed')
    parser.add_argument('--input_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--type', default='vae', help='type of transformation: aff or tps')
    parser.add_argument('--tps_num', type=int, default=3, help='dimension of s')
    parser.add_argument('--sdim', type=int, default=26, help='dimension of s')
    parser.add_argument('--hdim', type=int, default=256, help='dimension of h')
    parser.add_argument('--hdim_dec',type=int, default=None, help='dims of decoder')
    parser.add_argument('--num_hlayers', type=int, default=0, help='number of hlayers')
    parser.add_argument('--nepoch', type=int, default=40, help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=2, help='whether to run in the GPU')
    parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
    parser.add_argument('--num_train', type=int, default=60000, help='num train (default: 60000)')
    parser.add_argument('--nval', type=int, default=1000, help='num train (default: 1000)')
    parser.add_argument('--mb_size', type=int, default=100, help='mb_size (default: 500)')
    parser.add_argument('--n_class', type=int, default=0, help='number of classes')
    parser.add_argument('--model', default=None, nargs=2, help='model (default: base)')
    parser.add_argument('--optimizer', default='Adam', help='Type of optimiser')
    parser.add_argument('--lr', type=float, default=.001, help='Learning rate (default: .001)')
    parser.add_argument('--ortho_lr', type=float, default=.000001, help='Learning rate (default: .000001)')
    parser.add_argument('--mu_lr', type=float, default=[.05,.01], nargs=2,help='Learning rate (default: .05)')
    parser.add_argument('--lamda', type=float, default=0.0, help='Weight decay (default: 0)')
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
    parser.add_argument('--CONS', action='store_true', help='Output to consol')
    parser.add_argument('--sample', action='store_true', help='sample from distribution')
    parser.add_argument('--classify', action='store_true', help='Output to consol')
    parser.add_argument('--Diag', action='store_true', help='Output to consol')
    parser.add_argument('--sep', action='store_true', help='Output to consol')

    parser.add_argument('--output_prefix', default='', help='path to model')

    #parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    args = parser.parse_args()

    return (args)

def make_images(test,model,ex_file,args):

    if (model.feats==0):
        old_bsz=model.bsz
        model.bsz = 100
        model.setup_id(model.bsz)
        num_mu_iter=None
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if (args.n_class):
            for c in range(model.n_class):
                ind=(np.argmax(test[1],axis=1)==c)
                show_reconstructed_images([test[0][ind],test[1][ind]],model,ex_file,args.nti,c)
        else:
            show_reconstructed_images(test,model,ex_file,args.nti)

        if args.n_mix>0:
            for clust in range(args.n_mix):
                show_sampled_images(model,ex_file,clust)
        else:
            show_sampled_images(model, ex_file)

        model.bsz=old_bsz
        model.setup_id(old_bsz)

def create_image(XX, ex_file):
    mat = []
    t = 0
    for i in range(10):
        line = []
        for j in range(10):
            if (t<len(XX)):
                line += [XX[t].reshape((28,28))]
            else:
                line += [np.zeros((28,28))]
            t+=1
        mat+=[np.concatenate(line,axis=0)]
    manifold = np.concatenate(mat, axis=1)
    manifold = manifold[np.newaxis, :]
    img = np.concatenate([manifold, manifold, manifold], axis=0).transpose(1,2,0)

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
    create_image(XX, ex_file)


def show_reconstructed_images(test,model,ex_file,num_iter=None, cl=None):

    inp = torch.from_numpy(test[0][0:100].transpose(0, 3, 1, 2))
    if (cl is not None):
        X=model.recon(inp,num_iter,cl)
    else:
        X = model.recon(inp, num_iter)
    X = X.cpu().detach().numpy().reshape(inp.shape)
    XX=np.concatenate([inp[0:50],X[0:50]])
    if (cl is not None):
        create_image(XX,ex_file+'_recon'+'_'+str(cl))
    else:
        create_image(XX, ex_file + '_recon')

def add_occlusion(recon_data):
    recon_data[0][0:20,0:13,:,:]=0
    return recon_data

def add_clutter(recon_data):

    block_size=3
    num_clutter=2
    dimx=recon_data[0].shape[1]
    dimy=recon_data[0].shape[2]
    for im in recon_data[0]:
        for k in range(num_clutter):
            x=np.int(np.random.rand()*(dimx-block_size))
            y=np.int(np.random.rand()*(dimy-block_size))
            im[x:x+block_size,y:y+block_size,0]=np.ones((block_size,block_size))

    return recon_data




