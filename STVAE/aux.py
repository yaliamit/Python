import numpy as np
import pylab as py
import torch
from scipy.misc import imsave
import os

def process_args(parser):
    parser.add_argument('--transformation', default='aff', help='type of transformation: aff or tps')
    parser.add_argument('--type', default='vae', help='type of transformation: aff or tps')
    parser.add_argument('--sdim', type=int, default=26, help='dimension of s')
    parser.add_argument('--hdim', type=int, default=256, help='dimension of h')
    parser.add_argument('--num_hlayers', type=int, default=0, help='number of hlayers')
    parser.add_argument('--nepoch', type=int, default=40, help='number of training epochs')
    parser.add_argument('--gpu', type=bool, default=False, help='whether to run in the GPU')
    parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
    parser.add_argument('--num_train', type=int, default=60000, help='num train (default: 60000)')
    parser.add_argument('--nval', type=int, default=1000, help='num train (default: 1000)')
    parser.add_argument('--mb_size', type=int, default=100, help='mb_size (default: 500)')
    parser.add_argument('--model', default='base', help='model (default: base)')
    parser.add_argument('--optimizer', default='Adam', help='Type of optimiser')
    parser.add_argument('--lr', type=float, default=.001, help='Learning rate (default: .001)')
    parser.add_argument('--mu_lr', type=float, default=.05, help='Learning rate (default: .05)')
    parser.add_argument('--num_mu_iter', type=int, default=10, help='Learning rate (default: .05)')
    parser.add_argument('--wd', action='store_true', help='Use weight decay')
    parser.add_argument('--cl', type=int, default=None, help='class (default: None)')
    parser.add_argument('--run_existing', action='store_true', help='Use existing model')
    parser.add_argument('--nti', type=int, default=500, help='num test iterations (default: 100)')
    parser.add_argument('--nvi', type=int, default=20, help='num val iterations (default: 20)')
    parser.add_argument('--n_mix', type=int, default=0, help='num mixtures (default: 0)')
    parser.add_argument('--MM', action='store_true', help='Use max max')
    parser.add_argument('--OPT', action='store_true', help='Optimization instead of encoding')
    parser.add_argument('--CONS', action='store_true', help='Output to consol')

    args = parser.parse_args()

    return (args)

def create_image(XX, ex_file):
    mat = []
    t = 0
    for i in range(10):
        line = []
        for j in range(10):
            line += [XX[t].reshape((28,28))]
            t+=1
        mat+=[np.concatenate(line,axis=0)]
    manifold = np.concatenate(mat, axis=1)
    manifold = 1. - manifold[np.newaxis, :]
    img = np.concatenate([manifold, manifold, manifold], axis=0).transpose(1,2,0)

    if not os.path.isfile('_Images'):
        os.system('mkdir _Images')
    imsave('_Images/'+ex_file+'.png', img)

    print("Saved the sampled images")

def show_sampled_images(model,ex_file):
    model.bsz=100
    theta = torch.zeros(model.bsz, 6)
    X=model.sample_from_z_prior(theta)
    XX=X.cpu().detach().numpy()
    create_image(XX, ex_file)

def show_reconstructed_images(test,model,ex_file,num_iter=None):

    inp = torch.from_numpy(test[0][0:50].transpose(0, 3, 1, 2))
    X=model.recon(inp,num_iter)
    X = X.cpu().detach().numpy().reshape(inp.shape)
    XX=np.concatenate([inp,X])
    create_image(XX,ex_file+'_recon')

def rerun_on_train_test(model,train,test,args):
    trainMU, trainLOGVAR = model.initialize_mus(train, args.OPT)
    testMU, testLOGVAR = model.initialize_mus(test, args.OPT)
    if (args.OPT):
        model.setup_id(model.bsz)
        model.run_epoch(train, 0, args.nti, trainMU, trainLOGVAR, type='trest', fout=fout)
        model.run_epoch(test, 0, args.nti, testMU, testLOGVAR, type='test', fout=fout)
    else:
        model.run_epoch(test, 0, type='test')

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

def test_with_noise(test,model):

    ii=np.arange(0,test[0].shape[0],1)
    np.random.shuffle(ii)
    recon_data=[test[0][ii[0:20]].copy(),test[1][ii[0:20]].copy()]
    recon_data=add_clutter(recon_data)
    recon_ims=model.recon(recon_data, num_mu_iter=args.nti)
    rec=recon_ims.detach().cpu()
    py.figure(figsize=(3, 20))
    for t in range(20):
            py.subplot(20,3,3*t+1)
            py.imshow(test[0][ii[t],:,:,0])
            py.axis('off')
            py.subplot(20,3,3*t+2)
            py.imshow(recon_data[0][t, :, :, 0])
            py.axis('off')
            py.subplot(20,3,3*t+3)
            py.imshow(rec[t,0,:,:])
            py.axis('off')
    py.show()
    print("hello")


def re_estimate(model,train,args,fout):

            fout.write('Means and variances of latent variable before restimation\n')
            fout.write(str(model.MU.data)+'\n')
            fout.write(str(model.LOGVAR.data)+'\n')
            fout.flush()
            trainMU, trainLOGVAR = model.initialize_mus(train[0], args.OPT,args.MM)
            trainMU, trainLOGVAR = model.run_epoch(train, 0, 500,trainMU, trainLOGVAR, type='trest',fout=fout)
            model.MU = torch.nn.Parameter(torch.mean(trainMU, dim=0))
            model.LOGVAR = torch.nn.Parameter(torch.log(torch.var(trainMU, dim=0)))
            fout.write('Means and variances of latent variable after restimation\n')
            fout.write(str(model.MU.data) + '\n')
            fout.write(str(model.LOGVAR.data) + '\n')
            fout.flush()
            model.to(model.dv)
