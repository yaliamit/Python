import argparse
import numpy as np
import os
from scipy.misc import imsave

def process_args(parser):
    parser.add_argument('--filts', type=int, default=(3,3,3), help='size of filters')
    parser.add_argument('--feats', type=int, default=(1,16,32,128), help='number of filters')
    parser.add_argument('--num_char', type=int, default=5, help='number of characters')
    parser.add_argument('--filt_size_out', type=int, default=3, help='size of last layer filter')
    parser.add_argument('--bsz', type=int, default=100, help='mb_size (default: 500)')
    parser.add_argument('--nepoch', type=int, default=30, help='number of training epochs')
    parser.add_argument('--gpu', type=bool, default=False, help='whether to run in the GPU')
    parser.add_argument('--seed', type=int, default=1345, help='random seed (default: 1111)')
    parser.add_argument('--num_train', type=int, default=60000, help='num train (default: 60000)')
    parser.add_argument('--nval', type=int, default=(10,10), help='num train (default: 1000)')
    parser.add_argument('--model', default='base', help='model (default: base)')
    parser.add_argument('--optimizer', default='Adam', help='Type of optimiser')
    parser.add_argument('--lr', type=float, default=.001, help='Learning rate (default: .001)')
    parser.add_argument('--run_existing', action='store_true', help='Use existing model')
    parser.add_argument('--OPT', action='store_true', help='Optimization instead of encoding')
    parser.add_argument('--CONS', action='store_true', help='Output to consol')
    parser.add_argument('--wd', action='store_true', help='Output to consol')
    parser.add_argument('--output_prefix', default='', help='path to model')

    args = parser.parse_args()
    return (args)

def create_image(trin, trat, OUT, ex_file):
        mat = []
        t = 0
        ll=len(trin)
        page=[]
        t=0
        for j in range(3):
            col=[]
            for i in range(9):
                col+=[trin[t]]
                col+=[trat[t]]
                col+=[OUT[t]]
                t+=1
            COL = np.concatenate(col, axis=0)
            page+=[COL]
        manifold=np.concatenate(page,axis=1)
        manifold = manifold[np.newaxis, :]
        img = np.concatenate([manifold, manifold, manifold], axis=0).transpose(1, 2, 0)

        if not os.path.isfile('_Images'):
            os.system('mkdir _Images')
        imsave('_Images/' + ex_file + '.png', img)

        print("Saved the sampled images")

