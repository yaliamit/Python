import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
import h5py

def process_args(parser):
    parser.add_argument('--filts', type=int, default=(3,3,3,3), help='size of filters')
    parser.add_argument('--feats', type=int, default=(1,32,32,64,256), help='number of filters')
    parser.add_argument('--pools', type=int, default=   (2, 2, 1, 2), help='pooling')
    parser.add_argument('--drops', type=float, default=(1.,1.,1.,1.,.5))
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

def create_image(trin, TT, x_dim, ex_file):
        mat = []
        t = 0
        ll=len(trin)//63 *63
        page=[]
        t=0
        imlist=[]
        while (t<ll):
            page=[]
            for j in range(7):
                col=[]
                for i in range(9):
                    if (t<ll):
                        text=''.join(TT[t,:])
                        img = Image.new('L', (80+5, x_dim+20), 255)
                        imga = Image.fromarray(np.int8(trin[t,0,0:x_dim]*200))
                        img.paste(imga, (0,0))
                        draw = ImageDraw.Draw(img)
                        font = ImageFont.truetype("Arial.ttf", 16)
                        draw.text((0,x_dim), text, 0, font=font)
                        col += [np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])]
                        t+=1
                    else:
                        col += [np.ones((x_dim+20,85))*255]

                COL = np.concatenate(col, axis=0)
                page+=[COL]
            manifold = np.concatenate(page, axis=1)
            imlist.append(Image.fromarray(manifold))
        imlist[0].save("_Images/test.tif", compression="tiff_deflate", save_all=True,
                       append_images=imlist[1:])

        #if not os.path.isfile('_Images'):
        #    os.system('mkdir _Images')
        #imsave('_Images/' + ex_file + '.png', img)

        print("Saved the sampled images")

def make_boxes(bx,td):
    standard_size = (35, 150)
    boxes=[]
    for b,tr in zip(bx,td):
        a=np.zeros(standard_size)
        a[0:np.int32(b[1]),0:np.int32(b[0])]=1
        #a[tr[0,standard_size[0]:,:]<.3]=2
        boxes+=[a]
    boxes=np.array(boxes)
    return boxes


def get_data(args):
    with h5py.File('pairs.hdf5', 'r') as f:
        #key = list(f.keys())[0]
        # Get the data
        pairs = f['PAIRS']
        print('tr', pairs.shape)
        all_pairs=np.float32(pairs)/255.
        all_pairs=all_pairs[0:args.num_train]
        all_pairs=all_pairs.reshape(-1,1,all_pairs.shape[1],all_pairs.shape[2])
        lltr=np.int32(np.ceil(.8*len(all_pairs))//args.bsz *args.bsz)
        llte=np.int32((len(all_pairs)-lltr)//args.bsz * args.bsz)
        ii=np.array(range(lltr+llte))
        np.random.shuffle(ii)
        #bx=np.float32(f['BOXES'])
        #boxes=make_boxes(bx,all_pairs)
        train_data = all_pairs[ii[0:lltr]]
        #train_data_boxes=boxes[ii[0:lltr]]
        test_data=all_pairs[ii[lltr:lltr+llte]]
        #test_data_boxes=boxes[ii[lltr:lltr+llte]]
    with open('texts.txt','r') as f:
        TEXT = [line.rstrip() for line in f.readlines()]
        aa=sorted(set(' '.join(TEXT)))
        print(aa)
        if (' ' in aa):
            ll=len(aa)
            spa=0
        else:
            ll=len(aa)+1
            spa=ll-1
        args.ll = ll
        train_t=[TEXT[j] for j in ii[0:lltr]]
        test_t=[TEXT[j] for j in ii[lltr:lltr+llte]]
        lens=[len(r) for r in train_t]
        args.lenc=np.max(lens)
        train_text=np.ones((len(train_t),args.lenc))*spa
        for j,tt in enumerate(train_t):
            for i,ss in enumerate(tt):
                train_text[j,i]=aa.index(ss)
        test_text=np.ones((len(test_t),args.lenc))*spa
        for j,tt in enumerate(test_t):
            for i,ss in enumerate(tt):
                test_text[j,i]=aa.index(ss)
        train_text=np.int32(train_text)
        test_text=np.int32(test_text)
        print("hello")
        args.aa=aa

    return train_data, train_text, test_data, test_text

def add_shifts(input,S,T,dv):


    ss=input.shape
    ls=len(S)
    lt=len(T)
    input_s=input.repeat(1,ls*lt,1,1).view(ss[0]*ls*lt,ss[1],ss[2],ss[3])
    l=len(input_s)

    for i,s in enumerate(S):
        lls = np.arange(i*lt, l, ls*lt)
        for j,t in enumerate(T):
            llst=lls+j
            input_s[llst,:]=torch.cat((input_s[llst,:,:,s:],torch.zeros(len(llst),ss[1],ss[2],s,dtype=torch.float).to(dv)),dim=3)
            input_s[llst,:]=torch.cat((input_s[llst,:,t:,:],torch.zeros(len(llst),ss[1],t,ss[3],dtype=torch.float).to(dv)),dim=2)


    return input_s

def add_shifts_new(input,S,T):


    ss=input.shape
    ls=len(S)
    lt=len(T)
    input_s=np.repeat(input,ls*lt,axis=0)
    l=len(input_s)

    for i,s in enumerate(S):
        lls = np.arange(i*lt, l, ls*lt)
        for j,t in enumerate(T):
            llst=lls+j
            input_s[llst,:]=np.concatenate((input_s[llst,:,:,s:],np.ones((len(llst),ss[1],ss[2],s))),axis=3)
            input_s[llst,:]=np.concatenate((input_s[llst,:,t:,:],torch.ones((len(llst),ss[1],t,ss[3]))),axis=2)


    return input_s