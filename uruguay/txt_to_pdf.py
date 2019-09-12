import os
import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont, ImageOps
import h5py

standard_size=(300,70)
t=0
Images=[]

def create_image(text,old_img,newname,oldname):
    old_im=Image.new('L',standard_size,255)
    oim=old_img.convert('L')
    oim=ImageOps.invert(oim)
    oibx=oim.getbbox()
    oim=oim.crop(oibx)
    oim=ImageOps.invert(oim)
    old_im.paste(oim,(0,0))
    #old_im.show()
    #old_im.save(oldname)
    oibx=oim.getbbox()
    nold_im=np.array(old_im.getdata(),np.uint8).reshape(old_im.size[1], old_im.size[0])
    img = Image.new('L', (200,40), 0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Arial.ttf", 16)
    draw.text((0, 0), text,255,font=font)
    ibx=img.getbbox()
    ibx_in=(ibx[0],ibx[1],ibx[2]+1,ibx[3]+1)
    img=img.crop(ibx_in)
    img=ImageOps.invert(img)
    img=img.resize(oibx[2:4],resample=Image.BICUBIC)
    new_im=Image.new('L', standard_size,255)
    new_im.paste(img,(0,0))
    nnew_im=np.array(new_im.getdata(),np.uint8).reshape(new_im.size[1], new_im.size[0])
    IM=np.concatenate([nold_im,nnew_im],axis=0)
    return IM

#create_image('Shit','out.pdf')
#path=os.path.expanduser("~/Desktop/luisa-blocks-real")
path="/ga/amit/Desktop/luisa-blocks-real"


def produce_image(r):
    with open (r, "r") as f:
        data=f.readlines()
        if (len(data)>0):
            oldr = r[0:r.find('txt')] + 'tif'
            oldimg=Image.open(oldr)
            ra = r.split('/')[-1]
            newr = 'PAIRS/'+ra[0:ra.find('.txt')] + '_cl.tiff'
            oldr = 'PAIRS/'+ra[0:ra.find('.txt')] + '_or.tiff'
            IM=create_image(data[0],oldimg,newr,oldr)
            global Images
            Images+=[IM]

def im_proc(path,rr):
    for r in rr:
        if 'txt' in r:
            produce_image(path+'/'+r)
            global t
            t=t+1
            print(t,num_images,t==num_images)
            if (t==num_images):
                print("Hello", len(Images))
                with h5py.File('pairs.hdf5', 'w') as f:
                    dset = f.create_dataset("PAIRS", data=np.array(Images))
                    print("HH")
                exit()

def check_if_has_images(path):
    rr=os.listdir(path)
    rr.sort()
    if 'tif' in rr[0] or 'txt' in rr[0]:
        im_proc(path,rr)
    else:
        for r in rr:
            if not r.startswith('.'):
                check_if_has_images(path+'/'+r)
    with h5py.File('pairs.hdf5', 'w') as f:
        dset = f.create_dataset("PAIRS", data=np.array(Images))

num_images=np.int32(sys.argv[1])
check_if_has_images(path)




with h5py.File('pairs.hdf5', 'r') as f:
    key = list(f.keys())[0]
    # Get the data
    pairs = f[key]
    print('tr', pairs.shape)
    all_pairs=np.float32(pairs)
    ll=len(all_pairs)
    ii=np.array(range(ll))
    np.random.shuffle(ii)
    ll_tr=np.int32(np.ceil(.8*ll))
    train_data=all_pairs[ii[0:ll_tr]]
    test_data=all_pairs[ii[ll_tr:]]

    print("Hello")

