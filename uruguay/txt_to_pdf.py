import os
import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont, ImageOps
import h5py
standard_size=(80,35)


def create_image(text,old_img,newname,oldname):
    old_im=Image.new('L',standard_size,255)
    oim=old_img.convert('L')
    oim=ImageOps.scale(oim,.5,resample=Image.BICUBIC)
    oim=ImageOps.invert(oim)
    oibx=oim.getbbox()
    oim=oim.crop(oibx)
    oim=ImageOps.invert(oim)
    old_im.paste(oim,(0,0))
    #old_im.show()
    #old_im.save(oldname)
    oibx=oim.getbbox()
    nold_im=np.array(old_im.getdata(),np.uint8).reshape(old_im.size[1], old_im.size[0])
    # img = Image.new('L', (200,40), 0)
    # draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype("Arial.ttf", 16)
    # draw.text((0, 0), text,255,font=font)
    # ibx=img.getbbox()
    # ibx_in=(ibx[0],ibx[1],ibx[2]+1,ibx[3]+1)
    # img=img.crop(ibx_in)
    # img=img.resize(oibx[2:4],resample=Image.BICUBIC)
    # ibx=img.getbbox()
    # img = ImageOps.invert(img)
    # new_im=Image.new('L', standard_size,255)
    # new_im.paste(img,(0,0))
    # nnew_im=np.array(new_im.getdata(),np.uint8).reshape(new_im.size[1], new_im.size[0])
    # IM=np.concatenate([nold_im,nnew_im],axis=0)
    IM=nold_im
    return IM #, ibx[2:4]

#create_image('Shit','out.pdf')
if ('darwin' in sys.platform):
    path=os.path.expanduser("~/Desktop/luisa-blocks-real")
else:
    path="/ga/amit/Desktop/luisa-blocks-real"

print("path")
def all_alfa(text):

    alfa='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    lent=len(text)
    c=0
    for t in text:
        if (t in alfa):
            c+=1
    alf=c==lent
    return alf

def produce_image(r):
    with open (r, "r") as f:
        data=f.readlines()
        if (len(data)>0):
            if (all_alfa(data[0])):
                oldr = r[0:r.find('txt')] + 'tif'
                oldimg=Image.open(oldr)
                ra = r.split('/')[-1]
                newr = 'PAIRS/'+ra[0:ra.find('.txt')] + '_cl.tiff'
                oldr = 'PAIRS/'+ra[0:ra.find('.txt')] + '_or.tiff'
                if (len(data[0])<max_length):
                    text=data[0]+' '*(max_length-len(data[0]))
                else:
                    text=data[0][0:max_length]
                IM=create_image(text,oldimg,newr,oldr)
                global Images, IBX, TEXT
                Images+=[IM]
                #IBX+=[ibx]
                TEXT+=[text]


def make_hpy():
    with h5py.File('pairs.hdf5', 'w') as f:
        dset1 = f.create_dataset("PAIRS", data=np.array(Images))
        #dset2 = f.create_dataset("BOXES", data=np.array(IBX))
    with open('texts.txt','w') as f:
        for tx in TEXT:
            f.write('%s\n' % tx)


def im_proc(path,rr):
    for r in rr:
        if 'txt' in r:
            produce_image(path+'/'+r)
            #print(t,num_images,t==num_images)
            if (len(TEXT)>=num_images):
                print("Hello", len(Images))
                make_hpy()
                exit()

def check_if_has_images(path):
    rr=os.listdir(path)
    rr.sort()
    print(len(rr), len(TEXT))
    if 'tif' in rr[0] or 'txt' in rr[0]:
        im_proc(path,rr)
    else:
        for r in rr:
            if not r.startswith('.'):
                check_if_has_images(path+'/'+r)


Images=[]
IBX=[]
TEXT=[]


num_images=np.int32(sys.argv[1])
max_length=np.int32(sys.argv[2])
check_if_has_images(path)
make_hpy()





