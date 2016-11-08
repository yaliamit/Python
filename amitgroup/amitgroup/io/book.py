# Reading files from Yali Amit's book
# Author: Gustav Larsson
import struct
from array import array as python_array 
import numpy as np


def get_tr(path,index):
    aa=int(np.floor(index/100))
    rr=np.mod(index,100)
    filename=path+'/Images_'+str(aa)
    N, buf = _get_buf_and_length(filename)
    offset, = struct.unpack('>I', buf[4+rr*4:4+rr*4+4])
    tr, = struct.unpack('>I',buf[offset:offset+4])
    return tr

def _get_buf_and_length(filename):
    f = open(filename, 'rb')
    buf = f.read()
    N, = struct.unpack('>I', buf[:4]) 
    return N, buf

def _unpack_image(buf, index, asbytes):
    # Get offset
    offset, = struct.unpack('>I', buf[4+index*4:4+index*4+4])

    # Get label
    tr, = struct.unpack('>I',buf[offset:offset+4])
    # Get width and height
    width, height = struct.unpack('>II', buf[4+offset:4+offset+8])

    # Finally get the image data put into numpy array
    img = np.array(python_array('B', buf[4+offset+8:4+offset+8+width*height])).reshape((width, height))

    if asbytes:
        img=np.uint8(img)
        return image(tr,img)
    else:
        return image(tr,img/255.)

def load_imagep(path, index, asbytes=False):
    aa=int(np.floor(index/100))
    rr=np.mod(index,100)
    filename=path+'/Images_'+str(aa)
    imaget=load_image(filename, rr, asbytes)
    
    return(imaget)

def process_im(img,slant,DIM):
    if (slant):
        img=imslant(img)
    img=embedd_image(img,DIM)
    return(img)

def embedd_image(img, DIM):

   
    if (DIM>0 and DIM>max(img.shape)):
        nimg=np.zeros((DIM,DIM))
        sx=np.round((DIM-img.shape[0])/2)
        sy=np.round((DIM-img.shape[1])/2)
        nimg[sx:sx+img.shape[0],sy:sy+img.shape[1]]=img
    else:
        nimg=img

    return(nimg)   

def imslant(img):
    aa=np.transpose(img)
    dimx=aa.shape[0]
    dimy=aa.shape[1]
    dimxh=aa.shape[0]/2
    dimyh=aa.shape[1]/2
    ii=np.where(aa>0)
    cc=len(ii[0])
    xa=np.mean(ii[0])
    ya=np.mean(ii[1])
    slopenum=np.sum(ii[1]*(ii[0]-xa))
    slopeden=np.sum(ii[1]*(ii[1]-ya))
    slope=slopenum/slopeden
    #print slope, xa, ya
    ii=np.outer(range(28),np.ones(28))
    jj=np.outer(np.ones(28),range(28))
    fx=ii-xa+(jj-ya)*slope+dimxh
    fy=jj-ya+dimyh
    x=np.int16(fx)
    y=np.int16(fy)
    a=fx-x
    b=fy-y
    xx1=x+1
    yy1=y+1
    timg=np.zeros((2*dimx+img.shape[0],2*dimy+img.shape[1]))
    timg[dimx:(dimx+img.shape[0]),dimy:(dimy+img.shape[1])]=aa
    imout=a*b*timg[xx1+dimx,yy1+dimy]+a*(1.-b)*timg[xx1+dimx,y+dimy]+(1.-a)*b*timg[x+dimx,yy1+dimy]+(1.-a)*(1.-b)*timg[x+dimx,y+dimy]
    #imout=timg[x+dimxh,y+dimyh]
    
    return(np.ubyte(np.transpose(imout)))

def load_image(filename, index, asbytes=False):
    """
    Load one image from Yali Amit's book, specifically the FACES data.

    Parameters
    ----------
    filename: str 
        Filename with absolute or relative path.
    index: int
        Which image in the file to read.
    asbytes : bool
        If True, returns data as ``numpy.uint8`` in [0, 255] as opposed to ``numpy.float64`` in [0.0, 1.0].

    Returns
    -------
    image:
        Image data of shape `(rows, cols)`.
    """

    # TODO: This shouldn't have to load the entire image buffer!
    
    N, buf = _get_buf_and_length(filename)
    if not (0 <= index < N):
        raise TypeError("Invalid index")

    return _unpack_image(buf, index, asbytes)

def load_all_images(filename, asbytes=False):
    """
    Load images from Yali Amit's book, specifically the FACES data.

    Parameters
    ----------
    filename: str 
        Filename with absolute or relative path.
    asbytes : bool
        If True, returns data as ``numpy.uint8`` in [0, 255] as opposed to ``numpy.float64`` in [0.0, 1.0].

    Returns
    -------
    image:
        Image data of shape ``(N, rows, cols)``, where ``N`` is the number of images. 
    """

    N, buf = _get_buf_and_length(filename)
    images = []
    for i in range(N):
        images.append(_unpack_image(buf, i, asbytes))
    
    return np.array(images) 

class image:
    truth=None
    img=None
    features=None

    def __init__(self,tr,a,f=[]):
        self.truth=tr
        self.img=a
        self.features=f
