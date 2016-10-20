import pylab as py
import numpy as np
import scipy.misc

# Get fixed fonts from a word file
def get_fonts():

    rescale_size=20
    imsize=28
    im=py.imread('MNIST/digits.jpg')
    im=im[:,:,0]

    im=255-im
    im=(im>200)


    sum0=np.sum(im,axis=0)
    ii=np.where(sum0==0)[0]
    dii=np.int32(np.diff(ii))
    fii=np.where(dii>1)[0][0]

    im0=im[:,fii:fii+(dii[fii])]
    sum1=np.sum(im0,axis=1)

    ii=np.where(sum1==0)[0]
    dii=np.diff(ii)
    fii=np.where(dii>1)[0]

    IMS=[]
    for f in fii:
        b=ii[f]
        imt=im0[b:b+dii[f],:]
        st=np.sum(imt,axis=0)
        iist=np.where(st==0)[0]
        if (len(iist)>0):
            diist=np.int32(np.diff(iist))
            fdiist=np.where(diist>1)[0]
            if (len(fdiist)>0):
                imtt=imt[:,fdiist[0]:fdiist[0]+diist[fdiist[0]]]
            else:
                imtt=imt[:,len(iist):]
        else:
            imtt=imt

        imtts=scipy.misc.imresize(imtt,(rescale_size,rescale_size))
        xi,xj=np.where(imtts>0)
        mii=np.int32(np.round(np.mean(xi)))
        mjj=np.int32(np.round(np.mean(xj)))

        imtts0=np.zeros((imsize,imsize))
        imtts0[imsize/2-mii:imsize/2+(rescale_size-mii),imsize/2-mjj:imsize/2+(rescale_size-mjj)]=imtts
        imtts0=np.double(imtts0)/255.
        # py.imshow(imtts0)
        # py.show()
        IMS.append(imtts0)

    IMSS=np.array(IMS)
    print('done')

    return(IMSS)