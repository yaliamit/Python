import numpy as np
import pylab as py

import scipy.signal as signal
import scipy.interpolate as inp

def show_images(ims,num=None):

    py.figure(2)
    if (num is None):
        num=ims.shape[0]
    rn=np.int32(np.sqrt(num))
    cn=np.int32(np.ceil(num/rn))
    for i in range(num):
        py.subplot(rn,cn,i+1)
        py.imshow(ims[i,:,:,0])
        py.axis('off')

    py.show()



def generate_bigger_images(PARS):
    old_dim = np.int32(PARS['image_dim'])
    PARS['old_dim'] = old_dim
    PARS['image_dim'] = np.int32(PARS['big_image_dim'])
    dim_ratio = PARS['image_dim'] / old_dim
    PARS['max_num_blobs'] = PARS['max_num_blobs'] * dim_ratio * dim_ratio
    PARS['num_test'] = PARS['big_num_test']
    test = make_data(PARS['num_test'], PARS)
    test_batch = make_batch(test, old_dim, np.int32(PARS['coarse_disp']))
    PARS['batch_size'] = np.minimum(PARS['batch_size'], test_batch[0].shape[0])

    return (test, test_batch)
    #


def make_batch(test, old_dim, coarse_disp):
    tbatch = []
    for t in test[0]:
        for i in np.arange(0, t.shape[0], old_dim):
            for j in np.arange(0, t.shape[1], old_dim):
                tbatch.append(t[i:i + old_dim, j:j + old_dim, :])

    tbatch = np.array(tbatch)
    cbatch = []
    coarse_dim = np.int32(old_dim / coarse_disp)
    for tt in test[1]:
        for j in np.arange(0, tt.shape[1], coarse_dim):
            for i in np.arange(0, tt.shape[0], coarse_dim):
                cbatch.append(tt[i:i + coarse_dim, j:j + coarse_dim, :])
    cbatch = np.array(cbatch)
    batch = [tbatch, cbatch]

    return (batch)


def paste_batch(HYY, old_dim, new_dim, coarse_disp, nbp):
    num_per = np.int32(new_dim / old_dim)
    num_per2 = num_per * num_per

    cnew_dim = np.int32(new_dim / coarse_disp)
    cold_dim = np.int32(old_dim / coarse_disp)

    HY = []
    for i in np.arange(0, len(HYY), num_per2):
        hy = np.zeros((cnew_dim, cnew_dim, nbp))
        j = 0
        for y in np.arange(0, cnew_dim, cold_dim):
            for x in np.arange(0, cnew_dim, cold_dim):
                for k in range(nbp):
                    hy[x:x + cold_dim, y:y + cold_dim, k] = HYY[i + j][:, :, k]
                j += 1
        HY.append(hy)

    return (HY)


def make_data(num,PARS):
    G=[]
    GC=[]
    num_blobs = np.int32(np.floor(np.random.rand(num) * PARS['max_num_blobs']) + 1)

    for nb in num_blobs:
        g,gc=generate_image(PARS,num_blobs=nb)
        G.append(np.float32(g))
        GC.append(np.float32(gc))

    return([np.array(G),np.array(GC)])

def make_blobs(mux,muy,sigmas, Amps, image_dim):
    x, y = np.meshgrid(range(np.int32(image_dim)), range(np.int32(image_dim)))

    g = np.zeros(x.shape)
    for mx, my, si, a in zip(mux, muy,sigmas,Amps):
        g = g + a*np.exp(-(((x - mx) ** 2 + (y - my) ** 2) / (2.0 * si ** 2)))
        #g = np.maximum(g,np.exp(-(((x - mx) ** 2 + (y - my) ** 2) / (2.0 * sigma ** 2))))
    g = np.reshape(g, g.shape + (1,))

    return(g)

# Find  duplicate detections and merge them (if centers are too close.)
def clean_b(ii,jj,mux,muy,coarse_disp):
    ij=np.ones(mux.shape[0], dtype=bool)
    for i in range(mux.shape[0]):
        for j in range(mux.shape[0]):
            if (i != j and ij[i] and ij[j]):
                if np.sqrt((mux[i]-mux[j])*(mux[i]-mux[j])+(muy[i]-muy[j])*(muy[i]-muy[j])<coarse_disp):
                    ij[j]=False
                    mux[i]=.5*(mux[i]+mux[j])
                    muy[i]=.5*(muy[i]+muy[j])

    mux=mux[ij==1]
    muy=muy[ij==1]
    ii=ii[ij==1]
    jj=jj[ij==1]
    return(ii,jj,mux,muy)

# Make background
def background(n):
    im=np.random.randn(n,n)

    a=np.ones((3,3))/9.

    imc=signal.convolve2d(im,a,'same')
    imc=signal.convolve2d(imc,a,'same')
    imc=signal.convolve2d(imc,a,'same')

    return(imc)

def make_curve(image_dim):

    n=image_dim
    dell=4

    num_points=6*np.int32(image_dim/32)
    num_interp_points=20*np.int32(image_dim/32)
    theta_range=np.pi/3
    x=np.int32(np.round(np.random.rand()*n))
    y=np.int32(np.round(np.random.rand()*n))
    theta_new=np.random.rand()*2*np.pi
    dx=np.int32(np.round(np.cos(theta_new)*dell))
    dy=np.int32(np.round(np.sin(theta_new)*dell))

    C=[]
    theta_old=theta_new
    C.append([x,y])
    x=x+dx
    y=y+dy
    C.append([x,y])
    for i in range(num_points):
        theta_new=(np.random.rand()-.5)*theta_range+theta_old
        x = x+np.int32(np.round(np.cos(theta_new) * dell))
        y = y+np.int32(np.round(np.sin(theta_new) * dell))
        theta_old = theta_new
        C.append([x , y ])


    C=np.array(C)

    #py.plot(C[:,0],C[:,1])
    #py.axis([0,n,0,n])
    npo=C.shape[0]
    tt0=inp.splrep(range(npo),C[:,0],s=1)
    tt1=inp.splrep(range(npo),C[:,1],s=1)

    x = np.linspace(0, npo, num_interp_points)
    y0 = inp.splev(x, tt0)
    y1 = inp.splev(x,tt1)
    # py.plot(y0,y1)
    # py.show()

    l=y0.shape[0]
    g=make_blobs(y0,y1,np.repeat(2,l), np.repeat(1.,l),n)
    g=g/np.max(g)

    return(g)

def generate_image(PARS,num_blobs=1):

    image_dim=PARS['image_dim']
    coarse_disp=PARS['coarse_disp']
    mux=[]
    muy=[]

    mux.append(np.random.rand(1)*image_dim)
    muy.append(np.random.rand(1)*image_dim)

    for i in range(num_blobs-1):
        dist=0
        while (dist<PARS['blob_dist']):
            newmux=np.random.rand(1) * image_dim
            newmuy=np.random.rand(1) * image_dim
            dist=np.min(np.sqrt((newmux-np.array(mux))**2+(newmuy-np.array(muy))**2))
        #print('diss',dist)
        mux.append(newmux)
        muy.append(newmuy)


    sigmas=PARS['sigma'][0]+np.random.rand(num_blobs)*(PARS['sigma'][1]-PARS['sigma'][0])

    As=PARS['Amps'][0]+np.random.rand(num_blobs)*(PARS['Amps'][1]-PARS['Amps'][0])

    g = make_blobs(mux, muy, sigmas, As, image_dim)
    #py.imshow(g[:,:,0])
    #py.show()
    mux = np.array(mux) #* np.float32(image_dim / 2) + image_dim / 2
    muy = np.array(muy) #* np.float32(image_dim / 2) + image_dim / 2
    muxc = np.floor(mux / coarse_disp)
    muyc = np.floor(muy / coarse_disp)
    mucs0 = np.int32(np.array([muxc, muyc, np.zeros(muxc.shape)]))
    mucs1 = np.int32(np.array([muxc, muyc, np.ones(muxc.shape)]))
    mucs2 = np.int32(np.array([muxc, muyc, 2*np.ones(muxc.shape)]))
    mucs3 = np.int32(np.array([muxc, muyc, 3*np.ones(muxc.shape)]))

    gc = np.zeros((np.int32(image_dim/coarse_disp),np.int32(image_dim/coarse_disp), PARS['num_blob_pars']))
    gc[tuple(mucs0)] = (mux - (coarse_disp / 2 + coarse_disp * muxc))
    gc[tuple(mucs1)] = (muy - (coarse_disp / 2 + coarse_disp * muyc))
    if (PARS['num_blob_pars']>=4):
        gc[tuple(mucs2)] = sigmas[:,np.newaxis]
    if (PARS['num_blob_pars']>=5):
        gc[tuple(mucs3)] = As[:,np.newaxis]
    #print('Mean and sd of displacements:',np.mean(np.concatenate([gc[tuple(mucs0)],gc[tuple(mucs1)]])),
    #      np.std(np.concatenate([gc[tuple(mucs0)], gc[tuple(mucs1)]])))

    gc[:,:,PARS['num_blob_pars']-1]=np.logical_or(gc[:,:,0] != 0, gc[:,:,1]!=0)

    if ('background' in PARS and PARS['background']):
        bgd=background(image_dim)
        bgd=bgd[:,:,np.newaxis]
        g=np.maximum(g,bgd)

    if ('curve' in PARS and PARS['curve']):
        cr=make_curve(image_dim)
        g=np.maximum(g,cr)

    return(g,gc)

def extract_mus(hy,ii,jj,coarse_disp,PARS):
    l = len(ii)
    I0 = np.int32(np.concatenate([np.array(ii).reshape((1,-1)), np.array(jj).reshape((1,-1)), np.zeros((1,l))]))
    I1 = np.int32(np.concatenate([np.array(ii).reshape((1,-1)), np.array(jj).reshape((1,-1)), np.ones((1,l))]))
    mux=ii*coarse_disp+coarse_disp/2 + hy[tuple(I0)]
    muy=jj*coarse_disp+coarse_disp/2 + hy[tuple(I1)]

    # clean up close detections
    ii,jj,mux,muy=clean_b(ii,jj,mux,muy,coarse_disp)

    l = len(ii)
    if (hy.shape[-1]>3):
        I = np.int32(np.concatenate([np.array(ii).reshape((1,-1)), np.array(jj).reshape((1,-1)), 2*np.ones((1,l))]))
        sigmas=hy[tuple(I)]
    else:
        sigmas=PARS['sigma'][0]*np.ones(l)
    if (hy.shape[-1]>4):
        I = np.int32(np.concatenate([np.array(ii).reshape((1,-1)), np.array(jj).reshape((1,-1)), 3*np.ones((1,l))]))
        As = hy[tuple(I)]
    else:
        As=PARS['Amps'][0]*np.ones(l)
    return(mux,muy,sigmas,As)

def generate_image_from_estimate(PARS,hy,orig_image,orig_data):

    image_dim=orig_image.shape[0]

    coarse_disp=PARS['coarse_disp']
    #image_dim=PARS['image_dim']
    hys = hy[:,:,PARS['num_blob_pars']-1]>0
    [ii, jj] = np.where(hys > 0)


    mux,muy,sigmas,As=extract_mus(hy,ii,jj,coarse_disp,PARS)


    g=make_blobs(list(mux),list(muy),list(sigmas),list(As),image_dim)
    [It,Jt]=np.where(orig_data[:,:,PARS['num_blob_pars']-1]==1)
    muxt,muyt,sigmast,Ast=extract_mus(orig_data,It,Jt,coarse_disp,PARS)
    origg=make_blobs(list(muxt),list(muyt),list(sigmast),list(Ast),image_dim)
    print('mux')
    print(np.array([mux, muxt]))
    print('muy')
    print(np.array([muy, muyt]))
    print('sigmas')
    print(np.array([sigmas, sigmast]))
    print('As')
    print(np.array([As, Ast]))
    fig=py.figure(1)

    #py.subplot(1,3,1)
    #py.imshow(g[:,:,0])
    #py.title("Reconstruction")
    py.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    #py.subplot(1,3,2)
    py.title("Original")
    py.imshow(orig_image[:,:,0])
    for mx,my,s in zip(mux,muy,sigmas):
        circle=py.Circle((mx,my),radius=s,color="r",fill=False)
        ax.add_artist(circle)
    # py.subplot(1, 2, 2)
    # py.title("Orig R")
    # py.imshow(origg[:, :, 0])
    py.show()
    print("Hello")
    py.close(1)



