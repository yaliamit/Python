import numpy as np
import pylab as py
import PyQt5
import scipy.signal as signal
import scipy.interpolate as inp
from Conv_data import get_data

def add_nois(train,sigma):
    nois = np.random.normal(size=train[0].shape) * sigma
    trainn = train[0]+nois
    return trainn

def add_coordinate(train, val, test, PARS):
    if ('noise_sigma' in PARS and PARS['noise_sigma']>=0):
        trainn = add_nois(train, PARS['noise_sigma'])
        valn = add_nois(val, PARS['noise_sigma'])
        testn = add_nois(test, PARS['noise_sigma'])
        if ('corr' in PARS and PARS['corr']):
            train = (np.array([train[0], trainn]), train[1])
            val=(np.array([val[0],valn]),val[1])
            test=(np.array([test[0],testn]),test[1])
        else:
            train=(np.expand_dims(trainn, axis=0),train[1])
            test=(np.expand_dims(testn,axis=0),test[1])
            val=(np.expand_dims(valn,axis=0),val[1])
    else:
        train = (np.expand_dims(train[0], axis=0), train[1])
        val = (np.expand_dims(val[0],axis=0), val[1])
        test = (np.expand_dims(test[0],axis=0), test[1])
    return train,val,test


def acquire_data(PARS,type='class'):
    if ('blob' in PARS):

        train=make_data(PARS['num_train'],PARS)
        #show_images(train[0],num=100)
        val=make_data(PARS['num_val'],PARS)
        test=make_data(PARS['num_test'],PARS)
        image_dim = PARS['image_dim']
    else:
        train, val, test, image_dim = get_data(PARS)
        train, val, test = add_coordinate(train,val,test,PARS)
    return train, val, test, image_dim

def show_images(ims,num=None):

    py.figure(2)
    py.figure(figsize=(10, 10))
    if (num is None):
        num=ims.shape[0]
    
    rn=np.int32(np.sqrt(num))
    cn=np.int32(np.ceil(num/rn))
    for i in range(num):
        py.subplot(rn,cn,i+1)
        py.imshow(ims[i,:,:,0])
        py.axis('off')

    py.show()
    print('hello')


def add_noise(PARS,gauss_ker,g):

    image_dim=PARS['image_dim']
    if ('background' in PARS and PARS['background']):
        bgd=background(image_dim,gauss_ker,PARS['nchannels'])

        g=np.maximum(g,bgd)

    if ('curve' in PARS and PARS['curve']):
        for c in range(PARS['curve']):
            cr=make_curve(image_dim,PARS)
            g=np.maximum(g,cr)
    return(g)

def generate_bigger_images(PARS):
    old_dim = np.int32(PARS['image_dim'])
    PARS['old_dim'] = old_dim
    PARS['image_dim'] = np.int32(PARS['big_image_dim'])
    dim_ratio = PARS['image_dim'] / old_dim
    PARS['max_num_blobs'] = PARS['max_num_blobs'] * dim_ratio * dim_ratio
    PARS['num_test'] = PARS['big_num_test']
    test = make_data(PARS['num_test'], PARS)
    test_batch = make_batch(test, old_dim, np.int32(PARS['coarse_disp']))
    PARS['batch_size'] = np.minimum(PARS['batch_size'], test_batch[0].shape[1])

    return (test, test_batch)
    #


def make_batch(test, old_dim, coarse_disp):
    tbatch = []
    for t in test[0][0]:
        for i in np.arange(0, t.shape[0], old_dim):
            for j in np.arange(0, t.shape[1], old_dim):
                tbatch.append(t[i:i + old_dim, j:j + old_dim, :])

    tbatch = np.expand_dims(np.array(tbatch),axis=0)
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

    if ('blob' in PARS):
        G=[]
        GC=[]
        num_blobs = np.int32(np.floor(np.random.rand(num) * PARS['max_num_blobs']) + 1)
        gauss_ker=None
        if ('background' in PARS and PARS['background']):
            gauss_ker=make_gauss(PARS)
        for nb in num_blobs:
            g,gc=generate_image(PARS,num_blobs=nb)
            gnoise=add_noise(PARS,gauss_ker,g)
            if ('corr' in PARS and PARS['corr']):
                g=np.stack([g,gnoise])
                #gc=np.stack([gc,gc])
            else:
                g=gnoise
            G.append(np.float32(g))
            GC.append(np.float32(gc))
        G=np.array(G)

        GC=np.array(GC)
        if len(G.shape)==5:
            G=G.transpose((1,0,2,3,4))
        else:
            G = np.expand_dims(G, axis=0)
            #GC=GC.transpose((1,0,2,3,4))
        return(G,GC)


def make_gauss(PARS):
    image_dim=32

    x, y = np.meshgrid(range(np.int32(image_dim)), range(np.int32(image_dim)))

    g = np.zeros(x.shape)
    mx=16; my=16; si=PARS['noise_sigma']

    if (si==0):
        g[16,16]=1.
        sin=1
    else:
        g = np.exp(-(((x - mx) ** 2 + (y - my) ** 2) / (2.0 * si ** 2)))
        g=g/np.sqrt(np.sum(g*g))
        sin=np.int32(np.round(3*si))
    gg=PARS['background']*g[16-sin:16+sin+1,16-sin:16+sin+1]
    s=gg.shape+(PARS['nchannels'],)
    gg=np.dot(np.ones((PARS['nchannels'],1)),gg.ravel().reshape((1,-1))).transpose().reshape(s)
    return(gg)


cols=[[1.,0,0],[0.,1.,0],[0,0,1.]]

def make_blobs(mux,muy,sigmas, Amps, image_dim, nchannels=1):
    x, y = np.meshgrid(range(np.int32(image_dim)), range(np.int32(image_dim)))

    g = np.zeros(x.shape+(nchannels,))
    for mx, my, si, a in zip(mux, muy,sigmas,Amps):
        gt = a*np.exp(-(((x - mx) ** 2 + (y - my) ** 2) / (2.0 * si ** 2)))
        if (nchannels==3):
            col=np.array(cols[np.int32(np.floor(np.random.rand()*3))]).reshape(3,1)
            gt=np.dot(col,gt.ravel().reshape((1,-1))).transpose().reshape(g.shape)
        else:
            gt=gt.reshape(gt.shape+(1,))
        g = g + gt
        #g = np.maximum(g,np.exp(-(((x - mx) ** 2 + (y - my) ** 2) / (2.0 * sigma ** 2))))
    #g = np.reshape(g, g.shape + (1,))

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
def background(n,gauss_ker,nchannels):
    imc=np.random.randn(n,n,nchannels)

    for i in range(nchannels):
        imc[:,:,i]=signal.convolve2d(imc[:,:,i],gauss_ker[:,:,i],'same')

    return(imc)

def make_curve(image_dim,PARS):

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
    g=make_blobs(y0,y1,np.repeat(PARS['curve_width'],l), np.repeat(1.,l),n,nchannels=PARS['nchannels'])
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

    g = make_blobs(mux, muy, sigmas, As, image_dim, PARS['nchannels'])
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
    hys = hy[:,:,PARS['num_blob_pars']-1]>PARS['thresh']
    [ii, jj] = np.where(hys > 0)


    mux,muy,sigmas,As=extract_mus(hy,ii,jj,coarse_disp,PARS)


    g=make_blobs(list(mux),list(muy),list(sigmas),list(As),image_dim, PARS['nchannels'])
    [It,Jt]=np.where(orig_data[:,:,PARS['num_blob_pars']-1]==1)
    muxt,muyt,sigmast,Ast=extract_mus(orig_data,It,Jt,coarse_disp,PARS)

    print('mux')
    print(np.array([mux, muxt]))
    print('muy')
    print(np.array([muy, muyt]))
    print('sigmas')
    print(np.array([sigmas, sigmast]))
    print('As')
    print(np.array([As, Ast]))
    print(py.get_backend())
    py.ion()
    fig=py.figure(1)

    ax = fig.add_subplot(1, 1, 1)
    py.title("Original")
    py.imshow(orig_image[0,:,:,0])
    for mx,my,s in zip(mux,muy,sigmas):
        circle=py.Circle((mx,my),radius=s,color="r",fill=False)
        ax.add_artist(circle)
    py.show()
    print("Hello")
    #py.close(1)



