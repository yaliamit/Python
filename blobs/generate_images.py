import numpy as np
import pylab as py
import Conv_net_aux


def make_blobs(mux,muy,sigma,image_dim):
    #x, y = np.meshgrid(np.linspace(-1,1,image_dim), np.linspace(-1,1,image_dim))
    x, y = np.meshgrid(range(np.int32(image_dim)), range(np.int32(image_dim)))

    g = np.zeros(x.shape)
    for mx, my in zip(mux, muy):
        g = g + np.exp(-(((x - mx) ** 2 + (y - my) ** 2) / (2.0 * sigma ** 2)))
        #g = np.maximum(g,np.exp(-(((x - mx) ** 2 + (y - my) ** 2) / (2.0 * sigma ** 2))))
    g = np.reshape(g, g.shape + (1,))

    return(g)

def clean_b(mux,muy):
    ij=np.ones(mux.shape[0], dtype=bool)
    for i in range(mux.shape[0]):
        for j in range(mux.shape[0]):
            if (i != j and ij[i] and ij[j]):
                if np.sqrt((mux[i]-mux[j])*(mux[i]-mux[j])+(muy[i]-muy[j])*(muy[i]-muy[j])<coarse_disp*2./3.):
                    ij[j]=False
                    mux[i]=.5*(mux[i]+mux[j])
                    muy[i]=.5*(muy[i]+muy[j])

    mux=mux[ij==1]
    muy=muy[ij==1]
    return(mux,muy)


def generate_image(PARS,num_blobs=1):

    image_dim=PARS['image_dim']
    coarse_disp=PARS['coarse_disp']
    mux=[]
    muy=[]
    #mux.append(np.random.rand(1)*2-1)
    #muy.append(np.random.rand(1)*2-1)

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
    g = make_blobs(mux, muy, PARS['sigma'], image_dim)
    #py.imshow(g[:,:,0])
    #py.show()
    mux = np.array(mux) #* np.float32(image_dim / 2) + image_dim / 2
    muy = np.array(muy) #* np.float32(image_dim / 2) + image_dim / 2
    muxc = np.floor(mux / coarse_disp)
    muyc = np.floor(muy / coarse_disp)
    mucs0 = np.int32(np.array([muxc, muyc, np.zeros(muxc.shape)]))
    mucs1 = np.int32(np.array([muxc, muyc, np.ones(muxc.shape)]))

    gc = np.zeros((np.int32(image_dim/coarse_disp),np.int32(image_dim/coarse_disp), 3))
    gc[tuple(mucs0)] = (mux - (coarse_disp / 2 + coarse_disp * muxc))
    gc[tuple(mucs1)] = (muy - (coarse_disp / 2 + coarse_disp * muyc))
    #print('Mean and sd of displacements:',np.mean(np.concatenate([gc[tuple(mucs0)],gc[tuple(mucs1)]])),
    #      np.std(np.concatenate([gc[tuple(mucs0)], gc[tuple(mucs1)]])))
    gc[:,:,2]=np.logical_or(gc[:,:,0] != 0, gc[:,:,1]!=0)
    return(g,gc)

def generate_image_from_estimate(PARS,hy,orig_image):

    coarse_disp=PARS['coarse_disp']
    image_dim=PARS['image_dim']
    hys = hy[:,:,2]>0
    [ii, jj] = np.where(hys > 0)
    l=len(ii)
    I0 = np.int32(np.concatenate([np.array(ii).reshape((1,-1)), np.array(jj).reshape((1,-1)), np.zeros((1,l))]))
    I1 = np.int32(np.concatenate([np.array(ii).reshape((1,-1)), np.array(jj).reshape((1,-1)), np.ones((1,l))]))



    mux=ii*coarse_disp+coarse_disp/2 + hy[tuple(I0)]
    muy=jj*coarse_disp+coarse_disp/2 + hy[tuple(I1)]

    # clean up close detections
    #mux,muy=clean_b(mux,muy)


    g=make_blobs(list(mux),list(muy),PARS['sigma'],image_dim)

    py.subplot(1,2,1)
    py.imshow(g[:,:,0])
    py.subplot(1,2,2)
    py.imshow(orig_image[:,:,0])
    py.show()
    print("Hello")



