import tensorflow as tf
import numpy as np

def convert_conv_to_sparse(dshape,WR,sess,prob=None):


    # for ts in TS:
    #     if (type(ts) is not list and name in ts.name):
    #         dshape=ts.get_shape().as_list()[1:3]
    W=WR[0]
    R=WR[1]
    doR=len(R.shape)>2
    Wt=tf.convert_to_tensor(np.float32(W))
    Rt=tf.convert_to_tensor(np.float32(R))
    wshape=W.shape
    infe=wshape[2]
    outfe=wshape[3]
    din=dshape+[infe,]
    dimin=np.prod(din)
    dout=din[0:2]+[outfe,]
    dimout=np.prod(dout)


    XX=np.zeros([dimin,]+din)
    t=0
    for i in range(din[0]):
        for j in range(din[1]):
            for k in range(din[2]):
                XX[t,i,j,k]=1
                t+=1

    fac=din[0]
    inci=1
    inc=np.int32(dimin/fac)
    print('dimin',dimin,'inshape',[inc,]+din,'dout',dout)
    indsaw=[]
    valsaw=[]
    indsar=[]
    valsar=[]
    ii=0
    for t in range(0,dimin,inc):
        s=0
        print(ii,t)
        XX=np.zeros([inc,]+din)
        for i in np.arange(ii,ii+inci,1):
            for j in range(din[1]):
                for k in range(din[2]):
                    XX[s,i,j,k]=1
                    s+=1
        ii+=inci
        batch=tf.convert_to_tensor(np.float32(XX))
        # if (inc>500):
        #     steps=np.arange(0,inc,500)
        #     if (steps[-1]+)

        with tf.device("/cpu:0"):
            outw = sess.run(tf.nn.conv2d(batch,Wt,strides=[1,1,1,1],padding='SAME'))
        outw=np.reshape(outw,(inc,-1))
        valsw=outw[outw!=0]
        indsw=np.array(np.where(outw!=0))
        indsw[0]=indsw[0]+t
        indsaw.append(indsw.transpose())
        valsaw.append(valsw)
        if (doR):
            outr = sess.run(tf.nn.conv2d(batch,Rt,strides=[1,1,1,1],padding='SAME'))
            outr=np.reshape(outr,(inc,-1))
            valsr=outr[outr!=0]
            indsr=np.array(np.where(outr!=0))
            indsr[0]=indsr[0]+t
            indsar.append(indsr.transpose())
            valsar.append(valsr)
    cindsaw = np.concatenate(indsaw,axis=0)
    cvalsaw = np.concatenate(valsaw,axis=0)
    if (prob is not None and prob<1.):
            clen=cindsaw.shape[0]
            U = np.random.rand(clen)
            ii = np.where(U<prob)[0]
            cindsaw = cindsaw[ii,:]
            cvalsaw=cvalsaw[ii]
    INDSW=tf.convert_to_tensor(cindsaw,dtype=np.int64)
    VALSW=tf.convert_to_tensor(cvalsaw, dtype=np.float32)
    ndims=tf.convert_to_tensor([dimin,dimout],dtype=np.int64)

    SPW=tf.SparseTensor(indices=INDSW,values=VALSW,dense_shape=ndims)
    SPW=tf.sparse_transpose(SPW)
    SPR=None
    if (doR):
        cindsar = np.concatenate(indsar, axis=0)
        cvalsar = np.concatenate(valsar, axis=0)
        if (prob is not None and prob < 1.):
            clen = cindsar.shape[0]
            U = np.random.rand(clen)
            ii = np.where(U < prob)[0]
            cindsar = cindsar[ii, :]
            cvalsar = cvalsar[ii]
        INDSR=tf.convert_to_tensor(cindsar,dtype=np.int64)
        VALSR=tf.convert_to_tensor(cvalsar, dtype=np.float32)
        ndims=tf.convert_to_tensor([dimin,dimout],dtype=np.int64)

        SPR=tf.SparseTensor(indices=INDSR,values=VALSR,dense_shape=ndims)
        SPR=tf.sparse_transpose(SPR)

    return(SPW, SPR)

# Each layer comes in groups of 9 parameters
def F_transpose_and_clip(VS,sess,SDS=None):

    t=0
    for t in np.arange(0,len(VS),9):
        if (SDS is not None):
                sess.run(tf.assign(VS[t+7],tf.clip_by_value(VS[t+7],-SDS[t+7],SDS[t+7])))
                sess.run(tf.assign(VS[t + 4], tf.clip_by_value(VS[t + 4], -SDS[t + 4], SDS[t + 4])))
        # Indicates there is a real R feedback tensor. Otherwise the length is 1
        if (VS[t+8].get_shape().as_list()[0]==2):
            finds=VS[t+6]
            fvals=VS[t+7]
            fdims=VS[t+8]
        else:
            finds=VS[t+3]
            fvals=VS[t+4]
            fdims=VS[t+5]

        F=tf.SparseTensor(indices=finds,values=fvals,dense_shape=fdims)
        F=tf.sparse_transpose(F)

        sess.run(tf.assign(VS[t+0],F.indices))
        sess.run(tf.assign(VS[t+1],F.values))
        sess.run(tf.assign(VS[t+2],F.dense_shape))



def compare_params_sparse(sp, sh, VS, WR):
    for v in VS:
        if (sp in v.name and 'W' in v.name):
            if ('inds' in v.name):
                minds = v.eval()
            if ('vals' in v.name):
                mvals = v.eval()
            if ('dims' in v.name):
                mdims = v.eval()
    with tf.device("/cpu:0"):
        dm = tf.sparse_to_dense(sparse_indices=minds, sparse_values=mvals, output_shape=mdims)
        DM = dm.eval()
    outfe=WR[sp][0].shape[3]
    infe=WR[sp][0].shape[2]
    fdims=[WR[sp][0].shape[0],WR[sp][0].shape[1]]
    pfdims=np.prod(fdims)
    newshape=[np.int32(DM.shape[0]/outfe),outfe,sh[sp][0],sh[sp][1],infe]
    DM = np.reshape(DM, newshape)
    tt = [[] for i in range(outfe)]
    for i in range(newshape[0]):
        for f in range(newshape[1]):
            ww=np.where(DM[i,f,:,:,0])
            if (len(ww[0])==pfdims):
                tt[f].append(DM[i,f,ww[0],ww[1],0])
    print('sp',sp)
    for f in range(outfe):
        tt[f]=np.array(tt[f])
        print(f,np.max(np.std(tt[f],axis=0)/np.abs(np.mean(tt[f],axis=0))))

def get_weight_stats(SS):
            for ss in SS:
                V = ss.eval()
                #SDS.append(np.std(V))
                print(ss.name, ss.get_shape().as_list(), np.mean(V),np.std(V))