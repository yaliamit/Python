import tensorflow as tf
import numpy as np

def convert_vals_to_sparse(SPV):
    SP=None
    if (SPV is not None):
      SP={}
      for key,value in SPV.items():
        WW=[]
        for i in range(2):
            INDS=tf.convert_to_tensor(value[i][0],dtype=np.int64)
            VALS=tf.convert_to_tensor(value[i][1], dtype=np.float32)
            ndims=tf.convert_to_tensor(value[i][2],dtype=np.int64)
            A=tf.SparseTensor(indices=INDS,values=VALS,dense_shape=ndims)
            # if (i==1):
            #     A=tf.sparse_transpose(A)
            WW.append(A)

        SP[key]=WW
    return SP


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
    print('In convert to sparse:','dimin',dimin,'inshape',[inc,]+din,'dout',dout)
    indsaw=[]
    valsaw=[]
    indsar=[]
    valsar=[]
    ii=0
    for t in range(0,dimin,inc):
        s=0
        #print(ii,t)
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


def get_sparse_parameters(VS):
    SS = []
    for v in VS:
        if ('sparse' in v.name):
            SS.append(v)
    return(SS)

def get_sparse_parameters_eval(VGS,PARS):
    SPV=None

    if ('sparse' in PARS):
     SPV={}
     count=0
     lcount=0
     for sp in PARS['sparse']:
        for v in VGS:
            if sp in v.name:
             rw=v.name.split('/')[1][0]
             if rw is not 'F':
               if 'dims' in v.name:
                tdims=v.eval()
                lcount+=1
               elif 'vals' in v.name:
                tvals=v.eval()
                lcount+=1
               elif 'inds' in v.name:
                tinds=v.eval()
                lcount+=1
               if (lcount==3):
                 if rw == 'W':
                   W=[tinds,tvals,tdims]
                   count+=1
                 elif rw =='R':
                   R=[tinds,tvals,tdims]
                   count+=1
                 lcount=0

            if (count==2):
                SPV[sp]=[W,R]
                count=0
    return SPV

def convert_conv_layers_to_sparse(sparse_shape, WRS, sess, PARS):
    SP = {}
    for sp in PARS['sparse']:
        SP[sp] = convert_conv_to_sparse(sparse_shape[sp], WRS[sp], sess, PARS['force_global_prob'][0])
    return (SP)

# Each layer comes in groups of 9 parameters
def F_transpose_and_clip(VS,sess,SDS=None):

    t=0
    for t in np.arange(0,len(VS),9):
        # Clip weights at level corresponding to that layer.
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

def generate_some_images(tt,inp):

    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    F=[2,4]
    X=[18,8,14,10]
    Y=[6,12,14,23]

    for i,x in enumerate(X):
        for f in F:
            ttt=tt[x,Y[i],f,:].reshape(5,5)
            ax.imshow(ttt,cmap="gray")
            ax.axis('off')
            print(x,Y[i],f)
            print(ttt)
            s='fig'+str(inp)+'_'+str(x)+'_'+str(Y[i])+'_'+str(f)
            canvas = FigureCanvasAgg(fig)
            canvas.print_figure(s, dpi=80)


# Compare filters at different spatial locations to see if they are similar after running the sparse version.
def compare_params_sparse(sp, sh, VS, WR):
    for v in VS:
        if (sp in v.name and 'W' in v.name):
            if ('inds' in v.name):
                minds = v.eval()
            if ('vals' in v.name):
                mvals = v.eval()
            if ('dims' in v.name):
                mdims = v.eval()
    sdv=np.std(mvals)
    with tf.device("/cpu:0"):
        dm = tf.sparse_to_dense(sparse_indices=minds, sparse_values=mvals, output_shape=mdims)
        DM = dm.eval()
    outfe=WR[sp][0].shape[3]
    infe=WR[sp][0].shape[2]
    fdims=[WR[sp][0].shape[0],WR[sp][0].shape[1]]
    pfdims=np.prod(fdims)
    numloc=np.int32(DM.shape[0]/outfe)
    newshape=[numloc,outfe,sh[sp][0],sh[sp][1],infe]
    DM = np.reshape(DM, newshape)
    tt=np.zeros((sh[sp][0],sh[sp][1],outfe,pfdims))
    nonz=np.int32(np.zeros((sh[sp][0],sh[sp][1])))

    # Get the filters operating on the input_feature_index'th feature of the input layer.
    # Can do similar thing on each feature of the input layer.
    input_feature_index=0
    me = np.zeros(pfdims * outfe * infe)
    uqa = np.zeros(pfdims * outfe * infe)
    uqb = np.zeros(pfdims * outfe * infe)
    t = 0
    for inp in range(infe):
      for i in range(newshape[0]):
        x=np.int32(np.mod(i,sh[sp][0]))
        y=np.int32(i//sh[sp][1])
        for f in range(newshape[1]):
            ww=np.where(DM[i,f,:,:,inp])
            if (len(ww[0])==pfdims):
                if (f==0):
                    nonz[x,y]=1
                tt[x,y,f,:]=DM[i,f,ww[0],ww[1],inp]
      nonzi=np.where(nonz==1)
      sx=min(nonzi[0])
      ex=max(nonzi[0])+1
      sy=min(nonzi[1])
      ey=max(nonzi[1])+1

      generate_some_images(tt,inp)

      for f in range(outfe):
        for p in range(pfdims):
            ttt=tt[sx:ex,sy:ey,f,p]
            tttx=np.diff(ttt,axis=0)
            ttty=np.diff(ttt,axis=1)
            grada=np.sqrt(tttx[:,:-1]*tttx[:,:-1]+ttty[:-1,:]*ttty[:-1,:])
            gradr=grada/sdv#np.abs(ttt[:-1,:-1])
            me[t]=np.median(gradr)
            uqa[t]=np.percentile(gradr,75.)
            uqb[t]=np.percentile(gradr,90.)

            t+=1
    print(sp+'fgrad:',np.max(me),np.median(me),np.percentile(me,75),np.percentile(me,90))
    print(sp+'fgrad:',np.max(uqa),np.median(uqa),np.percentile(uqa,75),np.percentile(uqa,90))
    print(sp+'fgrad:',np.max(uqb),np.median(uqb),np.percentile(uqb,75),np.percentile(uqb,90))


def get_weight_stats(SS,update=False):
            SDS=[]
            for ss in SS:
                V = ss.eval()
                if (update):
                    SDS.append(np.std(V))
                print(ss.name, ss.get_shape().as_list(), np.mean(V),np.std(V),np.max(np.abs(V)))
            return(SDS)