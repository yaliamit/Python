

import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import data
from keras import backend as K
from keras.layers.convolutional import UpSampling2D


def conv_layer(input,batch_size,nonlin_scale,filter_size=[3,3],num_features=[1],prob=[1.,-1.],scale=0):
    
    # Get number of input features from input and add to shape of new layer
    shape=filter_size+[input.get_shape().as_list()[-1],num_features]
    shapeR=shape
    if (prob[1]==-1.):
        shapeR=[1,1]
    R = tf.get_variable('R',shape=shapeR)
    W = tf.get_variable('W',shape=shape) # Default initialization is Glorot (the one explained in the slides)
    input = tf.reshape(input, shape=[batch_size]+input.get_shape().as_list()[1:])

    #b = tf.get_variable('b',shape=[num_features],initializer=tf.zeros_initializer) 
    conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
    if (scale>0):
        conv = tf.clip_by_value(nonlin_scale * conv, -1., 1.)
    return(conv)

def grad_conv_layer(batch_size,below, back_propped, current, W, R, scale):
    w_shape=W.shape
    strides=[1,1,1,1]
    back_prop_shape=[-1]+(current.shape.as_list())[1:]
    out_backprop=tf.reshape(back_propped,back_prop_shape)
    if (scale>0):
        on_zero = K.zeros_like(out_backprop)
        out_backprop = scale * K.tf.where(tf.equal(tf.abs(current), 1.), on_zero, out_backprop)
    gradconvW=tf.nn.conv2d_backprop_filter(input=below,filter_sizes=w_shape,out_backprop=out_backprop,strides=strides,padding='SAME')
    input_shape=[batch_size]+(below.shape.as_list())[1:]
    
    filter=W
    if (len(R.shape.as_list())==4):
        print('using R')
        filter=R
    print('input_sizes',input_shape,'filter',filter.shape.as_list(),'out_backprop',out_backprop.shape.as_list())
    gradconvx=tf.nn.conv2d_backprop_input(input_sizes=input_shape,filter=filter,out_backprop=out_backprop,strides=strides,padding='SAME')
    
    return gradconvW, gradconvx


# In[3]:


def fully_connected_layer(input,batch_size,nonlin_scale, num_features,prob=[1.,-1.], scale=0):
    # Make sure input is flattened.
    ### TEMP
    #if (len(input.shape.as_list())==4):
    #    input=tf.transpose(input,[0,3,1,2])
    flat_dim=np.int32(np.array(input.get_shape().as_list())[1:].prod())
    input_flattened = tf.reshape(input, shape=[batch_size,flat_dim])
    shape=[flat_dim,num_features]
    shapeR=shape
    if (prob[1]==-1.):
        shapeR=[1]
    R_fc = tf.get_variable('R',shape=shapeR)
    #if (num_features==10):
    #    aa=np.load('../Class/Wnewdensf.npy')
    #else:
    #    aa=np.load('../Class/Wnewdensp.npy')
    #W_fc = tf.get_variable('W',initializer=aa)

    W_fc = tf.get_variable('W',shape=shape)
    fc = tf.matmul(input_flattened, W_fc)
    if (scale>0):
        fc = tf.clip_by_value(nonlin_scale * fc, -1., 1.)

    return(fc)

def grad_fully_connected(below, back_propped, current, W, R, scale=0):

    belowf=tf.contrib.layers.flatten(below)
    # Gradient of weights of dense layer
    if (scale>0):
        on_zero = K.zeros_like(back_propped)
        back_propped = scale * K.tf.where(tf.equal(tf.abs(current), 1.), on_zero, back_propped)
    gradfcW=tf.matmul(tf.transpose(belowf),back_propped)
    # Propagated error to conv layer.
    filter=W
    if (len(R.shape.as_list())==2):
        filter=R
    gradfcx=tf.matmul(back_propped,tf.transpose(filter))
    
    return gradfcW, gradfcx


def MaxPoolingandMask(input,pool_size, stride):

# We are assuming 'SAME' padding with 0's.
    shp=input.shape.as_list()
    ll=[]
    # Create pool_size x pool_size shifts of the data stacked on top of each pixel
    # to represent the pool_size x pool_size window with that pixel as upper left hand corner
    for j in range(pool_size):
        for k in range(pool_size):
            pp=np.int64(np.zeros((4,2)))
            pp[1,:]=[0,j]
            pp[2,:]=[0,k]
            input_pad=tf.pad(input,pp)
            ll.append(input_pad[:,j:j+shp[1],k:k+shp[2],:])

    shifted_images = tf.stack(ll, axis=0)
    # Get the max in each stack
    maxes = tf.reduce_max(shifted_images, axis=0, name='Max')
    # Expand to the stack
    cmaxes = tf.tile(tf.expand_dims(maxes, 0), [pool_size * pool_size, 1, 1, 1, 1])
    # Get the pooled maxes by jumping strides
    pooled = tf.strided_slice(maxes, [0, 0, 0, 0], shp, strides=[1, stride, stride, 1], name='Max')
    # Create the checker board filter based on the stride
    checker = np.zeros([pool_size*pool_size,shp[0]]+shp[1:4], dtype=np.bool)
    checker[:, :, 0::stride, 0::stride, :] = True
    Tchecker = tf.convert_to_tensor(checker) #get_variable(initializer=checker,name='checker',trainable=False)

    # Filter the cmaxes and checker
    JJJ=tf.cast(tf.logical_and(tf.equal(cmaxes,shifted_images),Tchecker),dtype=tf.float32)
    # Reshift the cmaxes so that the stack for each pixel has true for indices corresonding to the upper left corner of each window
    # that used that pixel as the max.
    jjj=[]
    for j in range(pool_size):
        for k in range(pool_size):
            ind = j * pool_size + k
            pp = np.int64(np.zeros((4, 2)))
            pp[1, :] = [j,0]
            pp[2, :] = [k,0]
            jj_pad=tf.pad(JJJ[ind,:,:,:,:],paddings=pp)
            jjj.append(jj_pad[:,0:shp[1],0:shp[2],:])
    # This a pool_sizexpool_size stack of masks one for each location of ulc using the pixel as max.
    mask=tf.stack(jjj,axis=0, name='Equal')

    return pooled, mask


def grad_pool(back_propped, pool, mask, pool_size, stride):
        gradx_pool = tf.reshape(back_propped, [-1] + (pool.shape.as_list())[1:])


        ll = []
        gradx_pool=UpSampling2D(size=[stride,stride])(gradx_pool)
        shp = gradx_pool.shape.as_list()
        # Stack gradx values for different ulc of windows reaching pixel, add those flagged by mask.
        for j in range(pool_size):
            for k in range(pool_size):
                pp = np.int64(np.zeros((4, 2)))
                pp[1, :] = [0,j]
                pp[2, :] = [0,k]
                gradx_pool_pad=tf.pad(gradx_pool, paddings=pp)
                ll.append(gradx_pool_pad[:,j:j+shp[1],k:k+shp[2],:])
        shifted_gradx_pool = tf.stack(ll, axis=0)
        gradx=tf.reduce_sum(tf.multiply(shifted_gradx_pool,mask),axis=0)

        return gradx

def MaxPoolingandMask_old(inputs, pool_size, strides,
                          padding='SAME'):

        pooled = tf.nn.max_pool(inputs, ksize=pool_size, strides=strides, padding=padding)
        upsampled = UpSampling2D(size=strides[1:3])(pooled)
        indexMask = K.tf.equal(inputs, upsampled)
        assert indexMask.get_shape().as_list() == inputs.get_shape().as_list()
        return pooled,indexMask

 
def unpooling(x,mask,strides):
    '''
    do unpooling with indices, move this to separate layer if it works
    1. do naive upsampling (repeat elements)
    2. keep only values in mask (stored indices) and set the rest to zeros
    '''
    on_success = UpSampling2D(size=strides)(x)
    on_fail = K.zeros_like(on_success)
    return K.tf.where(mask, on_success, on_fail)
 
 


def grad_pool_old(back_propped,pool,mask,pool_size):
        gradx_pool=tf.reshape(back_propped,[-1]+(pool.shape.as_list())[1:])
    #gradfcx=tf.reshape(gradfcx_pool,[-1]+(conv.shape.as_list())[1:])
        gradx=unpooling(gradx_pool,mask,pool_size)
        return gradx


def real_drop(parent, drop,batch_size):
    U = tf.less(tf.random_uniform([batch_size] + (parent.shape.as_list())[1:]),drop)
    #parent=tf.reshape(parent,[batch_size] + (parent.shape.as_list())[1:])
    Z = tf.zeros_like(parent)
    fac = tf.constant(1.) / (1. - drop)
    drop = K.tf.where(U, Z, parent * fac)
    return drop

def find_sibling(l,parent,PARS):
      
        for ly in PARS['layers']:
            if ('parent' in ly):
                q=ly['parent']
                if (ly is not l and type(q)==str and q in parent):
                    return q
        return None


def get_name(ts):
    if type(ts) == list:
        name = ts[0].name
        T=ts[0]
    else:
        name = ts.name
        T=ts
    return (name,T)

def create_network(PARS,x,y_,Train):

    EXTRAS=[]
    TS=[]
    ln=len(PARS['layers'])
    sibs={}
    for i,l in enumerate(PARS['layers']):
        parent=None
        prob=[1.,-1.]
        if ('force_global_prob' in PARS):
            prob=list(PARS['force_global_prob'])
        # Last output layer is fully connected to last hidden layer
        if (i==ln-1):
            prob[0]=1.
        if ('parent' in l):
            if ('input' in l['parent']):
                parent=x
            else:
                # Get list of parents
                if (type(l['parent'])==list):
                    parent=[] 
                    for s in l['parent']:
                        for ts in TS:
                            name,T=get_name(ts)
                            if s in name and not PARS['avoid_name'] in name:
                                parent.append(T)
                # Get single parent
                else:
                    for ts in TS:
                        name, T = get_name(ts)
                        if l['parent'] in name and not PARS['avoid_name'] in name:
                            parent=T
        if ('conv' in l['name']):
            scope_name=l['name']
            scale=0
            if ('non_linearity' in l and l['non_linearity']=='tanh'):
                scale=PARS['nonlin_scale']
                scope_name=l['name']+'nonlin'
            with tf.variable_scope(scope_name):
                TS.append(conv_layer(parent,PARS['batch_size'],PARS['nonlin_scale'], filter_size=list(l['filter_size']),num_features=l['num_filters'], prob=prob, scale=scale))
        elif ('dens' in l['name']):
            scope_name = l['name']
            scale = 0
            if('non_linearity' in l and l['non_linearity'] == 'tanh'):
                scale = PARS['nonlin_scale']
                scope_name = l['name'] + 'nonlin'
            with tf.variable_scope(scope_name):
                num_units=l['num_units']
                if ('final' in l):
                    num_units=PARS['n_classes']
                TS.append(fully_connected_layer(parent,PARS['batch_size'],PARS['nonlin_scale'], num_features=num_units,prob=prob,scale=scale))
        elif ('pool' in l['name']):
            with tf.variable_scope(l['name']):
                if (l['pool_size']==l['stride']):
                    pool, mask = MaxPoolingandMask_old(parent, [1]+list(l['pool_size'])+[1],strides=[1]+list(l['stride'])+[1])
                    TS.append([pool,l['pool_size'],l['stride']])
                else:
                    pool, mask = MaxPoolingandMask(parent, l['pool_size'][0],l['stride'][0])
                #EXTRAS.append(SI)
                    TS.append([pool,l['pool_size'][0],l['stride'][0]])
                TS.append(mask)
        elif ('drop' in l['name']):
            with tf.variable_scope(l['name']):
                ffac = 1. / (1. - l['drop'])
                drop=tf.cond(Train,lambda: real_drop(parent,l['drop'],PARS['batch_size']),lambda: parent)
                TS.append([drop,ffac])
        elif ('concatsum' in l['name']):
            with tf.variable_scope(l['name']):
                res_sum=tf.add(parent[0],parent[1])
                TS.append(res_sum)
            # This is a sum layer get its sibling
                joint_parent=find_sibling(l,l['parent'],PARS)
                if (joint_parent is not None):
                    name,T=get_name(TS[-1])
                    sibs[name]=joint_parent

    with tf.variable_scope('loss'):
       if (PARS['hinge']):
         yb=tf.cast(y_,dtype=tf.bool)
         cor=tf.boolean_mask(TS[-1],yb)
         cor = tf.nn.relu(1.-cor)
         #print(y_.shape,cor.shape)
         res=tf.boolean_mask(TS[-1],tf.logical_not(yb)) #tf.subtract(tf.ones_like(yb),yb))
         shp=TS[-1].shape.as_list()
         shp[1]=shp[1]-1
         res=tf.reshape(res,shape=shp)
         res=tf.reduce_sum(tf.nn.relu(1.+res),axis=1)
         #print('res',res.shape)
         loss=tf.reduce_mean(cor+PARS['dep_fac']*res/(PARS['n_classes']-1),name="hinge")
       else:
         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=TS[-1]),name="sm")

        
    # Accuracy computation
    with tf.variable_scope('helpers'):
        correct_prediction = tf.equal(tf.argmax(TS[-1], 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="ACC")
    print('sibs',sibs)
    return loss, accuracy, TS, sibs



def update_only_non_zero(V,gra, step):
    up=V-step*gra
    #up=K.tf.where(tf.equal(V,tf.constant(0.)),V,up)
    assign_op = tf.assign(V,up)
    return assign_op

def back_prop(loss,acc,TS,VS,x,PARS):
    # Get gradient of loss with respect to final output layer using tf gradient
    # The rest will be explicit backprop
    
    gradX=tf.gradients(loss,TS[0])

    gradx=gradX[0]
    lvs=len(VS)
    lts=len(TS)
    vs=0
    ts=0
    OPLIST=[]
    grad_hold_var={}
    parent=None
    all_grad=[]
    if (PARS['debug']):
        all_grad.append(gradx)
    for ts in range(lts):
        T=TS[ts]
        name,T=get_name(T)
        if (ts<lts-1):
                #pre=TS[ts+1]
                prename,pre=get_name(TS[ts+1])
                if (PARS['avoid_name'] in prename):
                    prename,pre=get_name(TS[ts+2])
                    #pre=TS[ts+2]
        else:
            pre=x
        # You have held a gradx from a higher up layer to be added to current one.

        if (parent is not None):

            pp=name.split('/')[0]
            ind=pp.find('nonlin')
            pp=pp[:ind]
            if (parent == pp):
                print(parent,'grad_hold',grad_hold_var[parent])
                gradx=tf.add(gradx,grad_hold_var[parent])
                parent=None
        if ('conv' in name):
            scale=0
            if ('nonlin' in name):
                scale=PARS['nonlin_scale']
            gradconvW, gradx = grad_conv_layer(PARS['batch_size'],below=pre,back_propped=gradx,current=T,W=VS[vs], R=VS[vs+1],scale=scale)
            assign_op_convW = update_only_non_zero(VS[vs],gradconvW,PARS['eta_init'])
            OPLIST.append(assign_op_convW)
            if (len(VS[vs+1].shape.as_list())==4):
                assign_op_convR=update_only_non_zero(VS[vs+1],gradconvW, PARS['Rstep_size'])
                OPLIST.append(assign_op_convR)
            if (PARS['debug']):
                all_grad.append(gradx)
            ts+=1
            vs+=2
        elif ('drop' in name):
            #fac=tf.constant(np.float32(name.split('x')[1]))
            Z = tf.equal(T, tf.constant(0.))
            gradx=K.tf.where(Z,T,tf.multiply(tf.reshape(gradx,T.shape),TS[ts][1]))
            #all_grad.append(Z)
            #all_grad.append(T)
            if (PARS['debug']):
                all_grad.append(gradx)
        elif (PARS['avoid_name'] in name):
            mask=T
            ts+=1
        elif ('Max' in name):
            if (TS[ts][1]==TS[ts][2]):
                gradx=grad_pool_old(gradx,T,mask,[2,2])
            else:
                gradx=grad_pool(gradx,T,mask,pool_size=TS[ts][1],stride=TS[ts][2])

            if (PARS['debug']):
                all_grad.append(gradx)
            ts+=1
        elif ('dens' in name):
            scale = 0
            if ('nonlin' in name):
                scale = PARS['nonlin_scale']
            gradfcW, gradx = grad_fully_connected(below=pre,back_propped=gradx,current=T, W=VS[vs],R=VS[vs+1], scale=scale)
            assign_op_fcW = update_only_non_zero(VS[vs],gradfcW,PARS['eta_init'])
            OPLIST.append(assign_op_fcW)
            if (len(VS[vs+1].shape.as_list())==2):
                assign_op_fcR = update_only_non_zero(VS[vs+1],gradfcW,PARS['Rstep_size'])
                OPLIST.append(assign_op_fcR)
            if (PARS['debug']):
                all_grad.append(gradx)
            ts+=1
            vs+=2
        if (name in PARS['sibs']):
            grad_hold=gradx
            parent=PARS['sibs'][name]
            grad_hold_var[parent]=grad_hold
    if (PARS['debug']):
        print('all_grad',len(all_grad))
        for cg in all_grad:
            OPLIST.append(cg)
    #print('Length of VS',len(VS),'Length of OPLIST',len(OPLIST))
    OPLIST.append(acc)
    OPLIST.append(loss)
    
    return OPLIST, len(all_grad)


def get_data(data_set):
    if ('cifar' in data_set):
        return(data.get_cifar(data_set=data_set))
    elif (data_set=="mnist"):
        return(data.get_mnist())





