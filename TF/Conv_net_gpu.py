
# coding: utf-8

# In[1]:


import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import sys
import mnist
# In[2]:


def conv_layer(input,filter_size=[3,3],num_features=[1],prob=[1.,-1.]):
    
    # Get number of input features from input and add to shape of new layer
    shape=filter_size+[input.get_shape().as_list()[-1],num_features]
    shapeR=shape
    if (prob[1]==-1.):
        shapeR=[1,1]
    R = tf.get_variable('R',shape=shapeR)
    W = tf.get_variable('W',shape=shape) # Default initialization is Glorot (the one explained in the slides)
    
    #b = tf.get_variable('b',shape=[num_features],initializer=tf.zeros_initializer) 
    conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.clip_by_value(conv,-1.,1.)
    
    return(conv)

def grad_conv_layer(below, back_propped, current, W, R):
    w_shape=W.shape
    strides=[1,1,1,1]
    back_prop_shape=[-1]+(current.shape.as_list())[1:]
    out_backprop=tf.reshape(back_propped,back_prop_shape)
    #on_zero = K.zeros_like(out_backprop)
    out_backpropF=out_backprop #K.tf.where(tf.equal(tf.abs(current),1.),on_zero,out_backprop)
    gradconvW=tf.nn.conv2d_backprop_filter(input=below,filter_sizes=w_shape,out_backprop=out_backpropF,strides=strides,padding='SAME')
    input_shape=[batch_size]+(below.shape.as_list())[1:]
    
    filter=W
    if (len(R.shape.as_list())==4):
        print('using R')
        filter=R
    print('input_sizes',input_shape,'filter',filter.shape.as_list(),'out_backprop',out_backprop.shape.as_list())
    gradconvx=tf.nn.conv2d_backprop_input(input_sizes=input_shape,filter=filter,out_backprop=out_backpropF,strides=strides,padding='SAME')
    
    return gradconvW, gradconvx


# In[3]:


def fully_connected_layer(input,num_features,prob=[1.,-1.]):
    # Make sure input is flattened.
    flat_dim=np.int32(np.array(input.get_shape().as_list())[1:].prod())
    input_flattened = tf.reshape(input, shape=[batch_size,flat_dim])
    shape=[flat_dim,num_features]
    shapeR=shape
    if (prob[1]==-1.):
        shapeR=[1]
    R_fc = tf.get_variable('R',shape=shapeR)
    W_fc = tf.get_variable('W',shape=shape)

    #b_fc = tf.get_variable('b',shape=[num_features],initializer=tf.zeros_initializer)
    fc = tf.matmul(input_flattened, W_fc) # + b_fc
    return(fc)

def grad_fully_connected(below, back_propped, W, R):
    
    belowf=tf.contrib.layers.flatten(below)
    # Gradient of weights of dense layer
    gradfcW=tf.matmul(tf.transpose(belowf),back_propped)
    # Propagated error to conv layer.
    filter=W
    if (len(R.shape.as_list())==2):
        filter=R
    gradfcx=tf.matmul(back_propped,tf.transpose(filter))
    
    return gradfcW, gradfcx


# In[4]:


from keras import backend as K
from keras.layers.convolutional import UpSampling2D
 
def MaxPoolingandMask(inputs, pool_size, strides,
                          padding='SAME'):

        pooled = tf.nn.max_pool(inputs, ksize=pool_size, strides=strides, padding=padding)
        upsampled = UpSampling2D(size=strides[1:3])(pooled)
        indexMask = K.tf.equal(inputs, upsampled)
        assert indexMask.get_shape().as_list() == inputs.get_shape().as_list()
        return pooled,indexMask
     
#def get_output_shape_for(self, input_shape):
#        return input_shape
 
 
def unpooling(x,mask,strides):
    '''
    do unpooling with indices, move this to separate layer if it works
    1. do naive upsampling (repeat elements)
    2. keep only values in mask (stored indices) and set the rest to zeros
    '''
    on_success = UpSampling2D(size=strides)(x)
    on_fail = K.zeros_like(on_success)
    return K.tf.where(mask, on_success, on_fail)
 
 


def grad_pool(back_propped,pool,mask,pool_size):
        gradx_pool=tf.reshape(back_propped,[-1]+(pool.shape.as_list())[1:])
    #gradfcx=tf.reshape(gradfcx_pool,[-1]+(conv.shape.as_list())[1:])
        gradx=unpooling(gradx_pool,mask,pool_size)
        return gradx


# In[5]:


def find_sibling(l,parent):
      
        for ly in PARS['layers']:
            if ('parent' in ly):
                q=ly['parent']
                if (ly is not l and type(q)==str and q in parent):
                    return q
        return None  

def create_network(PARS):
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
                            if s in ts.name and not 'Equal' in ts.name:
                                parent.append(ts)
                # Get single parent
                else:
                    for ts in TS:
                        if l['parent'] in ts.name and not 'Equal' in ts.name:
                            parent=ts
        if ('conv' in l['name']):
            with tf.variable_scope(l['name']):
                TS.append(conv_layer(parent, filter_size=list(l['filter_size']),num_features=l['num_filters'], prob=prob))
        elif ('dens' in l['name']):
            with tf.variable_scope(l['name']):
                num_units=l['num_units']
                if ('final' in l):
                    num_units=n_classes
                TS.append(fully_connected_layer(parent, num_features=num_units,prob=prob))
        elif ('pool' in l['name']):
            with tf.variable_scope(l['name']):
                pool, mask = MaxPoolingandMask(parent, [1]+list(l['pool_size'])+[1],                                           strides=[1]+list(l['stride'])+[1])
                TS.append(pool)
                TS.append(mask)
        elif ('drop' in l['name']):
            with tf.variable_scope(l['name']):
                U=tf.random_uniform([batch_size]+(parent.shape.as_list())[1:])<l['drop']
                Z=tf.zeros_like(parent)
                fac=1/(1-l['drop'])
                drop = K.tf.where(U,Z,parent*fac,name='probx{:.1f}x'.format(fac))
                TS.append(drop)
        elif ('concatsum' in l['name']):
            with tf.variable_scope(l['name']):
                res_sum=tf.add(parent[0],parent[1])
                TS.append(res_sum)
            # This is a sum layer get its sibling
                joint_parent=find_sibling(l,l['parent'])
                if (joint_parent is not None):
                    sibs[TS[-1].name]=joint_parent
    
    with tf.variable_scope('loss'):
       if (PARS['hinge']):
         cor=tf.boolean_mask(TS[-1],y_)
         cor = tf.nn.relu(1.-cor)
         #print(y_.shape,cor.shape)
         res=tf.boolean_mask(TS[-1],tf.subtract(tf.ones_like(y_),y_))
         shp=TS[-1].shape.as_list()
         shp[1]=shp[1]-1
         res=tf.reshape(res,shape=shp)
         res=tf.reduce_sum(tf.nn.relu(1.+res),axis=1)
         #print('res',res.shape)
         loss=tf.reduce_mean(cor+PARS['dep_fac']*res/(n_classes-1),name="hinge")
       else:
         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=TS[-1]),name="sm")

        
    # Accuracy computation
    with tf.variable_scope('helpers'):
        correct_prediction = tf.equal(tf.argmax(TS[-1], 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="ACC")
    print('sibs',sibs)
    return loss, accuracy, TS, sibs


# In[6]:


def update_only_non_zero(V,gra, step):
    up=V-step*gra
    up=K.tf.where(V==0,V,up)
    assign_op = tf.assign(V,up)
    return assign_op

def back_prop(): 
    # Get gradient of loss with respect to final output layer using tf gradient
    # The rest will be explicit backprop
    
    gradX=tf.gradients(loss,TS[0])
    corr = tf.equal(tf.argmax(TS[0], 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(corr, tf.float32))

    gradx=gradX[0]
    lvs=len(VS)
    lts=len(TS)
    vs=0
    ts=0
    OPLIST=[]
    grad_hold_var={}
    parent=None
    for ts in range(lts):
        T=TS[ts]
        if (ts<lts-1):
                pre=TS[ts+1]
                if ('Equal' in pre.name):
                    pre=TS[ts+2]
        else:
            pre=x
        # You have held a gradx from a higher up layer to be added to current one.
        if (parent is not None and parent == T.name.split('/')[0]):
            print(parent,'grad_hold',grad_hold_var[parent])
            gradx=tf.add(gradx,grad_hold_var[parent])
            parent=None
        if ('conv' in T.name):  
            gradconvW, gradx = grad_conv_layer(below=pre,back_propped=gradx,current=TS[ts],W=VS[vs], R=VS[vs+1])
            assign_op_convW = update_only_non_zero(VS[vs],gradconvW,step_size)
            #assign_op_convW=tf.assign(VS[vs],VS[vs]-step_size*gradconvW)
            OPLIST.append(assign_op_convW)
            if (len(VS[vs+1].shape.as_list())==4):
                assign_op_convR=update_only_non_zero(VS[vs+1],gradconvW, Rstep_size)
                #assign_op_convR=tf.assign(VS[vs+1],VS[vs+1]-Rstep_size*gradconvW)
                OPLIST.append(assign_op_convR)
            ts+=1
            vs+=2
        elif ('drop' in T.name):
            fac=np.float32(T.name.split('x')[1])
            gradx=gradx*fac
        elif ('Equal' in T.name):
            mask=TS[ts]
            ts+=1
        elif ('Max' in T.name):
            gradx=grad_pool(gradx,TS[ts],mask,[2,2])  
            ts+=1
        elif ('dens' in T.name):
            gradfcW, gradx = grad_fully_connected(W=VS[vs],R=VS[vs+1],back_propped=gradx,below=pre)
            assign_op_fcW = update_only_non_zero(VS[vs],gradfcW,step_size)
            #assign_op_fcW=tf.assign(VS[vs],VS[vs]-step_size*gradfcW)
            OPLIST.append(assign_op_fcW)
            if (len(VS[vs+1].shape.as_list())==2):
                assign_op_fcR = update_only_non_zero(VS[vs+1],gradfcW,Rstep_size)
                #assign_op_fcR=tf.assign(VS[vs+1],VS[vs+1]-Rstep_size*gradfcW)
                OPLIST.append(assign_op_fcR)
            ts+=1
            vs+=2
        if (T.name in sibs):
            grad_hold=gradx
            parent=sibs[T.name]
            grad_hold_var[parent]=grad_hold


    #print('Length of VS',len(VS),'Length of OPLIST',len(OPLIST))
    OPLIST.append(acc)
    OPLIST.append(loss)
    
    return OPLIST


# In[7]:


import h5py

def one_hot(values,n_values=10):
    n_v = np.maximum(n_values,np.max(values) + 1)
    oh=np.float32(np.eye(n_v)[values])
    return oh

def get_mnist():
    tr, trl, val, vall, test, testl = mnist.load_dataset()
    trl=one_hot(trl)
    vall=one_hot(vall)
    testl=one_hot(testl)
    return (tr,trl), (val,vall), (test,testl)

def get_cifar(data_set='cifar10'):
    
    filename = '../_CIFAR100/'+data_set+'_train.hdf5'
    print(filename)
    f = h5py.File(filename, 'r')
    key = list(f.keys())[0]
    # Get the data
    tr = f[key]
    print('tr',tr.shape)
    key = list(f.keys())[1]
    tr_lb=f[key]
    train_data=np.float32(tr[0:45000])/255.
    train_labels=one_hot(np.int32(tr_lb[0:45000]))
    val_data=np.float32(tr[45000:])/255.
    val_labels=one_hot(np.int32(tr_lb[45000:]))
    filename = '../_CIFAR100/'+data_set+'_test.hdf5'
    f = h5py.File(filename, 'r')
    key = list(f.keys())[0]
    # Get the data
    test_data = np.float32(f[key])/255.
    key = list(f.keys())[1]
    test_labels=one_hot(np.int32(f[key]))
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)




def get_data(data_set):
    if ('cifar' in data_set):
        return(get_cifar(data_set=data_set))
    elif (data_set=="mnist"):
        return(get_mnist())


# In[8]:


# Function to get loss and accuracy from only one run of the feature extraction network
#from scipy.special import logsumexp


# Run the iterations of one epoch
def run_epoch(train,Tr=True):
        t1=time.time()
        # Randomly shuffle the training data
        ii = np.arange(0, train[0].shape[0], 1)
        if (Tr):
            np.random.shuffle(ii)
        tr=train[0][ii]
        y=train[1][ii]
        lo=0.
        acc=0.
        ca=0.
        for j in np.arange(0,len(y),batch_size):
            batch=(tr[j:j+batch_size],y[j:j+batch_size])
            if (Tr):
                grad=sess.run(dW_OPs,feed_dict={x: batch[0], y_: batch[1]})
            else:
                grad=sess.run(dW_OPs[-2:], feed_dict={x:batch[0],y_:batch[1]})
            # print(j,grad[-1])
            acc+=grad[-2]
            lo+=grad[-1]
            ca+=1
        print('Epoch time',time.time()-t1)
        return acc/ca, lo/ca


def run_epoch_test(test):
    t1 = time.time()
    # Randomly shuffle the training data

    tr = test[0]
    y = test[1]
    lo = 0.
    acc = 0.
    ca = 0.
    for j in np.arange(0, len(y), batch_size):
        batch = (tr[j:j + batch_size], y[j:j + batch_size])
        act, lot = sess.run([accuracy,loss], feed_dict={x: batch[0], y_: batch[1]})
        #print(j, lot)
        acc += act
        lo += lot
        ca += 1
    print('Epoch time', time.time() - t1)
    return acc / ca, lo / ca

# In[9]:


def zero_out_weights():
        if (PARS['force_global_prob'][1]>=0 and PARS['force_global_prob'][0]<1.):
            print('Zeroing out weights at rate ',PARS['force_global_prob'][0])
            shape=v.get_shape().as_list()
            Z=tf.zeros(shape)
            U=tf.random_uniform(shape)
            zero_op=tf.assign(v,K.tf.where(U<PARS['force_global_prob'][0],v,Z))
            sess.run(zero_op)

# Run the training
import parse_net_pars as pp
import time
PARS={}

net=sys.argv[1]#'fncrc_try' #'fncrc_deep_tryR_avg'
print('net',net)
pp.parse_text_file(net,PARS,lname='layers', dump=True)
batch_size=PARS['batch_size']
step_size=PARS['eta_init']
num_epochs=PARS['num_epochs']
num_train=PARS['num_train']
data_set=PARS['data_set']
Rstep_size=list(PARS['force_global_prob'])[1]*step_size
print('Rstep_size',Rstep_size)

model_name="model"

train,val,test=get_data(data_set=data_set)
num_train=np.minimum(num_train,train[0].shape[0])
dim=train[0].shape[1]
nchannels=train[0].shape[3]
n_classes=train[1].shape[1]
print('n_classes',n_classes,'dim',dim,'nchannels',nchannels)
    
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, dim, dim, nchannels],name="x")
y_ = tf.placeholder(tf.float32, shape=[None,n_classes],name="y")


with tf.Session() as sess:
    
    # Create the network architecture with the above placeholdes as the inputs.
    
    loss, accuracy, TS, sibs =create_network(PARS) 
    TS.reverse()
    for t in TS:
        print(t)
    print(loss)
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    
    # Show trainable variables
    VS=tf.trainable_variables()
    VS.reverse()
    for v in VS:
        print(v.name,v.get_shape().as_list(),np.std(v.eval()))
        zero_out_weights()
    # Differences between W and R
    for t in np.arange(0,len(VS),2):
       print('t',t,'zeros',np.sum(VS[t].eval()==0), np.max(np.abs(VS[t].eval()-VS[t+1].eval())))
    dW_OPs=back_prop() 

   
    # Run epochs
    AC=[]
    VAC=[]
    ac, lo = run_epoch_test(test)
    print("Final results: before training")
    print("Test loss:\t\t\t{:.6f}".format(lo))
    print("Test acc:\t\t\t{:.6f}".format(ac))
    for i in range(num_epochs):  # number of epochs
        #ac,lo=\
        ac,lo=run_epoch(train)
        if (np.mod(i,1)==0):
            #lo,ac = get_stats(train[0][0:num_train],train[1][0:num_train],TS[0])
            #ac, lo = run_epoch_test(train)
            AC.append(ac)
            print("Final results: epoch",i)
            print("Train loss:\t\t\t{:.6f}".format(lo))
            print("Train acc:\t\t\t{:.6f}".format(ac))
            #vlo,vac = get_stats(val[0],val[1],TS[0])
            vac, vlo = run_epoch_test(val)
            VAC.append(vac)
            print("Final results: epoch", i)
            print("Val loss:\t\t\t{:.6f}".format(vlo))
            print("Val acc:\t\t\t{:.6f}".format(vac))
            #print('EPoch',i,'Validation loss, accuracy',vlo,vac)
            sys.stdout.flush()

    AC=np.array(AC)
    VAC=np.array(VAC)
    ac, lo = run_epoch(test,Tr=False)
    print("Final results: epoch", i)
    print("Test loss:\t\t\t{:.6f}".format(lo))
    print("Test acc:\t\t\t{:.6f}".format(ac))

    print('step', 0, 'aggegate accuracy', ac)
    #plt.plot(AC)
    #plt.plot(VAC)
    #plt.show()
    ACC=np.concatenate([np.expand_dims(AC,axis=1),np.expand_dims(VAC,axis=1)],axis=1)
    np.save('ACC',ACC)
    # Save model
    #tf.add_to_collection("optimizer", train_step)
    saver = tf.train.Saver()
    save_path = saver.save(sess, "tmp/"+model_name)
    print("Model saved in path: %s" % save_path)

