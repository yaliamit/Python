import tensorflow as tf
import numpy as np
from keras import backend as K
from Conv_layers import conv_layer, grad_conv_layer, fully_connected_layer, grad_fully_connected, grad_pool, grad_pool_disjoint_fast
from Conv_layers import sparse_fully_connected_layer, grad_sparse_fully_connected, MaxPoolingandMask, MaxPoolingandMask_disjoint_fast, real_drop

def find_ts(name,TS):
    for ts in TS:
        if (type(ts) is list):
            if (name in ts[0].name):
                return(ts)
        elif (name in ts.name):
            return(ts)

def find_wr(name,VS):
    W=None
    R=None
    for vs in VS:
      if ('sparse' not in vs.name):
        if (name in vs.name):
            if ('W' in vs.name):
                W=vs
            elif('R' in vs.name):
                R=vs
    return W,R

def re_initialize(shape):
    std=np.sqrt(6./((shape[0]+shape[1])*shape[2]*shape[3]))
    Wout=np.float32(np.random.uniform(-std,std,shape))
    Rout=np.float32(np.random.uniform(-std,std,shape))

    return Wout, Rout

def re_initialize_dense(shape):
    std=np.sqrt(6./(shape[0]+shape[1]))
    Wout=np.float32(np.random.uniform(-std,std,shape))
    Rout=np.float32(np.random.uniform(-std,std,shape))

    return Wout, Rout

# Create dictionary of future sparse layer parameters with name given by layer name
def get_parameters_s(VSIN,SP,TS, re_randomize=None):

    WRS={}
    sparse_shape = {}
    for sp in SP:
            Win,Rin=find_wr(sp,VSIN)
            wrs=[Win.eval(),Rin.eval()]
            if (re_randomize is not None):
              if sp in re_randomize:
                 Win, Rin = re_initialize(Win.shape.as_list())
                 wrs=[Win,Rin]
            WRS[sp]=wrs
            sparse_shape[sp] = find_ts(sp, TS).get_shape().as_list()[1:3]
    return(WRS,sparse_shape)

def get_parameters(VSIN,PARS, re_randomize=None):

    WR={}
    for i,l in enumerate(PARS['layers']):

        if ('conv' in l['name'] or 'dens' in l['name']):
            Win,Rin=find_wr(l['name'],VSIN)
            if (Win is not None):
                wrs = [Win.eval(), Rin.eval()]
                if re_randomize is not None and l['name'] in re_randomize:
                    if ('conv' in l['name']):
                        Win,Rin = re_initialize(Win.shape.as_list())
                    else:
                        Win, Rin = re_initialize_dense(Win.shape.as_list())
                    wrs = [Win, Rin]
                WR[l['name']]=wrs

    return(WR)






def find_joint_parent(l,parent,PARS):
      
        for ly in PARS['layers']:
            if ('parent' in ly):
                q=ly['parent']
                if (ly is not l and type(q)==str and q in parent):
                    return q
        return None

# If ts is a tensor just return name and tensor
# is ts is a list only first entry is a tensor, the rest are paremters, return name and tensor.
def get_name(ts):
    if type(ts) == list:
        name = ts[0].name
        T=ts[0]
    else:
        name = ts.name
        T=ts
    return (name,T)


def recreate_network(PARS,x,y_,Train,WR=None,SP=None):


            TS=[]
            ln=len(PARS['layers'])
            joint_parent={}
            for i,l in enumerate(PARS['layers']):
                parent=None
                prob=[1.,-1.]
                if ('force_global_prob' in PARS):
                    prob=list(PARS['force_global_prob'])
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
                                    if s in name and not 'Equal' in name:
                                        parent.append(T)
                        # Get single parent
                        else:
                            for ts in TS:
                                name, T = get_name(ts)
                                if l['parent'] in name and not 'Equal' in name:
                                    parent=T
                # First check if this is one of the new sparse layers - create it with existing parameter values.
                if (SP is not None and l['name'] in PARS['sparse']):
                    Win=SP[l['name']][0]
                    Rin=SP[l['name']][1]
                    if (Rin is None):
                        F=Win
                    else:
                        F=Rin
                    Finds=F.indices.eval()
                    Fvals=F.values.eval()
                    Fdims=F.dense_shape.eval()
                    Finds=Finds[:,[1,0]]
                    Fdims=Fdims[[1,0]]
                    Fin=tf.SparseTensor(indices=Finds,values=Fvals,dense_shape=Fdims)
                    Fin=tf.sparse_reorder(Fin)
                    scope_name = 'sparse'+l['name']
                    scale = 0
                    # with non-linearity - always clipped linearity
                    if('non_linearity' in l and l['non_linearity'] == 'tanh'):
                        scale = PARS['nonlin_scale']
                        scope_name = 'sparse'+l['name']+ 'nonlin'
                    with tf.variable_scope(scope_name):
                        num_units=(Win.dense_shape[0]).eval()
                        TS.append(sparse_fully_connected_layer(parent,PARS['batch_size'],PARS['nonlin_scale'], num_units=num_units, num_features=l['num_filters'], prob=prob,scale=scale, Win=Win,Rin=Rin, Fin=Fin))
                # Otherwise create regular layer either from scratch or with existing parameters.
                else:
                    if ('conv' in l['name']):
                        Win=None
                        Rin=None
                        if (WR is not None):
                            Win=WR[l['name']][0]
                            Rin=WR[l['name']][1]
                        scope_name=l['name']
                        scale=0
                        # with non-linearity - always clipped linearity
                        if ('non_linearity' in l and l['non_linearity']=='tanh'):
                            scale=PARS['nonlin_scale']
                            scope_name=l['name']+'nonlin'
                        with tf.variable_scope(scope_name):
                            TS.append(conv_layer(parent,PARS['batch_size'],PARS['nonlin_scale'], filter_size=list(l['filter_size']),num_features=l['num_filters'], prob=prob, scale=scale,Win=Win,Rin=Rin))
                    # Dense layer
                    elif ('dens' in l['name']):
                        Win = None
                        Rin = None
                        if (WR is not None):
                            Win=WR[l['name']][0]
                            Rin=WR[l['name']][1]
                        scope_name = l['name']
                        scale = 0
                        # with non-linearity - always clipped linearity
                        if('non_linearity' in l and l['non_linearity'] == 'tanh'):
                            scale = PARS['nonlin_scale']
                            scope_name = l['name'] + 'nonlin'
                        with tf.variable_scope(scope_name):
                            num_units=l['num_units']
                            # Make sure final layer has num_units=num_classes
                            if ('final' in l):
                                num_units=PARS['n_classes']
                            TS.append(fully_connected_layer(parent,PARS['batch_size'],PARS['nonlin_scale'], num_features=num_units,prob=prob,scale=scale, Win=Win,Rin=Rin))
                    # Pooling layer
                    elif ('pool' in l['name']):
                        with tf.variable_scope(l['name']):
                            # Quick computation pooling on disjoint regions
                            if (l['pool_size']==l['stride']):
                                pool, mask = MaxPoolingandMask_disjoint_fast(parent, [1]+list(l['pool_size'])+[1],strides=[1]+list(l['stride'])+[1])
                                TS.append([pool,l['pool_size'],l['stride']])
                            # More complex computation using shifts of arrays for stride < pool_size
                            else:
                                pool, mask = MaxPoolingandMask(parent, l['pool_size'][0],l['stride'][0])
                                TS.append([pool,l['pool_size'][0],l['stride'][0]])
                            # Keep record of mask for gradient computation
                            TS.append(mask)
                    # Drop layer
                    elif ('drop' in l['name']):
                        with tf.variable_scope(l['name']):
                            ffac = 1. / (1. - l['drop'])
                            # Only drop is place holder Train is True
                            drop=tf.cond(Train,lambda: real_drop(parent,l['drop'],PARS['batch_size']),lambda: parent)
                            TS.append([drop,ffac])
                    # Add two equal sized consecutive layers
                    elif ('concatsum' in l['name']):
                        with tf.variable_scope(l['name']):
                            res_sum=tf.add(parent[0],parent[1])
                            TS.append(res_sum)
                            # This is a sum layer hold its joint_parent with another other layer
                            j_parent=find_joint_parent(l,l['parent'],PARS)
                            if (j_parent is not None):
                                name,T=get_name(TS[-1])
                                joint_parent[name]=j_parent

            with tf.variable_scope('loss'):
               # Hinge loss
               if (PARS['hinge']):
                 yb=tf.cast(y_,dtype=tf.bool)
                 cor=tf.boolean_mask(TS[-1],yb)
                 cor = tf.nn.relu(1.-cor)
                 res=tf.boolean_mask(TS[-1],tf.logical_not(yb))
                 shp=TS[-1].shape.as_list()
                 shp[1]=shp[1]-1
                 res=tf.reshape(res,shape=shp)
                 res=tf.reduce_sum(tf.nn.relu(1.+res),axis=1)
                 loss=tf.reduce_mean(cor+PARS['off_class_fac']*res/(PARS['n_classes']-1),name="hinge")
               else:
                 # Softmax-logistic loss
                 loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=TS[-1]),name="sm")


            # Accuracy computation
            with tf.variable_scope('helpers'):
                correct_prediction = tf.equal(tf.argmax(TS[-1], 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="ACC")
            print('joint_parent',joint_parent)
            # joint_parent contains information on layers that are parents to two other layers which affects the gradient propagation.
            PARS['joint_parent'] = joint_parent
            TS.reverse()
            for t in TS:
                print(t)
            return loss, accuracy, TS


def update_only_non_zero(V,gra, step):
    up=V-step*gra
    up=K.tf.where(tf.equal(V,tf.constant(0.)),V,up)
    assign_op = tf.assign(V,up)
    return assign_op

def back_prop(loss,acc,TS,VS,x,PARS, non_trainable=None):
    # Get gradient of loss with respect to final output layer using tf gradient
    # The rest will be explicit backprop
    
    gradX=tf.gradients(loss,TS[0])

    gradx=gradX[0]
    lts=len(TS)
    vs=0
    ts=0
    OPLIST=[]
    grad_hold_var={}
    joint_parent=None
    all_grad=[]
    if (PARS['debug']):
        all_grad.append(gradx)
    for ts in range(lts):
        T=TS[ts]
        name,T=get_name(T)
        if (ts<lts-1):
                prename,pre=get_name(TS[ts+1])
                if ('Equal' in prename):
                    prename,pre=get_name(TS[ts+2])
        else:
            pre=x
        # You have held a gradx from a higher up layer to be added to current one.
        if (joint_parent is not None):
            pp=name.split('/')[0]
            ind=pp.find('nonlin')
            pp=pp[:ind]
            if (joint_parent == pp):
                print(joint_parent,'grad_hold',grad_hold_var[joint_parent])
                gradx=tf.add(gradx,grad_hold_var[joint_parent])
                joint_parent=None
        if ('conv' in name and not 'sparse' in name):
            scale=0
            shortname=name
            if ('nonlin' in name):
                shortname=name[0:name.find('nonlin')]
                scale=PARS['nonlin_scale']
            gradconvW, gradx = grad_conv_layer(PARS['batch_size'],below=pre,back_propped=gradx,current=T,W=VS[vs], R=VS[vs+1],scale=scale)
            if (non_trainable is None or (non_trainable is not None and shortname not in non_trainable)):
                assign_op_convW = update_only_non_zero(VS[vs],gradconvW,PARS['step_size'])
                OPLIST.append(assign_op_convW)
            # If an R variable exists and is a 4-dim array i.e. is active
            if (len(VS[vs+1].shape.as_list())==4):
             if (non_trainable is None or (non_trainable is not None and shortname not in non_trainable)):
                assign_op_convR=update_only_non_zero(VS[vs+1],gradconvW, PARS['Rstep_size'])
                OPLIST.append(assign_op_convR)
            if (PARS['debug']):
                all_grad.append(gradx)
            ts+=1
            vs+=2
        elif ('drop' in name):
            Z = tf.equal(T, tf.constant(0.))
            gradx=K.tf.where(Z,T,tf.multiply(tf.reshape(gradx,T.shape),TS[ts][1]))
            if (PARS['debug']):
                all_grad.append(gradx)
        elif ('Equal' in name):
            mask=T
            ts+=1
        elif ('Max' in name):
            if (TS[ts][1]==TS[ts][2]):
                gradx=grad_pool_disjoint_fast(gradx,T,mask,pre,TS[ts][1])
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
            assign_op_fcW = update_only_non_zero(VS[vs],gradfcW,PARS['step_size'])
            OPLIST.append(assign_op_fcW)
            # If an R variable exists and is a 2-dim matrix i.e. is active
            if (len(VS[vs+1].shape.as_list())==2):
                assign_op_fcR = update_only_non_zero(VS[vs+1],gradfcW,PARS['Rstep_size'])
                OPLIST.append(assign_op_fcR)
            if (PARS['debug']):
                all_grad.append(gradx)
            ts+=1
            vs+=2
        elif ('sparse' in name):
            scale = 0
            doR=(VS[vs+ 8].get_shape().as_list()[0] == 2)
            if ('nonlin' in name):
                scale = PARS['nonlin_scale']
            if (PARS['force_global_prob'][0]==1. or not doR):
                gradfcW, gradx, gradfcR = grad_sparse_fully_connected(below=pre,back_propped=gradx,current=T, F_inds=VS[vs], F_vals=VS[vs+1], F_dims=VS[vs+2], W_inds=VS[vs+3], R_inds=None,scale=scale)
            else:
                gradfcW, gradx, gradfcR = grad_sparse_fully_connected(below=pre,back_propped=gradx,current=T, F_inds=VS[vs], F_vals=VS[vs+1], F_dims=VS[vs+2], W_inds=VS[vs+3], R_inds=VS[vs+6],scale=scale)

            assign_op_fcW = update_only_non_zero(VS[vs+4],gradfcW,PARS['step_size'])
            OPLIST.append(assign_op_fcW)
            # If an R variable exists and is a 2-dim matrix i.e. is active
            if (doR):
                assign_op_fcR = update_only_non_zero(VS[vs+7],gradfcR,PARS['Rstep_size'])
                OPLIST.append(assign_op_fcR)
            if (PARS['debug']):
                all_grad.append(gradx)
            ts+=1
            vs+=9
        if (name in PARS['joint_parent']):
            grad_hold=gradx
            joint_parent=PARS['joint_parent'][name]
            grad_hold_var[joint_parent]=grad_hold
    if (PARS['debug']):
        print('all_grad',len(all_grad))
        for cg in all_grad:
            OPLIST.append(cg)
    #print('Length of VS',len(VS),'Length of OPLIST',len(OPLIST))
    OPLIST.append(acc)
    OPLIST.append(loss)
    
    return OPLIST, len(all_grad)

def zero_out_weights(PARS,VS,sess):

    for i, v in enumerate(VS):
        shape=v.get_shape().as_list()
        # After reversal, i=0 - first trainable variable is last dense layer W.
        #                 i=1 - second trainable variable is last dense layer R
        # Don't zero out these because with large numbers of classes the hinge loss doesn't work.
        if (i > 1):
             if (PARS['force_global_prob'][1] >= 0 and PARS['force_global_prob'][0] < 1.):
                if (not 'sparse' in v.name):
                    Z = tf.zeros(shape)
                    U = tf.random_uniform(shape)
                    zero_op = tf.assign(v, K.tf.where(tf.less(U, tf.constant(PARS['force_global_prob'][0])), v, Z))
                    sess.run(zero_op)
