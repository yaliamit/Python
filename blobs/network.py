import tensorflow as tf
import numpy as np
import time
import sys


def adapt_batch(PARS, batch, i):
    glf = PARS['global_L2_fac']
    if ('cl_range' in PARS):
        ii = np.where(np.argmax(batch[1], 1) > PARS['cl_range'])[0]
        batch[1][ii] = 0

    return batch, glf


def compute_accuracy(last_layer,y_,thresh_,PARS):
    if('blob' in PARS):
        with tf.variable_scope('helpers'):
            nbp = PARS['num_blob_pars'] - 1
            ya = y_[:, :, :, nbp]
            accuracy = []
            hy = tf.cast(tf.greater(last_layer[:, :, :, nbp], thresh_),dtype=tf.float32)

            acn=tf.reduce_sum(tf.abs(hy - ya) * ya) \
               / tf.reduce_sum(ya)
            acp = tf.reduce_sum(tf.abs(hy - ya) * (1-ya)) \
                  / tf.reduce_sum(1-ya)
            accuracy.append(tf.identity(acn,name="ACCN"))
            accuracy.append(tf.identity(acp,name="ACCP"))
            temp = tf.zeros_like(y_[:, :, :, 0])
            yas=tf.reduce_sum(ya,axis=[1,2])
            iyas=tf.where(tf.greater(yas,0))
            yas=tf.squeeze(tf.gather(yas,iyas,axis=0))
            for j in range(nbp):
                temp = temp + (y_[:, :, :, j] - last_layer[:, :, :, j]) * (y_[:, :, :, j] - last_layer[:, :, :, j]) * ya
            temps=tf.squeeze(tf.gather(temp,iyas,axis=0))
            ac=tf.reduce_mean(tf.reduce_sum(tf.sqrt(temps),axis=[1,2])/yas)
            accuracy.append(tf.identity(ac,name="DIST"))
    else:
        with tf.variable_scope('helpers'):
            ui = tf.greater(tf.reduce_sum(y_, 1), 0)
            vj = tf.greater(tf.reduce_sum(y_, 0), 0)
            num_labels = tf.reduce_sum(tf.cast(ui, dtype=tf.float32))
            correct_prediction = tf.equal(tf.argmax(tf.boolean_mask(tf.boolean_mask(last_layer,ui),vj,axis=1), 1), tf.argmax(tf.boolean_mask(tf.boolean_mask(y_,ui),vj,axis=1), 1))
            accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32), name="ACC")/num_labels

    return accuracy

def compute_L2_loss(y_,PLH,l,parent):
    ui = tf.greater_equal(tf.reduce_sum(y_, 1), 0)
    #num_no_labels = tf.reduce_sum(tf.cast(ui, dtype=tf.float32))
    #shp=y_.shape.as_list()
    #num=shp[0]
    p1=tf.boolean_mask(parent[1, :, :, :, :],ui)
    p0=tf.boolean_mask(parent[0, :, :, :, :],ui)
    L2_loss=PLH['global_L2_fac_'] * l['fac'] * tf.reduce_mean((p1-p0)*(p1-p0))

    return L2_loss


def compute_loss(last_layer, y_, L2loss, PARS):

    loss_ce=0
    if (PARS['hinge']):
        yb = tf.cast(y_, dtype=tf.bool)
        cor = tf.boolean_mask(last_layer, yb)
        cor = tf.nn.relu(1. - cor)
        res = tf.boolean_mask(last_layer, tf.logical_not(yb))
        shp = last_layer.shape.as_list()
        shp[1] = shp[1] - 1
        res = tf.reshape(res, shape=shp)
        res = tf.reduce_sum(tf.nn.relu(1. + res), axis=1)
        loss = L2loss + tf.reduce_mean(cor + PARS['off_class_fac'] * res / (PARS['n_classes'] - 1), name="hinge")
    elif ('blob' in PARS):
        nbp = PARS['num_blob_pars'] - 1
        ya = y_[:, :, :, nbp]
        ly = last_layer[:, :, :, nbp]
        ls1 = tf.reduce_sum(-ya * ly +
                            tf.math.softplus(ly), axis=[1, 2])
        loss = tf.reduce_mean(ls1)
        temp = tf.zeros_like(y_[:, :, :, 0])
        for j in range(nbp):
            temp = temp + (y_[:, :, :, j] - last_layer[:, :, :, j]) * (y_[:, :, :, j] - last_layer[:, :, :, j]) * ya
        loss1 = tf.reduce_mean(tf.reduce_sum(temp, axis=[1, 2]))
        loss = loss + loss1 + L2loss
        loss = tf.identity(loss, name="LOSS")
    else:
        # Softmax-logistic loss
        ui = tf.greater(tf.reduce_sum(y_, 1), 0)
        vj = tf.greater(tf.reduce_sum(y_, 0), 0)
        #num_labels = tf.reduce_sum(tf.cast(ui, dtype=tf.float32))
        #num = y_.shape.as_list(0)
        loss_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.boolean_mask(tf.boolean_mask(y_, ui),vj,axis=1),
                                                                        logits=tf.boolean_mask(tf.boolean_mask(last_layer,
                                                                                               ui),vj,axis=1)))

        loss = loss_ce + L2loss
        loss = tf.identity(loss, name="LOSS")
    return loss, loss_ce


def find_joint_parent(l, parent, PARS):
    for ly in PARS['layers']:
        if ('parent' in ly):
            q = ly['parent']
            if (ly is not l and type(q) == str and q in parent):
                return q
    return None

def get_trainable(PARS):
        tvars = tf.trainable_variables()
        g_vars = tvars
        if ('non_trainable' in PARS):
            g_vars = []
            for var in tvars:
                inn = True
                for nt in PARS['non_trainable']:
                    if nt in var.name:
                        inn = False
                        break
                if inn:
                    g_vars.append(var)
        return g_vars

def get_name(ts):
    if type(ts) == list:
        name = ts[0].name
        T=ts[0]
    else:
        name = ts.name
        T=ts
    return (name,T)

def conv_layer(input,filter_size=[3,3],num_features=[1]):

    # Get number of input features from input and add to shape of new layer
    shape=filter_size+[input.get_shape().as_list()[-1],num_features]
    W = tf.get_variable('W',shape=shape) # Default initialization is Glorot (the one explained in the slides)
    b = tf.get_variable('b',shape=[num_features],initializer=tf.zeros_initializer)
    conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')+b
    return conv

def fully_connected_layer(input,num_features,type=None):
    # Make sure input is flattened.
    flat_dim=np.int32(np.array(input.get_shape().as_list())[1:].prod())
    input_flattened = tf.reshape(input, shape=[-1,flat_dim])
    shape=[flat_dim,num_features]
    if (type=='normal'):
        W=np.float32(np.random.normal(0,3./(np.sqrt(flat_dim+num_features)),shape))
        W_fc = tf.get_variable('W',initializer=W)
    else:
        W_fc = tf.get_variable('W',shape=shape)
    b_fc = tf.get_variable('b',shape=[num_features],initializer=tf.zeros_initializer)
    fc = tf.matmul(input_flattened, W_fc) + b_fc
    return(fc)


def get_parent(l, TS, x_):
    shp = x_.get_shape().as_list()
    if ('input' in l['parent']):
        if (len(shp)==4):
            parent=tf.expand_dims(x_,axis=0)
        else:
            parent=x_
    else:
        # Get list of parents
        if (type(l['parent']) == list):
            parent = []
            for s in l['parent']:
                for ts in TS:
                    name, T = get_name(ts)
                    if s in name:
                        parent.append(T)
        # Get single parent
        else:
            for ts in TS:
                name, T = get_name(ts)
                if l['parent'] in name:
                    parent=T
    return(parent)

def get_numunits(l,PARS,shp):
        if ('final' in l):
            num_units = PARS['n_classes']
        elif ('pre' in l):
            num_units=np.int32(shp[-3]/PARS['coarse_disp'])*\
                      np.int32(shp[-2]/PARS['coarse_disp'])*\
                      PARS['num_blob_pars']
        else:
            num_units=l['num_units']
        return(num_units)



def recreate_network(PARS, PLH):
    TS = []
    joint_parent = {}
    x_ = PLH['x_']; y_ = PLH['y_'];

    shp = x_.get_shape().as_list()
    shp_y=y_.get_shape().as_list()



    L2loss=tf.constant(0.,dtype=tf.float32)
    for i, l in enumerate(PARS['layers']):
        parent = None
        if ('parent' in l):
            parent=get_parent(l,TS, x_)
            npa=np.array(parent.get_shape().as_list())[0]
        scope_name = l['name']
        if ('non_linearity' in l):
            scope_name = l['name'] + '_' + l['non_linearity']
        out=[]
        with tf.variable_scope(scope_name):
            if ('conv' in l['name']):
                for i in range(npa):
                  with tf.variable_scope(str(i)):
                    out.append(conv_layer(parent[i,:,:,:,:], filter_size=list(l['filter_size']),
                                         num_features=l['num_filters']))
            # Dense layer
            elif ('dens' in l['name']):
                num_units=get_numunits(l,PARS,shp)
                for i in range(npa):
                  with tf.variable_scope(str(i)):
                    out.append(fully_connected_layer(parent[i,], num_features=num_units,type='normal'))
            # Pooling layer
            elif ('pool' in l['name']):
                for i in range(npa):
                  with tf.variable_scope(str(i)):
                    out.append(tf.nn.max_pool(parent[i,:,:,:,:], ksize=[1,l['pool_size'][0],l['pool_size'][1],1], strides=[1,l['stride'][0],
                                                        l['stride'][1],1], padding='SAME'))
            # Drop layer
            elif ('drop' in l['name']):
                for i in range(npa):
                  with tf.variable_scope(str(i)):
                    out.append(tf.layers.dropout(parent[i], rate=l['drop'], training=PLH['training_']))
            # Sum two layers - used for resent
            elif ('concatsum' in l['name']):
                for i in range(npa):
                  with tf.variable_scope(str(i)):
                    out.append(tf.add(parent[0][i,:,:,:,:], parent[1][i,:,:,:,:]))
                # This is a sum layer hold its joint_parent with another other layer
                j_parent = find_joint_parent(l, l['parent'], PARS)
                if (j_parent is not None):
                    name, T = get_name(TS[-1])
                    joint_parent[name] = j_parent
            elif ('comp' in l['name']):
                if (npa==2):
                    L2loss=compute_L2_loss(y_,PLH,l,parent)
                out=tf.expand_dims(parent[PLH['index_'],:,:,:,:],axis=0)
            elif ('reshape' in l['name']):
                if ('final' in l):
                        shape=[-1,np.int32(shp[-3]/PARS['coarse_disp']),np.int32(shp[-2]/PARS['coarse_disp'])
                                                           , PARS['num_blob_pars']]
                        for i in range(npa):
                            out.append(tf.reshape(parent[i,],shape))
                else:
                        for i in range(npa):
                            out.append(tf.reshape(parent[i,:,:,:,:], [-1,]+list(l['new_shape'])))
            if ('non_linearity' in l):
                if (l['non_linearity'] == 'relu'):
                  for ou in out:
                    ou = tf.nn.relu(ou)
                elif (l['non_linearity'] == 'tanh'):
                  for ou in out:
                    ou = tf.clip_by_value(PARS['nonlin_scale'] * ou, -1., 1.)

            if (type(out) == list):
                    out=tf.stack(out)

        if ('input' not in l['name']):
            TS.append(out)
    last_layer = TS[-1]
    if ('blob' in PARS):
        if len(last_layer.get_shape().as_list())> len(shp_y):
            last_layer=last_layer[0,]
    else:
        if len(last_layer.get_shape().as_list())> len(shp_y):
            last_layer = last_layer[0,]
    with tf.variable_scope('loss'):
        loss, loss_ce = compute_loss(last_layer,y_,L2loss,PARS)

    # Accuracy computation
    last_layer = tf.identity(last_layer,name="LAST")
    accuracy=compute_accuracy(last_layer,y_,PLH['thresh_'],PARS)


    print('joint_parent', joint_parent)
    # joint_parent contains information on layers that are parents to two other layers which affects the gradient propagation.
    PARS['joint_parent'] = joint_parent
    for t in TS:
        print(t)
    return loss, accuracy, last_layer



def run_epoch(train, PLH, OPS, PARS, sess, i, type='Training'):

        mode = 'Class'
        if ('blob' in PARS):
            mode = 'blob'
        # The version of the data that gets classified after the two laters are compared
        ind = 0 # The clean version
        if ('corr' in PARS and PARS['corr'] and type != 'Training'):
            ind = 1 # The noisty version that wasn't used for training the classifier.
        t1 = time.time()
        # Randomly shuffle the training data


        if (len(train[0].shape)==5):
            ii = np.arange(0, train[0].shape[1], 1)
            if (type == 'Training'):
                np.random.shuffle(ii)
            tr = train[0][:, ii, :, :, :]
        else:
            ii = np.arange(0, train[0].shape[0], 1)
            if (type == 'Training'):
                np.random.shuffle(ii)
            tr = train[0][ii, :, :, :]
        y = train[1][ii, :]
        cso = 0; acco = 0; accon = 0; accop = 0; disto = 0; ca = 0; l2cso=0
        HY = []
        thresh = 0.
        if (type == 'Test' and mode == 'blob'):
            thresh = PARS['thresh']
        # Run disjoint batches on shuffled data
        for j in np.arange(0, len(y), PARS['batch_size']):
            if (len(train[0].shape) == 5):
                batch = (tr[:, j:j + PARS['batch_size'], :, :, :], np.copy(y[j:j + PARS['batch_size']]))
            else:
                batch = (tr[j:j + PARS['batch_size'], :, :, :], np.copy(y[j:j + PARS['batch_size']]))

            if (mode == 'blob'):
                if (type == 'Training'):
                    csi, acc, _ = sess.run([OPS['cs'], OPS['accuracy'], OPS['train_step']],
                                           feed_dict={PLH['x_']: batch[0], PLH['y_']: batch[1], PLH['lr_']: PARS['step_size']
                                                       ,PLH['thresh_']: thresh}) #,PLH['training_']: True})
                else:

                    csi, acc, ts = sess.run([OPS['cs'], OPS['accuracy'], OPS['TS']],
                                            feed_dict={PLH['x_']: batch[0], PLH['y_']: batch[1], PLH['lr_']: PARS['step_size']
                                                       ,PLH['thresh_']: thresh}) #,PLH['training_']: False
                    if (type == 'Test'):
                        HY.append(ts)
                accon += acc[0]
                accop += acc[1]
                disto += acc[2]

                cso += csi

            elif (mode == 'Class'):

                batch, glf = adapt_batch(PARS, batch, i)
                if (type == 'Training'):
                    csi, acc, ts, _ = sess.run([OPS['cs'], OPS['accuracy'], OPS['TS'], OPS['train_step']],
                                               feed_dict={PLH['x_']: batch[0], PLH['y_']: batch[1],
                                                          PLH['lr_']: PARS['step_size'],
                                                          PLH['training_']: True, PLH['index_']: ind, PLH['global_L2_fac_']:glf})
                else:
                    csi, acc, ts = sess.run([OPS['cs'], OPS['accuracy'], OPS['TS']],
                                            feed_dict={PLH['x_']: batch[0], PLH['y_']: batch[1], PLH['lr_']: PARS['step_size'],
                                                       PLH['training_']: False, PLH['index_']: ind, PLH['global_L2_fac_']:glf})
                acco += acc
                cso += csi
                l2cso += ts[0]
            ca += 1

        print('Epoch time', time.time() - t1)
        print("Final results: epoch", str(i))
        if (mode == 'blob'):
            print(type + " dist:\t\t\t{:.6f}".format(disto / ca))
            print(type + " accn:\t\t\t{:.6f}".format(accon / ca))
            print(type + " accp:\t\t\t{:.6f}".format(accop / ca))
            print(type + " loss:\t\t\t{:.6f}".format(cso / ca))
        else:
            print(type + " accn:\t\t\t{:.6f}".format(acco / ca))
            print(type + " loss:\t\t\t{:.6f}".format(cso / ca))
            print(type + " L2Loss:\t\t\t{:.6f}".format(l2cso/ca))
        sys.stdout.flush()
        return (HY)