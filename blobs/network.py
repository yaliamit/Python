import tensorflow as tf
import numpy as np

def find_joint_parent(l, parent, PARS):
    for ly in PARS['layers']:
        if ('parent' in ly):
            q = ly['parent']
            if (ly is not l and type(q) == str and q in parent):
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

def conv_layer(input,filter_size=[3,3],num_features=[1]):

    # Get number of input features from input and add to shape of new layer
    shape=filter_size+[input.get_shape().as_list()[-1],num_features]
    W = tf.get_variable('W',shape=shape) # Default initialization is Glorot (the one explained in the slides)
    b = tf.get_variable('b',shape=[num_features],initializer=tf.zeros_initializer)
    conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')+b
    return conv

def fully_connected_layer(input,num_features):
    # Make sure input is flattened.
    flat_dim=np.int32(np.array(input.get_shape().as_list())[1:].prod())
    input_flattened = tf.reshape(input, shape=[-1,flat_dim])
    shape=[flat_dim,num_features]
    W_fc = tf.get_variable('W',shape=shape)
    b_fc = tf.get_variable('b',shape=[num_features],initializer=tf.zeros_initializer)
    fc = tf.matmul(input_flattened, W_fc) + b_fc
    return(fc)

def recreate_network(PARS, x_, y_, training_):
    TS = []
    joint_parent = {}
    for i, l in enumerate(PARS['layers']):
        parent = None
        if ('parent' in l):
            if ('input' in l['parent']):
                parent = x_
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
                            parent = T
        if ('conv' in l['name']):
            scope_name = l['name']
            if ('non_linearity' in l):
                scope_name = l['name'] + '_' + l['non_linearity']
            with tf.variable_scope(scope_name):
                if('non_linearity' in l):
                    out = conv_layer(parent, filter_size=list(l['filter_size']),
                                     num_features=l['num_filters'])
                    if (l['non_linearity'] == 'relu'):
                        out = tf.nn.relu(out)
                    elif (l['non_linearity'] == 'tanh'):
                        out = tf.clip_by_value(PARS['nonlin_scale'] * out, -1., 1.)
            TS.append(out)
            # Dense layer
        elif ('dens' in l['name']):
                scope_name = l['name']
                if ('final' in l):
                    num_units = PARS['n_classes']
                else:
                    num_units=l['num_units']

                if ('non_linearity' in l):
                    scope_name=l['name']+'_'+l['non_linearity']
                with tf.variable_scope(scope_name):
                     #flattened=tf.contrib.layers.flatten(parent)
                     #flattened=flattened[:,:,tf.newaxis]
                     #out=tf.contrib.layers.fully_connected(parent,num_outputs=num_units,activation_fn=None,trainable=True)
                     out = fully_connected_layer(parent, num_features=num_units)
                     if ('non_linearity' in l):
                        if(l['non_linearity'] == 'relu'):
                            out=tf.nn.relu(out)
                        elif(l['non_linearity'] == 'tanh'):
                            out= tf.clip_by_value(PARS['nonlin_scale'] * out, -1., 1.)
                TS.append(out)

            # Pooling layer
        elif ('pool' in l['name']):
                with tf.variable_scope(l['name']):
                    out=tf.nn.max_pool(parent, ksize=[1,l['pool_size'][0],l['pool_size'][1],1], strides=[1,l['stride'][0],l['stride'][1],1], padding='SAME')
                    TS.append(out)
            # Drop layer
        elif ('drop' in l['name']):
                with tf.variable_scope(l['name']):
                    out=tf.layers.dropout(input, rate=l['drop'], training=training_)
                    TS.append(out)
        elif ('concatsum' in l['name']):
                with tf.variable_scope(l['name']):
                    out = tf.add(parent[0], parent[1])
                    TS.append(out)
                    # This is a sum layer hold its joint_parent with another other layer
                    j_parent = find_joint_parent(l, l['parent'], PARS)
                    if (j_parent is not None):
                        name, T = get_name(TS[-1])
                        joint_parent[name] = j_parent
        elif ('reshape' in l['name']):
                with tf.variable_scope(l['name']):
                    out = tf.reshape(parent, [-1,]+list(l['new_shape']))
                    TS.append(out)
    last_layer = TS[-1]
    with tf.variable_scope('loss'):
        # Hinge loss
        if (PARS['hinge']):
            yb = tf.cast(y_, dtype=tf.bool)
            cor = tf.boolean_mask(TS[-1], yb)
            cor = tf.nn.relu(1. - cor)
            res = tf.boolean_mask(TS[-1], tf.logical_not(yb))
            shp = TS[-1].shape.as_list()
            shp[1] = shp[1] - 1
            res = tf.reshape(res, shape=shp)
            res = tf.reduce_sum(tf.nn.relu(1. + res), axis=1)
            loss = tf.reduce_mean(cor + PARS['off_class_fac'] * res / (PARS['n_classes'] - 1), name="hinge")
        elif ('blob' in PARS):

            ya=y_[:,:,:,2]
            ly=last_layer[:,:,:,2]
            ls1=tf.reduce_sum(-ya * ly +
                          tf.math.softplus(ly), axis=[1, 2])
            loss = tf.reduce_mean(ls1)

            loss = loss + tf.reduce_mean(
               tf.reduce_sum((y_[:, :, :, 0] - last_layer[:, :, :, 0]) *
                             (y_[:, :, :, 0] - last_layer[:, :, :, 0]) * y_[:, :, :, 2]
                             + (y_[:, :, :, 1] - last_layer[:, :, :, 1]) *
                             (y_[:, :, :, 1] - last_layer[:, :, :, 1]) * y_[:, :, :, 2],
                             axis=[1, 2]))
            loss=tf.identity(loss,name="LOSS")
        else:
            # Softmax-logistic loss
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=TS[-1]), name="LOSS")

    # Accuracy computation
    last_layer = tf.identity(TS[-1],name="LAST")
    if (PARS['hinge']):
        with tf.variable_scope('helpers'):
            correct_prediction = tf.equal(tf.argmax(last_layer, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="ACC")
    elif(PARS['blob']):
        with tf.variable_scope('helpers'):
            accuracy = []
            hy = tf.cast(tf.greater(last_layer[:, :, :, 2], 0),dtype=tf.float32)
            ac=tf.reduce_sum(tf.abs(hy - y_[:, :, :, 2]) * y_[:, :, :,2]) \
               / tf.reduce_sum(y_[:, :, :, 2])
            accuracy.append(tf.identity(ac,name="ACC"))
            ac=tf.reduce_sum((tf.abs(last_layer[:, :, :, 0] - y_[:, :, :, 0]) +
                                           np.abs(last_layer[:, :, :, 1] - y_[:, :, :, 1]))
                                          * y_[:, :, :, 2]) /tf.reduce_sum(y_[:,:,:,2])
            accuracy.append(tf.identity(ac,name="DIST"))
    print('joint_parent', joint_parent)
    # joint_parent contains information on layers that are parents to two other layers which affects the gradient propagation.
    PARS['joint_parent'] = joint_parent
    for t in TS:
        print(t)
    return loss, accuracy, last_layer