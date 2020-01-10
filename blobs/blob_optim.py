import tensorflow as tf
import numpy as np

def make_blobs_tf(mux,muy,sigmas, Amps):

    bp=np.concatenate([np.array(mux), np.array(muy), np.array(sigmas), np.array(Amps)]).reshape((4,len(mux))).transpose()
    bp=np.float32(bp)

    return bp

def diff(g):
    dgx = g[1:, :] - g[:-1, :]
    dgy = g[:, 1:] - g[:, :-1]
    dg=tf.concat([dgx,tf.transpose(dgy)],axis=0)
    return dg

def diff_np(g):
    dgx = g[1:, :] - g[:-1, :]
    dgy = g[:, 1:] - g[:, :-1]
    dg=np.concatenate([dgx,np.transpose(dgy)],axis=0)
    return dg

def loss_blobs_tf(doimage,oimage,ll,blp,image_dim,lam, lamd):

    image_dim2=image_dim*image_dim
    with tf.variable_scope('blobp'):
        blobp = tf.get_variable('blobp',dtype=tf.float32,initializer=blp)
    with tf.variable_scope('lpis'):
        lpis=tf.get_variable('lpis',shape=[ll],dtype=tf.float32,initializer=tf.zeros_initializer)
    x, y = tf.meshgrid(tf.range(image_dim), tf.range(image_dim))

    x=tf.to_float(x)
    y=tf.to_float(y)
    llf=tf.to_float(ll)
    g = tf.zeros_like(x)
    pis=blobp[:,3]#pis=llf*tf.nn.softmax(lpis)
    for i in range(ll):
        dd=((x - blobp[i, 0]) * (x - blobp[i, 0]) + (y - blobp[i, 1]) * (y - blobp[i, 1]))
        gt =  tf.exp(-dd/ (2.0 * blobp[i,2] * blobp[i,2]))
        g = g+blobp[i,3]*gt
    dg=diff(g)
    ent=lam*tf.reduce_mean(blobp[:,3]*blobp[:,3])
    dloss=tf.reduce_mean((doimage-dg)*(doimage-dg),name='DLOSS')
    gloss=lamd*tf.reduce_mean((oimage-g)*(oimage-g),name='GLOSS')
    loss=dloss+gloss+ent

    return loss, gloss, dloss, pis, g, dg, ent

def optimize_blobs_tf(mux,muy,sigmas, Amps, image_dim, orig_image,PARS):

    step_size=PARS['optim_step_size']
    lam=PARS['optim_lam']
    lamd=PARS['lamd']
    tf.reset_default_graph()
    blobp= make_blobs_tf(mux, muy, sigmas, Amps)
    oimage=orig_image[0,:,:,0]
    doimage=diff_np(oimage)

    num_epochs=1000
    with tf.Session() as sess:
        loss, gloss, dloss, pis, g, dg, ent=loss_blobs_tf(doimage,oimage,len(mux),blobp,image_dim,lam,lamd)
        tvars = tf.trainable_variables()
        train_step = tf.train.AdamOptimizer(learning_rate=step_size).minimize(loss,var_list=tvars)
        sess.run(tf.global_variables_initializer())
        loss2, gloss2, dloss2, pis2, gg2, dgg2, ent2 = sess.run([loss, gloss, dloss, pis, g, dg, ent])
        #print(gloss2, dloss2,ent2,loss2)
        for i in range(num_epochs):
            loss2,gloss2,dloss2,pis2,gg2,dgg2, ent2,_=sess.run([loss, gloss, dloss, pis,g, dg, ent, train_step])
            #print(gloss2, dloss2, ent2, loss2)

        new_blobp=sess.run(tvars[0])

    mux_new=new_blobp[:,0]
    muy_new=new_blobp[:,1]
    sigmas_new=new_blobp[:,2]
    As_new=new_blobp[:,3]

    return mux_new,muy_new,sigmas_new,As_new