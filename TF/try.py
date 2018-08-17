import tensorflow as tf
import numpy as np
import time
import Conv_net_gpu

#train, val, test = Conv_net_gpu.get_data(data_set='cifar10')

#tro=np.tile(train[0],(1,1,1,8))
tro=np.float32(np.random.rand(1,8,8,1))
fac=np.float32(np.array(range(16)).reshape(1,4,4,1))
print(tro.shape)

tf.reset_default_graph()
input=tf.convert_to_tensor(tro)
tfac=tf.convert_to_tensor(fac)
pool_size=3
stride=2
input_test=tf.layers.max_pooling2d(input,[pool_size,pool_size],[stride,stride],padding='SAME')
ltest=tf.reduce_sum(tf.multiply(input_test,tfac))
dpl=tf.gradients(ltest,input)
inputp=Conv_net_gpu.MaxPoolingandMask(input,pool_size,stride)
lp=tf.reduce_sum(tf.multiply(inputp[0],tfac))
dpg=tf.gradients(lp,inputp[0])
dpi=Conv_net_gpu.grad_pool(dpg, inputp[0], inputp[1], pool_size, stride)
#inputp=Conv_net_gpu.MaxPoolingandMask_old(input,[1,pool_size,pool_size,1],[1,stride,stride,1])

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    dpgo=sess.run(dpg)
    pooled, mask = sess.run(inputp)
    pooled_test=sess.run(input_test)
    gradi=sess.run(dpi)
    grad_test=sess.run(dpl)

    print('tro',tro[0,:,:,0])
    print('dpgo',dpgo[0][0,:,:,0])
    print('pooled',pooled[0,:,:,0])
    print('pooled_test',pooled_test[0,:,:,0])
    print('mask',mask[0,0,:,:,0])
    print('gradi',gradi[0][0,:,:,0])
    #print('grad_pool_shift',gradi[1][0,:,:,0])
    print('grad_test',grad_test[0][0,:,:,0])
print('done')



