import tensorflow as tf
import numpy as np
import time
import Conv_net_gpu

train, val, test = Conv_net_gpu.get_data(data_set='cifar10')

tro=np.tile(train[0],(1,1,1,8))
print(tro.shape)

tf.reset_default_graph()
input=tf.convert_to_tensor(tro[0:500])

pool_size=2
stride=2
inputp=Conv_net_gpu.MaxPoolingandMask(input,pool_size,stride)
#inputp=Conv_net_gpu.MaxPoolingandMask_old(input,[1,pool_size,pool_size,1],[1,stride,stride,1])

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    t1=time.time()
    pooled,mask, SI=sess.run(inputp)
    t2=time.time()
    print('time1',t2-t1)
    pooled,mask, SI=sess.run(inputp)
    t3=time.time()
    print('time2',t3-t2)
    pooled, mask, SI = sess.run(inputp)
    t4 = time.time()
    print('time3', t4 - t3)
    pooled, mask, SI = sess.run(inputp)
    t5 = time.time()
    print('time4', t5 - t4)
print('done')



