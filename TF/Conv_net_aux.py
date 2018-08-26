import tensorflow as tf
import time
import numpy as np
from keras import backend as K
import parse_net_pars as pp


def zero_out_weights(PARS,v,sess):
    if (PARS['force_global_prob'][1] >= 0 and PARS['force_global_prob'][0] < 1.):
        print('Zeroing out weights at rate ', PARS['force_global_prob'][0])
        shape = v.get_shape().as_list()
        Z = tf.zeros(shape)
        U = tf.random_uniform(shape)
        zero_op = tf.assign(v, K.tf.where(tf.less(U, tf.constant(PARS['force_global_prob'][0])), v, Z))
        sess.run(zero_op)

# Run the iterations of one epoch


def process_parameters(net):
    PARS = {}
    pp.parse_text_file(net, PARS, lname='layers', dump=True)
    PARS['step_size'] = PARS['eta_init']
    Rstep_size = list(PARS['force_global_prob'])[1] * PARS['step_size']
    print('Rstep_size', Rstep_size)
    PARS['Rstep_size'] = Rstep_size
    PARS['nonlin_scale'] = .5

    return PARS

def print_results(type,epoch,lo,ac):
    print("Final results: epoch", str(epoch))
    print(type+" loss:\t\t\t{:.6f}".format(lo))
    print(type+" acc:\t\t\t{:.6f}".format(ac))
