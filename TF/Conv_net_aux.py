#import parse_net_pars as pp
import subprocess as commands
import Conv_net_gpu
import Conv_net_aux
import tensorflow as tf
from Conv_data import get_data, rotate_dataset_rand
import numpy as np
import time

def process_param_line(line):
    # Split parameter line on :
    s = str.split(line, ':')
    s1 = str.strip(s[1], ' ;\n')
    # Analyze value
    try:
        # Float or int
        a = float(s1)
        if '.' not in s1:
            a = int(s1)
    # None, True, False, or list of names.
    except ValueError:
        if (s1 == 'None'):
            a = None
        elif (s1 == 'True'):
            a = True
        elif (s1 == 'False'):
            a = False
        else:
            if '(' in s1:
                aa = str.split(str.strip(s1, ' ()\n'), ',')
                a = []
                for aaa in aa:
                    a.append(float(aaa))
                a = tuple(a)
            else:
                s11 = s1.split(',')
                if (len(s11) == 1):
                    a = s1
                else:
                    a = []
                    for ss in s11:
                        try:
                            aa = int(ss)
                            a.append(aa)
                        except ValueError:
                            if (ss != ''):
                                a.append(ss)

    return (s[0], a)


def process_network_line(line, global_drop):
    # break line on the ; each segment is a parameter for the layer of that line
    sss = str.split(line, ';')
    lp = {}
    for ss in sss:
        # Split between parameter name and value
        s = str.split(ss, ':')
        s1 = str.strip(s[1], ' \n')
        # Process the parameter value
        # A nonlinearity function
        if ('lasagne' in s1):
            # if ('rectify' in s1):
            #     lp['non_linearity']=lasagne.nonlinearities.rectify
            # elif ('rect_sym' in s1):
            #     lp['non_linearity']=make_net.rect_sym
            # elif ('sigmoid' in s1):
            #     lp['non_linearity']=lasagne.nonlinearities.sigmoid
            # elif ('tanh' in s1):
            #     lp['non_linearity']=lasagne.nonlinearities.ScaledTanH(scale_in=.5,scale_out=2.4)
            # elif ('softmax' in s1):
            #     lp['non_linearity']=lasagne.nonlinearities.softmax
            # else:
            #     lp['non_linearity']=lasagne.nonlinearities.linear
            print('lasagne')
        else:
            a = ''
            # A number
            s1 = str.strip(s[1], ' \n')
            try:
                a = float(s1)
                if '.' not in s1:
                    a = int(s[1])
            # A tuple, a list or the original string
            except ValueError:
                if '(' in s[1]:
                    aa = str.split(str.strip(s1, ' ()\n'), ',')
                    a = []
                    try:
                        int(aa[0])
                        for aaa in aa:
                            a.append(int(aaa))
                        a = tuple(a)
                    except ValueError:
                        for aaa in aa:
                            a.append(float(aaa))
                        a = tuple(a)
                elif '[' in s[1]:
                    aa = str.split(str.strip(s1, ' []\n'), ',')
                    a = []
                    for aaa in aa:
                        a.append(aaa)
                elif (s1 == 'None'):
                    a = None
                elif (s1 == 'True'):
                    a = True
                elif (s1 == 'False'):
                    a = False
                else:
                    a = s1
            # Add a global drop value to drop layers
            s0 = str.strip(s[0], ' ')
            if (s0 == 'drop' and global_drop is not None):
                lp[s0] = global_drop
            else:
                lp[s0] = a
    return (lp)


def parse_text_file(net_name, NETPARS, lname='layers', dump=False):
    LAYERS = []
    if (net_name is not None):
        f = open(net_name + '.txt', 'r')
        for line in f:
            if (dump):
                print(line)
            line = str.strip(line, ' ')
            ll = str.split(line, '#')
            if (len(ll) > 1):
                line = str.strip(ll[0], ' ')
                if (line == ''):
                    continue
            else:
                line = ll[0]
            if ('name' in line or 'dict' in line):
                if ('global_drop' in NETPARS):
                    gd = NETPARS['global_drop']
                else:
                    gd = None
                lp = process_network_line(line, gd)
                if ('name' in line):
                    LAYERS.append(lp)
                else:
                    NETPARS[lp['dict']] = lp
                    del NETPARS[lp['dict']]['dict']
            else:
                [s, p] = process_param_line(line)
                NETPARS[s] = p

        f.close()
        # Check if NETPARS is using hinge loss use sigmoid non-linearity on final dense layer
        # Otherwise use softmax

        NETPARS[lname] = LAYERS


def dump_pars(NETPARS):
    import collections
    NETPARS = collections.OrderedDict(sorted(NETPARS.items()))
    for key in NETPARS:
        if (type(NETPARS[key]) is not list):
            print(key + ":" + str(NETPARS[key]))
    for key in NETPARS:
        if (type(NETPARS[key]) is list):
            print(key)
            for l in NETPARS[key]:
                if (key == 'layers'):
                    print(l, )
                else:
                    print(l, " ")

def process_parameters(net):
    PARS = {}
    parse_text_file(net, PARS, lname='layers', dump=True)
    #PARS['step_size'] = PARS['eta_init']
    PARS['Rstep_size'] = list(PARS['force_global_prob'])[1] * PARS['step_size']
    print('Rstep_size', PARS['Rstep_size'])
    PARS['nonlin_scale'] = .5

    return PARS
def sparse_process_parameters(PARS):
      PARS['step_size']=PARS['sparse_step_size']
      if ('sparse_batch_size' in PARS):
          PARS['batch_size']=PARS['sparse_batch_size']
      if ('sparse_global_prob' in PARS):
          PARS['force_global_prob']=PARS['sparse_global_prob']
      PARS['Rstep_size'] = list(PARS['force_global_prob'])[1] * PARS['step_size']
      print('Rstep_size', PARS['Rstep_size'])



def print_results(type,epoch,lo,ac):
    print("Final results: epoch", str(epoch))
    print(type+" loss:\t\t\t{:.6f}".format(lo))
    print(type+" acc:\t\t\t{:.6f}".format(ac))

def plot_OUTPUT(name='OUTPUT',code='',first=None,last=None):

    import numpy as np
    import pylab as py
    py.ion()
    havetrain=False
    oo=commands.check_output('grep Posi ' + name + '.txt  | cut -d" " -f2,3', shell=True)
    bp=[]
    bt=np.fromstring(commands.check_output('grep Train ' + name + '.txt | grep acc | cut -d":" -f2',shell=True),sep='\n\t\t\t')
    loss=np.fromstring(commands.check_output('grep Train ' + name + '.txt | grep loss | cut -d":" -f2',shell=True),sep='\n\t\t\t')
    fig=py.figure(2)
    py.plot(loss)
    py.figure(1)
    bv=np.fromstring(commands.check_output('grep Val ' + name + '.txt | grep acc | cut -d":" -f2',shell=True),sep='\n\t\t\t')

    ss='grep aggegate ' + name + '.txt | cut -d"," -f4 | cut -d")" -f1'
    try:
        aa=commands.check_output(ss,shell=True)
        atest=np.fromstring(aa,sep='\n\t\t\t')
        print(atest)
        if (type(atest) is np.ndarray and len(atest) > 0):
            atest = atest[-1]
        # ss = 'grep Post-train ' + name + '.txt | grep acc | cut -d":" -f2'
        # atrain = np.fromstring(commands.check_output(ss, shell=True), sep='\n\t\t\t')
        # if (type(atrain) is np.ndarray and len(atrain) > 0):
        #     havetrain = True
        #     atrain = atrain[-1]
    except:
        print('aggeg not found')


    print('Final',atest) #,atrain)
    if (first is not None and last is not None):
        bt=bt[first:last]
        bv=bv[first:last]
        if (bp!=[]):
            bp=bp[first:last]
        print(bv[-1],bt[-1])
    else:
        print(len(bt),bv[-1],bt[-1])
        if (havetrain>0):
            py.plot(len(bt)-2, atest, 'go', markersize=4)
            #py.plot(len(bt)-2, atrain, 'bo', markersize=4)
    py.plot(bt,label='train '+code)
    py.plot(bv,label='val '+code)
    if (bp!=[]):
        py.plot(bp,label='Pos')
    py.legend(loc=4)

    py.show()

def setup_net(PARS,OPS,  WR=None, SP=None, non_trainable=None):
    # Create the network architecture with the above placeholdes as the inputs.
    # TS is a list of tensors or tensors + a list of associated parameters (pool size etc.)
    loss, accuracy, TS = Conv_net_gpu.recreate_network(PARS, OPS['x'], OPS['y_'], OPS['Train'],WR=WR,SP=SP)
    VS = tf.trainable_variables()
    VS.reverse()

    dW_OPs, lall = Conv_net_gpu.back_prop(loss, accuracy, TS, VS, OPS['x'], PARS,non_trainable=non_trainable)


    OPS['loss']=loss
    OPS['accuracy']=accuracy
    OPS['TS']=TS
    OPS['VS']=VS
    OPS['dW_OPs']=dW_OPs
    OPS['lall']=lall

def run_epoch(train,i,OPS,PARS,sess,type='Train'):
    t1 = time.time()

    # Randomly shuffle the training data
    ii = np.arange(0, train[0].shape[0], 1)
    if ('Train' in type):
        np.random.shuffle(ii)
    tr = train[0][ii]
    if ('shift' in PARS or 'saturation' in PARS and type=='Train_sparse'):
        shift=0
        saturation=False
        if ('shift' in PARS):
            shift=PARS['shift']
        if ('saturation' in PARS):
            saturation=PARS['saturation']
        tr=rotate_dataset_rand(tr,shift=shift,saturation=saturation,gr=0)

    y = train[1][ii]
    lo = 0.
    acc = 0.
    ca = 0.
    batch_size=PARS['batch_size']
    for j in np.arange(0, len(y), batch_size):
        batch = (tr[j:j + batch_size], y[j:j + batch_size])
        if ('Train' in type):
            grad = sess.run(OPS['dW_OPs'], feed_dict={OPS['x']: batch[0], OPS['y_']: batch[1], OPS['Train']: True})
            acc += grad[-2]
            lo += grad[-1]
        else:
            act, lot = sess.run([OPS['accuracy'], OPS['loss']], feed_dict={OPS['x']: batch[0], OPS['y_']: batch[1], OPS['Train']: False})
            acc += act
            lo += lot
        ca += 1

    print('Epoch time', time.time() - t1)
    Conv_net_aux.print_results(type, i, lo/ca, acc/ca)
    return acc / ca, lo / ca

def finalize(test,OPS,PARS,net,sess):
            ac, lo= run_epoch(test,0,OPS,PARS,sess,type='Test')
            print('step,','0,', 'aggegate accuracy,', ac)
            saver = tf.train.Saver()
            save_path = saver.save(sess, "tmp/model_" + net.split('/')[1])
            print("Model saved in path: %s" % save_path)



