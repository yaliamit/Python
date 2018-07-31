from __future__ import print_function


# import lasagne
# import pickle
# import os
# import make_net
# import sys
import numpy as np
import h5py


def one_hot(values,n_values=10):
    n_v = np.maximum(n_values,np.max(values) + 1)
    oh=np.eye(n_v)[values]
    return oh

def get_cifar_10():
    tr=np.float32(np.load('/project2/cmsc25025/mnist/CIFAR_10.npy'))
    tr_lb=np.int32(np.load('/project2/cmsc25025/mnist/CIFAR_labels.npy'))
    #tr=tr.reshape((-1,np.prod(np.array(tr.shape)[1:4])))
    train_data=tr[0:45000]/255.
    train_labels=one_hot(tr_lb[0:45000])
    val_data=tr[45000:]/255.
    val_labels=one_hot(tr_lb[45000:])
    test_data=np.float32(np.load('/project/cmsc25025/mnist/CIFAR_10_test.npy'))
    #test_data=test_data.reshape((-1,np.prod(np.array(test_data.shape)[1:4])))
    test_data=test_data/255.
    test_labels=one_hot(np.int32(np.load('/project/cmsc25025/mnist/CIFAR_labels_test.npy')))
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

def get_cifar(data_set='cifar10'):
    
    filename = '/project2/cmsc25025/mnist/'+data_set+'_train.hdf5'
    f = h5py.File(filename, 'r')
    key = list(f.keys())[0]
    # Get the data
    tr = f[key]
    key = list(f.keys())[1]
    tr_lb=f[key]
    train_data=np.float32(tr[0:45000])/255.
    train_labels=one_hot(tr_lb[0:45000])
    val_data=np.float32(tr[45000:])/255.
    val_labels=one_hot(tr_lb[45000:])
    filename = '/project2/cmsc25025/mnist/'+data_set+'_test.hdf5'
    f = h5py.File(filename, 'r')
    key = list(f.keys())[0]
    # Get the data
    test_data = np.float32(f[key])/255.
    key = list(f.keys())[1]
    test_labels=one_hot(f[key])
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

def get_mnist():

    labels=np.float32(np.load('/project/cmsc25025/mnist/MNIST_labels.npy'))
    data=np.float64(np.load('/project/cmsc25025/mnist/MNIST.npy'))
    print(data.shape)
    data=np.float32(data)/255.
    train_dat=data[0:50000]
    train_labels=one_hot(np.int32(labels[0:50000]))
    val_dat=data[50000:60000]
    val_labels=one_hot(np.int32(labels[50000:60000]))
    test_dat=data[60000:70000]
    test_labels=one_hot(np.int32(labels[60000:70000]))
    
    return (train_dat, train_labels), (val_dat, val_labels), (test_dat, test_labels)

def get_data(data_set):
    if ('cifar' in data_set):
        return(get_cifar(data_set=data_set))
    elif (data_set=="mnist"):
        return(get_mnist())
    elif (data_set=="mnist_transform"):
        return(get_mnist_trans())

def process_param_line(line):

            # Split parameter line on :
            s=str.split(line,':')
            s1=str.strip(s[1],' ;\n')
            # Analyze value
            try:
                # Float or int
                a=float(s1)
                if '.' not in s1:
                    a=int(s1)
            # None, True, False, or list of names.
            except ValueError:
                if (s1=='None'):
                    a=None
                elif (s1=='True'):
                    a=True
                elif (s1=='False'):
                    a=False
                else:
                    if '(' in s1:
                            aa=str.split(str.strip(s1,' ()\n'),',')
                            a=[]
                            for aaa in aa:
                                a.append(float(aaa))
                            a=tuple(a)
                    else:
                        s11=s1.split(',')
                        if (len(s11)==1):
                            a=s1
                        else:
                            a=[]
                            for ss in s11:
                                try:
                                    aa=int(ss)
                                    a.append(aa)
                                except ValueError:
                                    if (ss != ''):
                                        a.append(ss)


            return(s[0],a)

def process_network_line(line,global_drop):
        # break line on the ; each segment is a parameter for the layer of that line
            sss=str.split(line,';')
            lp={}
            for ss in sss:
                # Split between parameter name and value
                s=str.split(ss,':')
                s1=str.strip(s[1],' \n')
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
                    a=''
                    # A number
                    s1=str.strip(s[1],' \n')
                    try:
                        a=float(s1)
                        if '.' not in s1:
                            a=int(s[1])
                    # A tuple, a list or the original string
                    except ValueError:
                        if '(' in s[1]:
                            aa=str.split(str.strip(s1,' ()\n'),',')
                            a=[]
                            try:
                                int(aa[0])
                                for aaa in aa:
                                    a.append(int(aaa))
                                a=tuple(a)
                            except ValueError:
                                for aaa in aa:
                                    a.append(float(aaa))
                                a=tuple(a)
                        elif '[' in s[1]:
                            aa=str.split(str.strip(s1,' []\n'),',')
                            a=[]
                            for aaa in aa:
                                a.append(aaa)
                        elif (s1=='None'):
                            a=None
                        elif (s1=='True'):
                            a=True
                        elif (s1=='False'):
                            a=False
                        else:
                            a=s1
                    # Add a global drop value to drop layers
                    s0=str.strip(s[0],' ')
                    if (s0=='drop' and global_drop is not None):
                        lp[s0]=global_drop
                    else:
                        lp[s0]=a
            return(lp)



def parse_text_file(net_name,NETPARS,lname='layers', dump=False):


        LAYERS=[]
        if (net_name is not None):
            f=open(net_name+'.txt','r')
            for line in f:
                if (dump):
                     print(line,end="")
                line=str.strip(line,' ')
                ll=str.split(line,'#')
                if (len(ll)>1):
                    line=str.strip(ll[0],' ')
                    if(line==''):
                        continue
                else:
                    line=ll[0]
                if ('name' in line or 'dict' in line):
                    if ('global_drop' in NETPARS):
                        gd=NETPARS['global_drop']
                    else:
                        gd=None
                    lp=process_network_line(line,gd)
                    if ('name' in line):
                        LAYERS.append(lp)
                    else:
                        NETPARS[lp['dict']]=lp
                        del NETPARS[lp['dict']]['dict']
                else:
                    [s,p]=process_param_line(line)
                    NETPARS[s]=p

            f.close()
        # Check if NETPARS is using hinge loss use sigmoid non-linearity on final dense layer
        # Otherwise use softmax



            NETPARS[lname]=LAYERS


def dump_pars(NETPARS):
        import collections
        NETPARS = collections.OrderedDict(sorted(NETPARS.items()))
        for key in NETPARS:
            if (type(NETPARS[key]) is not list):
                print(key+":"+str(NETPARS[key]))
        for key in NETPARS:
            if (type(NETPARS[key]) is list):
                print(key)
                for l in NETPARS[key]:
                    if (key=='layers'):
                        print(l,)
                    else:
                        print(l," ",end="")