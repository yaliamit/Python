import mnist
import numpy as np
import scipy.misc
import crop
import theano
import theano.tensor as T
import time
import lasagne
import os



def make_seqs(NETPARS):
    slength=NETPARS['slength']
    num_seqs=NETPARS['num_seqs']
    from_font=False
    if ('from_font' in NETPARS):
        from_font=NETPARS['from_font']
    if ('seed' in NETPARS):
        np.random.seed(NETPARS['seed'])

    imheight=48
    imwidth=48
    imheight1=np.int32(imheight*1.)
    imwidth1=np.int32(imwidth*1.)
    testheight=max(imheight,imheight1)
    testmarg=np.int32(np.floor(testheight-imheight)/2)
    testmarg1=np.int32(np.floor(testheight-imheight1)/2)
    print("Loading data...")
    # Load mnist data
    X_train, y_train, X_val, y_val, X_test, y_test = mnist.load_dataset()
    if (from_font):
        Xfont=crop.get_fonts()

    incr=30
    lrange=3
    lrange2=lrange*2+1
    # Random digit labels for each sequence.
    labels=np.floor(np.random.rand(num_seqs,slength)*10)
    TEST=[]
    TEST1=[]

    # Create num_seqs sequences of length slength
    for s in range(num_seqs):
        # Index of first digit
        begin=begin1=np.int32(0)
        # Two different sequences.
        ii=labels[s,:]
        test=np.zeros((testheight,imwidth*slength))
        test1=np.zeros((testheight,imwidth*slength))

        # Loop over length and place digits of the two sequences
        for k in range(slength):
            # Examples of the digit k of the two sequences.
            jj=np.where(y_val==ii[k])[0]
            rs=np.int32(np.floor(np.random.rand(2)*np.double(len(jj))))

            # Get the sample and resize to correct height and width
            # Version 1
            sample=scipy.misc.imresize(np.squeeze(X_val[jj[rs[0]],]),(imheight,imwidth))
            if (not from_font):
                sample1=np.squeeze(X_val[jj[rs[1]],])
            else:
                # Could be asked to get it from font.
                sample1=Xfont[np.int32(ii[k]),]
            sample1=scipy.misc.imresize(sample1,(imheight1,imwidth1))
            # Add samples to images
            test[testmarg:testmarg+sample.shape[0],begin:begin+sample.shape[1]]=\
                np.maximum(sample,test[testmarg:testmarg+sample.shape[0],begin:begin+sample.shape[1]])
            test1[testmarg1:testmarg1+sample1.shape[0],begin1:begin1+sample1.shape[1]]=\
                np.maximum(sample1,test1[testmarg1:testmarg1+sample1.shape[0],begin1:begin1+sample1.shape[1]])
            # Increment begin index for next digits
            begin+=np.int32(np.floor(np.random.rand()*lrange2)-lrange+incr)
            begin1+=np.int32(np.floor(np.random.rand()*lrange2)-lrange+incr)

        TEST.append(test)
        TEST1.append(test1)


    if (NETPARS['gr']):
        import pylab as py
        ii=range(num_seqs)
        np.random.shuffle(ii)
        for i in range(5):
            py.figure(num=1,figsize=(12,2),dpi=80)
            py.subplot(121)
            py.imshow(TEST[ii[i]],aspect='equal')
            py.axis('off')
            py.subplot(122)
            py.imshow(TEST1[ii[i]],aspect='equal')
            py.axis('off')
            py.show()

    TEST=np.floatX(np.expand_dims(np.array(TEST),1))
    TEST1=np.floatX(np.expand_dims(np.array(TEST1),1))

    return TEST,TEST1