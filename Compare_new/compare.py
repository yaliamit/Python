#from __future__ import print_function

import numpy as np
import lasagne
import theano
import theano.tensor as T
import os
import make_net


# Standardize an array of images in one operation
def np_standardize(input):


    s=np.std(input,axis=2,keepdims=True)#.reshape((n0,1)),reps=n1)
    m=np.mean(input,axis=2,keepdims=True)

    output=(input-m)/s
    return np.squeeze(output)




# Get shifted correlations of two arrays of images and sum on vertical axis.
def get_shifted_correlations(input_std,NETPARS):

    num=input_std.shape[1]
    leng=input_std.shape[-1]
    vdim=input_std.shape[-2]
    dim=input_std.shape[2]
    sr=NETPARS['corr_shift']
    corrs=np.zeros((num,leng,2*sr+1))
    for l in range(leng):
        # Loop over range of possible shifts
        for ll in np.arange(l-sr,l+sr+1):
            if (ll>=0 and ll<leng):
                tcor=np.sum(input_std[0,...,l]*input_std[1,...,ll],axis=1)/dim
                # Add correlations vertically at same horizontal location for dp
                if (tcor.ndim>1):
                    corrs[:,l,ll-l+sr]=np.sum(tcor,axis=1)
                else:
                    corrs[:,l,ll-l+sr]=tcor
    corrs=(corrs+np.float32(1.))/np.float32(2.)
    return(corrs)

# Dynamic programming optimization of matching cost
def optimize_dp(corrs,NETPARS):

    jump=1
    # Original range of shifts for correlation computation
    sr=NETPARS['corr_shift']
    # Current search range for optimizing must be less than sr
    srr=sr
    num=corrs.shape[0]
    leng=corrs.shape[1]
    nsr=corrs.shape[2]
    table_state=-np.ones((num,leng,nsr))
    table_cost=-10000*np.ones((num,leng,nsr))
    table_cost[:,0,]=corrs[:,0,]
    for l in np.arange(jump,leng,jump):
        prel=l-jump
        # For each state (shift) of l find best allowable (shift) state of l-1
        for s in np.arange(l-srr,l+srr+1,jump):
            if (s>=0 and s<leng):
                lowt=np.maximum(prel-srr,0);
                # Can't use a matching location that comes before the matching location of the previous step
                hight=np.minimum(prel+srr+1,s+1)
                if (hight>lowt):
                    iit=np.arange(lowt,hight,jump)-prel+sr
                    curr=np.max(table_cost[:,prel,iit],axis=1)
                    tcurr=np.argmax(corrs[:,prel,iit],axis=1)
                    table_state[:,l,s-l+sr]=tcurr+lowt-prel+sr
                    table_cost[:,l,s-l+sr]=corrs[:,l,s-l+sr]+curr



    maxc=np.max(table_cost[:,-1,],axis=1)

    return(maxc)



def run_network_on_all_pairs(NETPARS):

    num_seqs=NETPARS['num_seqs']

    import make_seqs
    ims1, ims1a=make_seqs.make_seqs(NETPARS)


    input_var=theano.typed_list.TypedListType(theano.tensor.dtensor4)()
    for j in range(2):
            theano.typed_list.append(input_var,T.dtensor4())
    NETPARS['layers'][0]['dimx']=ims1.shape[2]
    NETPARS['layers'][0]['dimy']=ims1.shape[3]
    NETPARS['layers'][0]['num_input_channels']=ims1.shape[1]
    network=make_net.build_cnn_on_pars(input_var, NETPARS)

    if (os.path.isfile(NETPARS['net_name']+'.npy')):
        spars=np.load(NETPARS['net_name']+'.npy')
        lasagne.layers.set_all_param_values(network,spars)
    else:
        print('no network '+NETPARS['net_name']+'.npy')
        return
    test_corr = lasagne.layers.get_output(network, deterministic=True)
    test_fn = theano.function([input_var], [test_corr])


    # Get low dim mapping (256) of every window (more or less) for two sequences with same digit labels
    tcorr_same=test_fn([ims1,ims1a])
    # Standardize for correlation computation as inner product
    tt_same_std=np_standardize(tcorr_same[0])
    temp=np.copy(tt_same_std)
    ii=np.arange(0,num_seqs)
    # Randomly shuffle second sequence so that match is non-trivial i.e not 1-1, 2-2 etc.
    np.random.shuffle(ii)
    iii=np.copy(ii)
    temp[1,]=tt_same_std[1,iii]
    dps=np.zeros((num_seqs,num_seqs))
    # For each cyclic permutation of second sequence get optimal correlations for same index images of two sequences.
    for n in range(num_seqs):
        temp[1,]=tt_same_std[1,np.roll(ii,-n),]
        corrs_same=get_shifted_correlations(temp,NETPARS)
        # Allow small shifts in correlated subimages.
        dps[n,]=optimize_dp(corrs_same,NETPARS)
    dps=dps.transpose()
    dpss=dps.copy()
    for n in range(num_seqs):
        dpss[n,]=np.roll(dps[n,],n)
    print("done with DP")
    # We want to minimize not maximize.
    dpss=np.max(dpss)-dpss

    #dps=dps.transpose()
    match_them(dpss,iii)
    print('done ')


# Hungarian algorithm for matching the two sequences based on optimized matching of each pair.
def match_them(matrix,iii):
    from munkres import Munkres, print_matrix
    omatrix=np.copy(matrix)
    oii=np.argmin(omatrix,axis=1)

    m = Munkres()
    indexes = m.compute(matrix)
    jjj=np.zeros((len(iii),2))
    # k -> perm^{-1}(k)
    for i in range(len(iii)):
        jjj[iii[i],0]=iii[i]
        jjj[iii[i],1]=i
    #print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    for i, (row, column) in enumerate(indexes):
        value = omatrix[row][column]
        total += value
        print '(%s:%d,%s:%d,%s:%d,%s:%d) -> %f' % ('sequence',row,'truth', jjj[i,1],'match',column,'closest', oii[i], value)
    print 'total cost: %d' % total
    error=np.double(np.sum(np.array(indexes)[:,1]!=jjj[:,1]))/np.double(len(iii))
    error_e=np.double(np.sum(np.array(indexes)[:,1]!=oii))/np.double(len(iii))
    print('ERROR with joint matching',error,'ERROR with simple max of pairwise math',error_e)