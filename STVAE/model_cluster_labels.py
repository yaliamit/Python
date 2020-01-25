import numpy as np


def assign_cluster_labels(args,train,test):

    numcl=np.max(train[1])+1
    maxpitr=np.argmax(train[0][:,-args.n_mix:],axis=1)
    maxpite=np.argmax(test[0][:,-args.n_mix:],axis=1)

    cls=-np.ones(args.n_mix)
    correct=0
    for c in range(args.n_mix):

        if np.sum(maxpitr==c)>0:
            cls[c]=np.argmax(np.unique(train[1][maxpitr==c],return_counts=True)[1])

            correct+=np.sum(test[1][maxpite==c]==cls[c])


    print(correct/test[0].shape[0])
    print("hello")