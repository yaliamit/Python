import numpy as np


def assign_cluster_labels(args,train,test,fout):

    numcl=np.max(train[1])+1
    # Get the highest scoring mixture component for each example train then test.
    maxpitr=np.argmax(train[0][:,-args.n_mix:],axis=1)
    mmptr=np.max(train[0][:,-args.n_mix:],axis=1)
    iitr=np.where(mmptr>=.5)[0]
    maxpite=np.argmax(test[0][:,-args.n_mix:],axis=1)
    mmpte=np.max(test[0][:,-args.n_mix:],axis=1)
    iite = np.where(mmpte >= .7)[0]
    cls=-np.ones(args.n_mix)
    correctte=0; correctte_s=0
    correcttr=0; correcttr_s=0
    # For each cluster
    for c in range(args.n_mix):

        # If more than one training data assigned to that cluster find the majority class label
        # in that cluster.
        if np.sum(maxpitr==c)>0:
            uu=np.unique(train[1][maxpitr==c],return_counts=True)
            cls[c]=uu[0][np.argmax(uu[1])]
            # Find how many test examples in this cluster have the label cls[c]
            correctte+=np.sum(test[1][maxpite==c]==cls[c])
            correctte_s+=np.sum(test[1][iite][maxpite[iite]==c]==cls[c])
            # Find how many train examples in this cluster have the label cls[c]
            correcttr+=np.sum(train[1][maxpitr==c]==cls[c])
            correcttr_s+=np.sum(train[1][iitr][maxpitr[iitr]==c]==cls[c])

    acc=correctte/test[0].shape[0]
    fout.write('\n ====> Ep {}: {} Full loss: {:.4F}, Full acc: {:.4F} \n'.format('test', 0,
                                                                                  0,
                                                                                  acc))
    print(correctte_s/len(iite),len(iite))
    acc = correcttr / train[0].shape[0]
    fout.write('\n ====> Ep {}: {} Full loss: {:.4F}, Full acc: {:.4F} \n'.format('train', 0,
                                                                                  0,
                                                                                  acc))
    print(correcttr_s / len(iitr),len(iitr))