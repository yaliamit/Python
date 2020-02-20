
import sys
import numpy as np
import time
import network
import torch


def train_new(args,train,test,device):

    fout=sys.stdout
    print("In from hidden number of training",train[0].shape[0])
    print('In train new:')
    print(str(args))
    val = None


    args.lr=args.hid_lr
    net=network.network(device,args,args.hid_layers, args.hid_lnti).to(device)
    temp=torch.zeros(1,train[0].shape[1]).to(device)
    bb=net.forward(temp)
    # tot_pars = 0
    # for keys, vals in net.state_dict().items():
    #     fout.write(keys + ',' + str(np.array(vals.shape)) + '\n')
    #     tot_pars += np.prod(np.array(vals.shape))
    #fout.write('tot_pars for fc,' + str(tot_pars) + '\n')
    scheduler=None
    tran=[train[0],train[0],train[1]]
    for epoch in range(args.nepoch):
        if (scheduler is not None):
            scheduler.step()
        t1=time.time()
        net.run_epoch(tran,epoch, d_type='train',fout=fout)
        if (val is not None):
                net.run_epoch(val,epoch, type='val',fout=fout)
        if (np.mod(epoch,10)==9 or epoch==0):
            fout.write('epoch: {0} in {1:5.3f} seconds'.format(epoch,time.time()-t1))
            fout.flush()


    tes=[test[0],test[0],test[1]]
    net.run_epoch(tes, 0, d_type='test', fout=fout)
    fout.flush()




