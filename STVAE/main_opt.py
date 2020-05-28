import torch
from models_opt import STVAE_OPT
from models_mix import STVAE_mix
from models import STVAE
from models_mix_by_class import STVAE_mix_by_class
import numpy as np
import os
import sys
import argparse
import time
from Conv_data import get_data
from models import  get_scheduler
import aux
from class_on_hidden import train_new
import network
import mprep
from model_cluster_labels import assign_cluster_labels
from classify import classify




def test_models(ARGS, SMS, test, fout):

    len_test = len(test[0]);
    testMU = None;
    testLOGVAR = None;
    testPI = None
    CL_RATE = [];
    ls = len(SMS)
    CF = [conf] + list(np.zeros(ls - 1))
    # Combine output from a number of existing models. Only hard ones move to next model?
    tes=[test[0],test[0],test[1]]
    if (ARGS[0].n_class):
        for sm, model, args, cf in zip(SMS, models, ARGS, CF):
            model.load_state_dict(sm['model.state.dict'])
            if 'vae' in args.type:
                testMU, testLOGVAR, testPI = model.initialize_mus(tes[0], args.OPT)
            print(cf)
            iid, RY, cl_rate, acc = model.run_epoch_classify(tes, 'test', fout=fout, num_mu_iter=args.nti, conf_thresh=cf)
            CL_RATE += [cl_rate]
            len_conf = len_test - np.sum(iid)
            print("current number", len_conf)
            if (len_conf > 0):
                print(float(cl_rate) / len_conf)
        print(np.float(np.sum(np.array(CL_RATE))) / len_test)
    else:
        for sm, model, args, cf in zip(SMS, models, ARGS, CF):
            model.load_state_dict(sm['model.state.dict'])
            if 'vae' in args.type:
                testMU, testLOGVAR, testPI = model.initialize_mus(tes[0], args.OPT)
            model.run_epoch(tes, 0, args.nti, testMU, testLOGVAR, testPI, d_type='test', fout=fout)

def train_model(model, args, ex_file, DATA, fout):

    fout.write("Num train:{0}\n".format(DATA[0][0].shape[0]))
    train=DATA[0]; val=DATA[1]; test=DATA[2]
    trainMU=None; trainLOGVAR=None; trPI=None
    testMU=None; testLOGVAR=None; testPI=None
    valMU=None; valLOGVAR=None; valPI=None

    if 'vae' in args.type:
        trainMU, trainLOGVAR, trPI = model.initialize_mus(train[0], args.OPT)
        valMU, valLOGVAR, valPI = model.initialize_mus(val[0], args.OPT)
        testMU, testLOGVAR, testPI = model.initialize_mus(test[0], args.OPT)

    scheduler = get_scheduler(args, model)

    tes = [test[0], test[0], test[1]]
    if (val[0] is not None):
        vall=[val[0],val[0],val[1]]
    for epoch in range(args.nepoch):

        if (scheduler is not None):
            scheduler.step()
        t1 = time.time()
        tre = aux.erode(args.erode, train[0])
        tran = [train[0], tre, train[1]]
        trainMU, trainLOGVAR, trPI = model.run_epoch(tran, epoch, args.num_mu_iter, trainMU, trainLOGVAR, trPI,d_type='train', fout=fout)
        if (val[0] is not None):
             model.run_epoch(vall, epoch, args.nvi, valMU, valLOGVAR, valPI, d_type='val', fout=fout)

        fout.write('{0:5.3f}s'.format(time.time() - t1))
        fout.flush()



    if 'net' in args.type:
        if args.model_out is not None:
            ss=args.model_out+'.pt'
        else:
            ss='network.pt'
        torch.save({'args': args,
                    'model.state.dict': model.state_dict()}, '_output/'+ss)
    if 'vae' in args.type:
        fout.write('writing to ' + ex_file + '\n')
        torch.save({'args': args,
                    'model.state.dict': model.state_dict()}, '_output/' + ex_file + '.pt')
        aux.make_images(train, model, ex_file, args)
        if (args.n_class):
            model.run_epoch_classify(tran, 'train', fout=fout, num_mu_iter=args.nti)
            model.run_epoch_classify(tes, 'test', fout=fout, num_mu_iter=args.nti)
        elif args.cl is None:
            if args.hid_layers is None:
                model.run_epoch(tes, 0, args.nti, testMU, testLOGVAR, testPI, d_type='test', fout=fout)
    else:
            model.run_epoch(tes, 0, args.nti, None, None, None, d_type='test', fout=fout)




########################################
# Main Starts
#########################################
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
    description='Variational Autoencoder with Spatial Transformation')


args=aux.process_args(parser)
# reinit means you are taking part of an existing network as fixed and updating some other parts.
if args.rerun or args.reinit:
    args.run_existing=True

ARGS, STRINGS, EX_FILES, SMS = mprep.get_names(args)

# Get data device and output file
fout, device= mprep.setups(args, EX_FILES)
DATA=mprep.get_data_pre(args,args.dataset)
if not hasattr(args,'opt_jump'):
    args.opt_jump=1
    args.enc_conv=False

if 'vae' in args.type:
    models=mprep.get_models(device, fout, DATA[0][0].shape,STRINGS,ARGS,locals())
if args.network:
    sh=DATA[0][0].shape
    # parse the existing network coded in ARGS[0]
    arg=ARGS[0]
    if not hasattr(arg,'embedd_mult'):
        arg.embedd_mult=False
    if args.reinit: # Parse the new network
        arg=args
    if arg.layers is not None:
        nf=sh[1]
        arg.lnti, arg.layers_dict = mprep.get_network(arg.layers,nf=nf)
        model = network.network(device, arg, arg.layers_dict, arg.lnti).to(device)
        temp = torch.zeros(1, sh[1], sh[2], sh[3]).to(device)
        bb = model.forward(temp)
        net_models = [model]
        if 'vae' not in args.type:
            models=net_models
sample=args.sample
#classify=args.classify
reinit=args.reinit
run_existing=args.run_existing
conf=args.conf
embedd=args.embedd
num_test=args.num_test
num_train=args.num_train
nepoch=args.nepoch
lr=args.lr
network_flag=args.network
#ARGS[0].nti=args.nti
#ARGS[0].num_test=num_test
if (ARGS[0]==args):
    fout.write('Printing Args from input args\n')
    fout.write(str(ARGS[0]) + '\n')
else:
    fout.write('Printing Args from read-in model\n')
    fout.write(str(ARGS[0]) + '\n')
    fout.write('Printing Args from input args\n')
    fout.write(str(args) + '\n')

fout.flush()

if reinit:
    lnti, layers_dict = mprep.get_network(SMS[0]['args'].layers, nf=nf)
    model_old = network.network(device, SMS[0]['args'], layers_dict, lnti).to(device)
    temp = torch.zeros(1, sh[1], sh[2], sh[3]).to(device)
    bb = model_old.forward(temp)
    model_old.load_state_dict(SMS[0]['model.state.dict'])

    params = model_old.named_parameters()
    params2 = model.named_parameters()
    dict_params2 = dict(params2)
    # Loop over parameters of N1
    for name, param in params:
        if name in dict_params2:
            dict_params2[name].data.copy_(param.data)
    model.load_state_dict(dict_params2)




    # pretrained_dict = {}
    # model_dict=model.state_dict()
    # dict_params_new = dict(model.named_parameters())
    #
    # for k,kn in zip(SMS[0]['model.state.dict'].items(),model_dict.items()):
    #     if k[0].split('.')[1] not in args.update_layers and  k[0] in dict_params_new:
    #         print('copying:'+k[0].split('.')[1])
    #         dict_params_new[k[0]].data.copy_(k[1].data)
    #         #pretrained_dict[k[0]]=k[1]
    # #model_dict.update(pretrained_dict)
    # #model.load_state_dict(model_dict)
    # model.load_state_dict(dict_params_new)
    temp_data=model.get_embedding(DATA[0])
    train_model(model, args, EX_FILES[0], DATA, fout)
    if (args.embedd):
        if args.hid_dataset is not None:
            print('getting:'+args.hid_dataset)
            DATA = mprep.get_data_pre(args, args.hid_dataset)
        tr = model.get_embedding(DATA[0]) #.detach().cpu().numpy()
        tr = tr.reshape(tr.shape[0], -1)
        trh = [tr, DATA[0][1]]

        te = model.get_embedding(DATA[2]) #.detach().cpu().numpy()
        te = te.reshape(te.shape[0], -1)
        teh = [te, DATA[2][1]]
        args.embedd = False
        args.update_layers=None
        args.lr=args.hid_lr
        train_new(args, trh, teh, device)
    quit()

# if (args.classify):
#     t1 = time.time()
#     classify(train,test,image_dim,opt_pre,opt_post,opt_mix,opt_class,device,args,fout,locals())
#     fout.write('Classified in {1:5.3f} seconds\n'.format(time.time()-t1))
#     exit()

if (run_existing):

    if sample:
        model=models[0]
        model.load_state_dict(SMS[0]['model.state.dict'])
        aux.make_images(DATA[2],model,EX_FILES[0],args)
    elif network_flag:
        if 'vae' in args.type:
            model = models[0]
            model.load_state_dict(SMS[0]['model.state.dict'])
            dat, HVARS = aux.prepare_recons(model, DATA, args,fout)
            if args.hid_layers is not None:
                train_new(args, HVARS[0], HVARS[2], device)
            # assign_cluster_labels(args,HVARS[0],HVARS[2],fout)
            # train_new(args, HVARS[0], HVARS[2], device)
        elif embedd:

            net_model=net_models[0]
            net_model.load_state_dict(SMS[0]['model.state.dict'])
            #cc=net_model.get_binary_signature(DATA[0])
            model.embedd_layer = args.embedd_layer

            tr = net_model.get_embedding(DATA[0]) #.detach().cpu().numpy()
            tr = tr.reshape(tr.shape[0], -1)
            trh = [tr, DATA[0][1]]
            te = net_model.get_embedding(DATA[2]) #.detach().cpu().numpy()
            te = te.reshape(te.shape[0], -1)
            teh = [te, DATA[2][1]]
            args.embedd = False
            args.update_layers=None
            args.type='net'
            args.nepoch=nepoch
            args.num_train=num_train
            args.lr=lr
            train_new(args, trh, teh, device)
        else: # Test a sequence of models
            if args.layers is not None and not args.rerun:
                args.type='net'
                test_models(ARGS,SMS,DATA[2],fout)

else: # Totally new network
    if 'vae' in args.type:
        train_model(models[0], args, EX_FILES[0], DATA, fout)
        dat,HVARS=aux.prepare_recons(models[0],DATA,args,fout)
        #assign_cluster_labels(args,HVARS[0],HVARS[2],fout)
        if args.hid_layers is not None:
                train_new(args, HVARS[0], HVARS[2], device)
        #args.type = 'net'
    else:
        train_model(net_models[0],args,EX_FILES[0],DATA,fout)
        if args.embedd:
            if args.hid_dataset is not None:
                print('getting:' + args.hid_dataset)
                DATA = mprep.get_data_pre(args, args.hid_dataset)
            tr=net_models[0].get_embedding(DATA[0]) #.detach().cpu().numpy()
            tr=tr.reshape(tr.shape[0],-1)
            trh=[tr,DATA[0][1]]
            te=net_models[0].get_embedding(DATA[2]) #.detach().cpu().numpy()
            te=te.reshape(te.shape[0],-1)
            teh=[te,DATA[2][1]]
            args.embedd=False
            args.update_layers=None
            train_new(args,trh,teh,device)


fout.write('DONE\n')
fout.flush()
if (not args.CONS):
        fout.close()



