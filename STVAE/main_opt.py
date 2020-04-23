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
        #model.get_binary_signature(train)
        torch.save({'args': args,
                    'model.state.dict': model.state_dict()}, '_output/network.pt')
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
ARGS, STRINGS, EX_FILES, SMS = mprep.get_names(args)
if args.rerun:
    args.run_existing=True
# Get data device and output file
fout, device, DATA= mprep.setups(args, EX_FILES)

if not hasattr(ARGS[0],'opt_jump'):
    ARGS[0].opt_jump=1
    ARGS[0].enc_conv=False
ARGS[0].binary_thresh=args.binary_thresh
if 'vae' in args.type:
    models=mprep.get_models(device, fout, DATA[0][0].shape,STRINGS,ARGS,locals())
if args.network:
    sh=DATA[0][0].shape
    # parse the existing network coded in ARGS[0]
    arg=ARGS[0]
    if args.layers is not None:
        nf=sh[1]
        arg.lnti, arg.layers_dict = mprep.get_network(arg.layers,nf=nf)
        model = network.network(device, arg, arg.layers_dict, arg.lnti).to(device)
        temp = torch.zeros(1, sh[1], sh[2], sh[3]).to(device)
        bb = model.forward(temp)
        net_models = [model]
        if 'vae' not in args.type:
            models=net_models
sample=args.sample
classify=args.classify
reinit=args.reinit
run_existing=args.run_existing
conf=args.conf
embedd=args.embedd
num_test=args.num_test
num_train=args.num_train
nepoch=args.nepoch
lr=args.lr
network=args.network
ARGS[0].nti=args.nti
ARGS[0].num_test=num_test

if reinit:
    #model=mprep.make_model(device, fout, STRINGS[0],args,locals(), DATA[0][0].shape[1],DATA[0][0].shape[2], device, fout)
    model.load_state_dict(SMS[0]['model.state.dict'])
    ARGS=[args]
    strings, ex_file = mprep.process_strings(args)
    EX_FILES=[ex_file]
    tes = [DATA[2][0], DATA[2][0], DATA[2][1]]
    model.run_epoch(tes, 0, args.nti, None, None, None, d_type='test', fout=fout)
    exit()
fout.write(str(ARGS[0]) + '\n')
fout.flush()
# if (args.classify):
#     t1 = time.time()
#     classify(train,test,image_dim,opt_pre,opt_post,opt_mix,opt_class,device,args,fout,locals())
#     fout.write('Classified in {1:5.3f} seconds\n'.format(time.time()-t1))
#     exit()

if (run_existing and not reinit):

    if sample:
        model=models[0]
        model.load_state_dict(SMS[0]['model.state.dict'])
        aux.make_images(DATA[2],model,EX_FILES[0],args)
    elif network:
        if 'vae' in args.type:
            model = models[0]
            model.load_state_dict(SMS[0]['model.state.dict'])
            args=ARGS[0]
            dat, HVARS = aux.prepare_recons(model, DATA, args,fout)
            assign_cluster_labels(args,HVARS[0],HVARS[2],fout)
            train_new(args, HVARS[0], HVARS[2], device)
        elif embedd:
            net_model=net_models[0]
            net_model.load_state_dict(SMS[0]['model.state.dict'])
            #cc=net_model.get_binary_signature(DATA[0])
            tr = net_model.get_embedding(DATA[0]).detach().cpu().numpy()
            tr = tr.reshape(tr.shape[0], -1)
            trh = [tr, DATA[0][1]]
            te = net_model.get_embedding(DATA[2]).detach().cpu().numpy()
            te = te.reshape(te.shape[0], -1)
            teh = [te, DATA[2][1]]
            args.embedd = False
            args.type='net'
            args.nepoch=nepoch
            args.num_train=num_train
            args.lr=lr
            train_new(args, trh, teh, device)
        else:
            if args.layers is not None and not args.rerun:
                args.type='net'
                test_models(ARGS,SMS,DATA[2],fout)
    else:
        model=models[0]
        model.load_state_dict(SMS[0]['model.state.dict'])
        dat, HVARS = aux.prepare_recons(model, DATA, args, fout)
        #if args.hid_layers is not None:
        #        train_new(args, HVARS[0], HVARS[2], device)

else:
    #if ('vae' in args.type):
    if 'vae' in args.type:
        train_model(models[0], ARGS[0], EX_FILES[0], DATA, fout)
        dat,HVARS=aux.prepare_recons(models[0],DATA,args,fout)
        #assign_cluster_labels(args,HVARS[0],HVARS[2],fout)
        if args.hid_layers is not None:
                train_new(args, HVARS[0], HVARS[2], device)
            #args.type = 'net'
    else:
        train_model(net_models[0],args,EX_FILES[0],DATA,fout)
        if args.embedd:
            tr=net_models[0].get_embedding(DATA[0]).detach().cpu().numpy()
            tr=tr.reshape(tr.shape[0],-1)
            trh=[tr,DATA[0][1]]
            te=net_models[0].get_embedding(DATA[2]).detach().cpu().numpy()
            te=te.reshape(te.shape[0],-1)
            teh=[te,DATA[2][1]]
            args.embedd=False
            train_new(args,trh,teh,device)


# trainMU=None;trainLOGVAR=None;trainPI=None
# if args.classify:
#     args.nepoch=1000; args.lr=.01
#     train_new(models[0],args,train,test,device)

fout.write('DONE\n')
fout.flush()
if (not args.CONS):
        fout.close()



