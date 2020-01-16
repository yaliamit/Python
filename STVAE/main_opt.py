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
from classify import classify




def test_models(ARGS, SMS, test, fout):
    iid = None;
    len_test = len(test[0]);
    ACC = [];
    CL_RATE = [];
    ls = len(SMS)
    CF = [conf] + list(np.zeros(ls - 1))
    # Combine output from a number of existing models. Only hard ones move to next model?
    if (ARGS[0].n_class):
        for sm, model, args, cf in zip(SMS, models, ARGS, CF):
            model.load_state_dict(sm['model.state.dict'])
            testMU, testLOGVAR, testPI = model.initialize_mus(test[0], args.OPT)
            print(cf)
            iid, RY, cl_rate, acc = model.run_epoch_classify(test, 'test', fout=fout, num_mu_iter=args.nti, conf_thresh=cf)
            CL_RATE += [cl_rate]
            len_conf = len_test - np.sum(iid)
            print("current number", len_conf)
            if (len_conf > 0):
                print(float(cl_rate) / len_conf)
        print(np.float(np.sum(np.array(CL_RATE))) / len_test)
    else:
        for sm, model, args, cf in zip(SMS, models, ARGS, CF):
            model.load_state_dict(sm['model.state.dict'])
            testMU, testLOGVAR, testPI = model.initialize_mus(test[0], args.OPT)
            model.run_epoch(test, 0, args.nti, testMU, testLOGVAR, testPI, d_type='test', fout=fout)

def train_model(model, args, ex_file, DATA, fout):

    fout.write("Num train:{0}\n".format(DATA[0][0].shape[0]))
    train=DATA[0]; val=DATA[1]; test=DATA[2]
    trainMU=None; trainLOGVAR=None; trPI=None
    if 'vae' in args.type:
        trainMU, trainLOGVAR, trPI = model.initialize_mus(train[0], args.OPT)
        valMU, valLOGVAR, valPI = model.initialize_mus(val[0], args.OPT)
        testMU, testLOGVAR, testPI = model.initialize_mus(test[0], args.OPT)

    scheduler = get_scheduler(args, model)

    tes = [test[0], test[0], test[1]]
    for epoch in range(args.nepoch):

        if (scheduler is not None):
            scheduler.step()
        t1 = time.time()
        tre = aux.erode(args.erode, train[0])
        tran = [train[0], tre, train[1]]
        trainMU, trainLOGVAR, trPI = model.run_epoch(tran, epoch, args.num_mu_iter, trainMU, trainLOGVAR, trPI,
                                                     d_type='train', fout=fout)
        if (val[0] is not None):
            model.run_epoch(val, epoch, args.nvi, valMU, valLOGVAR, valPI, d_type='val', fout=fout)

        fout.write('{0:5.3f}s'.format(time.time() - t1))
        fout.flush()

    fout.write('writing to ' + ex_file + '\n')


    if 'vae' in args.type:
        torch.save({'args': args,
                    'model.state.dict': model.state_dict()}, '_output/' + ex_file + '.pt')
        aux.make_images(train, model, ex_file, args)
        if (args.n_class):
            model.run_epoch_classify(tran, 'train', fout=fout, num_mu_iter=args.nti)
            model.run_epoch_classify(tes, 'test', fout=fout, num_mu_iter=args.nti)
        elif args.cl is None:
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

# Get data device and output file
fout, device, DATA= mprep.setups(args, EX_FILES)

if 'vae' in args.type:
    models=mprep.get_models(device, fout, DATA[0][0].shape,STRINGS,ARGS,locals())
if args.network:
    net_models=mprep.get_network(device, DATA[0][0].shape,args)
    if 'vae' not in args.type:
        models=net_models
sample=args.sample
classify=args.classify
reinit=args.reinit
run_existing=args.run_existing
conf=args.conf
num_test=args.num_test
network=args.network
ARGS[0].nti=args.nti
ARGS[0].num_test=num_test

if reinit:
    models[0].load_state_dict(SMS[0]['model.state.dict'])
    model_new=mprep.make_model(device, fout, STRINGS[0],args,locals(), DATA[0][0].shape[1],DATA[0][0].shape[2], device, fout)
    model_new.conv.conv.weight.data=models[0].conv.conv.weight.data
    models=[model_new]
    ARGS=[args]
    strings, ex_file = mprep.process_strings(args)
    EX_FILES=[ex_file]

fout.write(str(ARGS[0]) + '\n')
fout.flush()
# if (args.classify):
#     t1 = time.time()
#     classify(train,test,image_dim,opt_pre,opt_post,opt_mix,opt_class,device,args,fout,locals())
#     fout.write('Classified in {1:5.3f} seconds\n'.format(time.time()-t1))
#     exit()

if (run_existing and not reinit):
    if (classify):
        train_new(models[0],args,DATA,device)
    elif sample:
        model=models[0]
        model.load_state_dict(SMS[0]['model.state.dict'])
        aux.make_images(DATA[2],model,EX_FILES[0],args)
    elif network:
        model = models[0]
        model.load_state_dict(SMS[0]['model.state.dict'])
        if ('vae' in args.type):
            dat, HVARS = aux.prepare_recons(model, DATA, args)
            train_new(model, args, HVARS[0], HVARS[2], device)
        else:
            dat=DATA
        args.type='net'
        train_model(net_models[0], args, EX_FILES[0], dat, fout)
    else:
        test_models(ARGS,SMS,DATA[2],fout)
else:
    #if ('vae' in args.type):
    train_model(models[0], ARGS[0], EX_FILES[0], DATA, fout)
    if ('vae' in args.type and args.network):
            dat,HVARS=aux.prepare_recons(models[0],DATA,args)
            train_new(models[0], args, HVARS[0], HVARS[2], device)
            args.type = 'net'
            train_model(net_models[0],args,EX_FILES[0],dat,fout)

# trainMU=None;trainLOGVAR=None;trainPI=None
# if args.classify:
#     args.nepoch=1000; args.lr=.01
#     train_new(models[0],args,train,test,device)

fout.write('DONE\n')
fout.flush()
if (not args.CONS):
        fout.close()

