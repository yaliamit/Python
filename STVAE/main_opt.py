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
from classify import classify


def get_names(args):
    ARGS = []
    STRINGS = []
    EX_FILES = []
    SMS = []
    if (args.run_existing):
        # This overides model file name
        names = args.model
        for i, name in enumerate(names):
            sm = torch.load('_output/' + name + '.pt',map_location='cpu')
            SMS += [sm]
            if ('args' in sm):
                args = sm['args']
            ARGS += [args]
            strings, ex_file = process_strings(args)
            STRINGS += [strings]
            EX_FILES += [ex_file]
    else:
        ARGS.append(args)
        strings, ex_file = process_strings(args)
        STRINGS += [strings]
        EX_FILES += [ex_file]

    return ARGS, STRINGS, EX_FILES, SMS

def get_network(sh,ARGS):

    models=[]
    model=network.network(device,sh[1],sh[2],ARGS[0]).to(device)
    models+=[model]

    return models

def get_models(sh,STRINGS,ARGS, locs):

    h = sh[1]
    w = sh[2]

    models = []
    for strings, args in zip(STRINGS, ARGS):
        model=make_model(strings,args,locs, h, w, device, fout)
        models += [model]

    return models


def  make_model(strings,args,locs, h, w, device, fout):
        model = locs['STVAE' + strings['opt_mix'] + strings['opt_class']](h, w, device, args).to(device)
        tot_pars = 0
        for keys, vals in model.state_dict().items():
            fout.write(keys + ',' + str(np.array(vals.shape)) + '\n')
            tot_pars += np.prod(np.array(vals.shape))
        fout.write('tot_pars,' + str(tot_pars) + '\n')
        return model

def setups(args, EX_FILES):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_gpu = args.gpu and torch.cuda.is_available()
    if (use_gpu and not args.CONS):
        fout = open('_OUTPUTS/OUT_' + EX_FILES[0] + '.txt', 'w')
    else:
        args.CONS = True
        fout = sys.stdout


    device = torch.device("cuda:" + str(args.gpu - 1) if use_gpu else "cpu")
    fout.write('Device,' + str(device) + '\n')
    fout.write('USE_GPU,' + str(use_gpu) + '\n')

    args = args
    PARS = {}
    PARS['data_set'] = args.dataset
    PARS['num_train'] = args.num_train // args.mb_size * args.mb_size
    PARS['nval'] = args.nval
    if args.cl is not None:
        PARS['one_class'] = args.cl

    train, val, test, image_dim = get_data(PARS)
    if (args.num_test>0):
        ntest=test[0].shape[0]
        ii=np.arange(0, ntest, 1)
        np.random.shuffle(ii)
        test=[test[0][ii[0:args.num_test]], test[1][ii[0:args.num_test]]]
    print('num_train', train[0].shape[0])


    return fout, device, [train, val, test]

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

            if (iid is not None):
                test = [test[0][iid], test[1][iid]]
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

    for epoch in range(args.nepoch):
        if (scheduler is not None):
            scheduler.step()
        t1 = time.time()
        trainMU, trainLOGVAR, trPI = model.run_epoch(train, epoch, args.num_mu_iter, trainMU, trainLOGVAR, trPI,
                                                     d_type='train', fout=fout)
        if (val[0] is not None):
            model.run_epoch(val, epoch, args.nvi, valMU, valLOGVAR, valPI, d_type='val', fout=fout)

        fout.write('epoch: {0} in {1:5.3f} seconds\n'.format(epoch, time.time() - t1))
        fout.flush()

    fout.write('writing to ' + ex_file + '\n')

    torch.save({'args': args,
                'model.state.dict': model.state_dict()}, '_output/' + ex_file + '.pt')
    if 'vae' in args.type:
        aux.make_images(train, model, ex_file, args)
        if (args.n_class):
            model.run_epoch_classify(train, 'train', fout=fout, num_mu_iter=args.nti)
            model.run_epoch_classify(test, 'test', fout=fout, num_mu_iter=args.nti)
        elif args.cl is None:
            model.run_epoch(test, 0, args.nti, testMU, testLOGVAR, testPI, d_type='test', fout=fout)
    else:
        model.run_epoch(test, 0, args.nti, None, None, None, d_type='test', fout=fout)


def process_strings(args):
    strings={'opt_pre':'', 'mm_pre':'', 'opt_post':'', 'opt_mix':'', 'opt_class':'', 'cll':''}
    if (args.OPT):
        strings['opt_pre']='OPT_'
        strings['opt_post']='_OPT'
    if (args.only_pi):
        strings['opt_pre'] = 'PI_'
        strings['opt_post'] = '_PI'
    if (args.n_mix>=1):
        strings['opt_mix']='_mix'
    if (args.MM):
        strings['mm_pre']='_MM'
    if (args.n_class>0):
        strings['opt_class']='_by_class'
    if (args.cl is not None):
        strings['cll']=str(args.cl)
    ex_file = strings['opt_pre'] + strings['opt_class'] + args.type + '_' + args.transformation + \
              '_' + str(args.num_hlayers) + '_mx_' + str(args.n_mix) + '_sd_' + str(args.sdim) + '_cl_' + strings['cll']
    return strings, ex_file


########################################
# Main Starts
#########################################
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
    description='Variational Autoencoder with Spatial Transformation')

args=aux.process_args(parser)
ARGS, STRINGS, EX_FILES, SMS = get_names(args)

# Get data device and output file
fout, device, DATA= setups(args, EX_FILES)

if 'vae' in args.type:
    models=get_models(DATA[0][0].shape,STRINGS,ARGS,locals())
if args.network:
    net_models=get_network(DATA[0][0].shape,ARGS)
sample=args.sample
classify=args.classify
reinit=args.reinit
run_existing=args.run_existing
conf=args.conf
num_test=args.num_test

ARGS[0].nti=args.nti
ARGS[0].num_test=num_test

if reinit:
    models[0].load_state_dict(SMS[0]['model.state.dict'])
    model_new=make_model(STRINGS[0],args,locals(), DATA[0][0].shape[1],DATA[0][0].shape[2], device, fout)
    model_new.conv.conv.weight.data=models[0].conv.conv.weight.data
    models=[model_new]
    ARGS=[args]
    strings, ex_file = process_strings(args)
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
        aux.make_images(DATA[2],model,EX_FILES[0],ARGS[0])
    else:
        test_models(ARGS,SMS,DATA[2],fout)
else:
    if ('vae' in args.type):
        train_model(models[0], ARGS[0], EX_FILES[0], DATA, fout)
    if (args.network):
        dat=[]
        for k in range(3):
            if (DATA[k][0] is not None):
                INP = torch.from_numpy(DATA[k][0].transpose(0, 3, 1, 2))
                INP = INP[0:args.network_num_train]
                RR=[]
                for j in np.arange(0, INP.shape[0], 500):
                    inp=INP[j:j+500]
                    rr=models[0].recon(inp,0)
                    RR+=[rr.detach().cpu().numpy()]
                RR=np.concatenate(RR)
                tr=RR.reshape(-1,1,28,28).transpose(0,2,3,1)
                dat+=[[tr,DATA[k][1][0:args.network_num_train]]]
            else:
                dat+=[DATA[k]]
        print("Hello")
        args.type='net'
        train_model(net_models[0],ARGS[0],EX_FILES[0],dat,fout)

# trainMU=None;trainLOGVAR=None;trainPI=None
# if args.classify:
#     args.nepoch=1000; args.lr=.01
#     train_new(models[0],args,train,test,device)

fout.write('DONE\n')
fout.flush()
if (not args.CONS):
        fout.close()

