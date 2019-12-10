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
from classify import classify


def get_models(args):
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

def  make_model(strings,args,locs, h, w, device, fout):
        model = locs['STVAE' + strings['opt_mix'] + strings['opt_class']](h, w, device, args).to(device)
        tot_pars = 0
        for keys, vals in model.state_dict().items():
            fout.write(keys + ',' + str(np.array(vals.shape)) + '\n')
            tot_pars += np.prod(np.array(vals.shape))
        fout.write('tot_pars,' + str(tot_pars) + '\n')
        return model

def setups(ARGS, EX_FILES , STRINGS, locs):
    torch.manual_seed(ARGS[0].seed)
    np.random.seed(ARGS[0].seed)
    use_gpu = ARGS[0].gpu and torch.cuda.is_available()
    if (use_gpu and not ARGS[0].CONS):
        fout = open('_OUTPUTS/OUT_' + EX_FILES[0] + '.txt', 'w')
    else:
        ARGS[0].CONS = True
        fout = sys.stdout


    device = torch.device("cuda:" + str(ARGS[0].gpu - 1) if use_gpu else "cpu")
    fout.write('Device,' + str(device) + '\n')
    fout.write('USE_GPU,' + str(use_gpu) + '\n')

    args = ARGS[0]
    PARS = {}
    PARS['data_set'] = ARGS[0].dataset
    PARS['num_train'] = ARGS[0].num_train // ARGS[0].mb_size * ARGS[0].mb_size
    PARS['nval'] = ARGS[0].nval
    if args.cl is not None:
        PARS['one_class'] = ARGS[0].cl

    train, val, test, image_dim = get_data(PARS)
    print('num_train', train[0].shape[0])

    h = train[0].shape[1]
    w = train[0].shape[2]

    models = []
    for strings, args in zip(STRINGS, ARGS):
        model=make_model(strings,args,locs, h, w, device, fout)
        models += [model]

    return fout, device, [train, val, test], image_dim, models

def test_models(ARGS, SMS, test, fout):
    iid = None;
    len_test = len(test[0]);
    ACC = [];
    CL_RATE = [];
    ls = len(SMS)
    CF = [conf] + list(np.zeros(ls - 1))
    # Combine output from a number of existing models. Only hard ones move to next model?
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

def train_model(model, args, ex_file, DATA, fout):

    train=DATA[0]; val=DATA[1]; test=DATA[2]
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
    aux.make_images(train, model, ex_file, args)
    if (args.n_class):
        model.run_epoch_classify(train, 'train', fout=fout, num_mu_iter=args.nti)
        model.run_epoch_classify(test, 'test', fout=fout, num_mu_iter=args.nti)
    elif args.cl is None:
        model.run_epoch(test, 0, args.nti, testMU, testLOGVAR, testPI, d_type='test', fout=fout)


def process_strings(args):
    strings={'opt_pre':'', 'mm_pre':'', 'opt_post':'', 'opt_mix':'', 'opt_class':'', 'cll':''}
    if (args.OPT):
        strings['opt_pre']='OPT_'
        strings['opt_post']='_OPT'
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

sample=args.sample
classify=args.classify
reinit=args.reinit
run_existing=args.run_existing
conf=args.conf

ARGS, STRINGS, EX_FILES, SMS = get_models(args)

fout, device, DATA, image_dim, models = setups(ARGS, EX_FILES, STRINGS, locals())

if reinit:
    models[0].load_state_dict(SMS[0]['model.state.dict'])
    model_new=make_model(STRINGS[0],args,locals(), DATA[0][0].shape[1],DATA[0][0].shape[2], device, fout)
    model_new.conv.conv.weight.data=models[0].conv.conv.weight.data
    models=[model_new]
    ARGS=[args]
    strings, ex_file = process_strings(args)
    EX_FILE=[ex_file]

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
    train_model(models[0], ARGS[0], EX_FILES[0], DATA, fout)


# trainMU=None;trainLOGVAR=None;trainPI=None
# if args.classify:
#     args.nepoch=1000; args.lr=.01
#     train_new(models[0],args,train,test,device)

fout.write('DONE\n')
fout.flush()
if (not args.CONS):
        fout.close()

