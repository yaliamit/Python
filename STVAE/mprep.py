import torch
from models_opt import STVAE_OPT
from models_mix import STVAE_mix
from models import STVAE
from models_mix_by_class import STVAE_mix_by_class
import numpy as np
import sys
from Conv_data import get_data
import network
from edges import pre_edges
from torch_edges import Edge
from get_net_text import get_network

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
    elif (args.rerun):
        strings, ex_file = process_strings(args)
        sm = torch.load('_output/' + ex_file + '.pt', map_location='cpu')
        SMS += [sm]
        if ('args' in sm):
            args = sm['args']
        ARGS+=[args]
        STRINGS += [strings]
        EX_FILES += [ex_file]

    else:
        ARGS.append(args)
        strings, ex_file = process_strings(args)
        STRINGS += [strings]
        EX_FILES += [ex_file]

    return ARGS, STRINGS, EX_FILES, SMS





def get_models(device, fout, sh,STRINGS,ARGS, locs):



    models = []
    for strings, args in zip(STRINGS, ARGS):
        model=make_model(strings,args,locs, sh[1:], device, fout)
        models += [model]

    return models


def  make_model(strings,args,locs, sh, device, fout):
        model = locs['STVAE' + strings['opt_mix'] + strings['opt_class']](sh, device, args).to(device)
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

    return fout, device

def get_data_pre(args,dataset):


    PARS = {}
    PARS['data_set'] = dataset
    PARS['num_train'] = args.num_train // args.mb_size * args.mb_size
    PARS['nval'] = args.nval
    if args.cl is not None:
        PARS['one_class'] = args.cl

    train, val, test, image_dim = get_data(PARS)
    if (False): #args.edges):
        train=[pre_edges(train[0],dtr=args.edge_dtr).transpose(0,3,1,2),np.argmax(train[1], axis=1)]
        test=[pre_edges(test[0],dtr=args.edge_dtr).transpose(0,3,1,2),np.argmax(test[1], axis=1)]
        if val[0] is not None:
            val = [pre_edges(val[0],dtr=args.edge_dtr).transpose(0, 3, 1, 2), np.argmax(val[1], axis=1)]
    else:
        train = [train[0].transpose(0, 3, 1, 2), np.argmax(train[1], axis=1)]
        test = [test[0].transpose(0, 3, 1, 2), np.argmax(test[1], axis=1)]
        if val[0] is not None:
            val = [val[0].transpose(0, 3, 1, 2), np.argmax(val[1], axis=1)]
    if args.edges:
        ed = Edge(device, dtr=.03).to(device)
        edges=[]
        jump=10000
        for j in np.arange(0,train[0].shape[0],jump):
            tr=torch.from_numpy(train[0][j:j+jump]).to(device)
            edges+=[ed(tr).cpu().numpy()]
        train=[np.concatenate(edges,axis=0),train[1]]
        edges_te=[]
        for j in np.arange(0,test[0].shape[0],jump):
            tr=torch.from_numpy(test[0][j:j+jump]).to(device)
            edges_te+=[ed(tr).cpu().numpy()]
        test=[np.concatenate(edges_te,axis=0),test[1]]
        if val[0] is not None:
            edges_va = []
            for j in np.arange(0, test[0].shape[0], jump):
                tr = torch.from_numpy(val[0][j:j + jump]).to(device)
                edges_va += [ed(tr).cpu().numpy()]
            val = [np.concatenate(edges_va,axis=0), val[1]]
    if (args.num_test>0):
        ntest=test[0].shape[0]
        ii=np.arange(0, ntest, 1)
        np.random.shuffle(ii)
        test=[test[0][ii[0:args.num_test]], test[1][ii[0:args.num_test]]]
    print('num_train', train[0].shape[0])


    return [train, val, test]

