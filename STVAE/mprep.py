import torch
from models_opt import STVAE_OPT
from models_mix import STVAE_mix
from models import STVAE
from models_mix_by_class import STVAE_mix_by_class
import numpy as np
import sys
from Conv_data import get_data
import network


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
    else:
        ARGS.append(args)
        strings, ex_file = process_strings(args)
        STRINGS += [strings]
        EX_FILES += [ex_file]

    return ARGS, STRINGS, EX_FILES, SMS

def process_network_line(line, global_drop):
    # break line on the ; each segment is a parameter for the layer of that line
    if line[0]=='#':
        return None
    sss = str.split(line, ';')
    lp = {}
    for ss in sss:
        # Split between parameter name and value
        s = str.split(ss, ':')
        #s1 = str.strip(s[1], ' \n')
        # Process the parameter value
        # A nonlinearity function
        a = ''
        # A number
        s1 = str.strip(s[1], ' \n')
        try:
            a = float(s1)
            if '.' not in s1:
                a = int(s[1])
        # A tuple, a list or the original string
        except ValueError:
            if '(' in s[1]:
                aa = str.split(str.strip(s1, ' ()\n'), ',')
                a = []
                try:
                    int(aa[0])
                    for aaa in aa:
                        a.append(int(aaa))
                    a = tuple(a)
                except ValueError:
                    for aaa in aa:
                        a.append(float(aaa))
                    a = tuple(a)
            elif '[' in s[1]:
                aa = str.split(str.strip(s1, ' []\n'), ',')
                a = []
                for aaa in aa:
                    a.append(aaa)
            elif (s1 == 'None'):
                a = None
            elif (s1 == 'True'):
                a = True
            elif (s1 == 'False'):
                a = False
            else:
                a = s1
        # Add a global drop value to drop layers
        s0 = str.strip(s[0], ' ')
        if (s0 == 'drop' and global_drop is not None):
            lp[s0] = global_drop
        else:
            lp[s0] = a
    return (lp)



def get_network(device, sh,ARGS):

    if hasattr(ARGS,'layers'):
        LP=[]
        for line in ARGS.layers:
            lp = process_network_line(line, None)
            if lp is not None:
                LP += [lp]
        ARGS.layers=LP
    models=[]
    model=network.network(device,sh[1],sh[2],ARGS,ARGS.layers).to(device)
    models+=[model]
    if  hasattr(ARGS,'hid_layers'):
        LP = []
        for line in ARGS.hid_layers:
            lp=process_network_line(line, None)
            if lp is not None:
                LP += [lp]
        ARGS.hid_layers=LP
    return models

def get_models(device, fout, sh,STRINGS,ARGS, locs):

    h = sh[2]
    w = sh[3]

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
    train = [train[0].transpose(0, 3, 1, 2), np.argmax(train[1], axis=1)]
    test = [test[0].transpose(0, 3, 1, 2), np.argmax(test[1], axis=1)]
    if val[0] is not None:
        val = [val[0].transpose(0, 3, 1, 2), np.argmax(val[1], axis=1)]
    if (args.num_test>0):
        ntest=test[0].shape[0]
        ii=np.arange(0, ntest, 1)
        np.random.shuffle(ii)
        test=[test[0][ii[0:args.num_test]], test[1][ii[0:args.num_test]]]
    print('num_train', train[0].shape[0])


    return fout, device, [train, val, test]

