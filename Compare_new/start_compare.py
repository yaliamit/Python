import parse_net_pars
import run_compare
import pickle
import compare
import manage_OUTPUT
import os
import sys

parms={}
parms['net']='igor2_maxout'
parms['output_net']=None
parms['mod_net']=None
parms['TRAIN']=True
parms['mult']=1
parms['USE_EXISTING']=False
parms['single']=True
parms=manage_OUTPUT.process_args(sys.argv,parms)

if (not parms['TRAIN']):
    parms['USE_EXISTING']=True
if (parms['single']):

    if (parms['USE_EXISTING']):
        net=parms['net']
        if (os.path.isfile(net+'.txt')):
            print('read parameter file',net+'.txt')
            NETPARS={}
            parse_net_pars.parse_text_file(net,NETPARS,lname='layers')
            # Modifications of parameters come from mod_net_name
            if (parms['mod_net'] is not None and parms['TRAIN']):
                parse_net_pars.parse_text_file(parms['mod_net'],NETPARS,lname='INSERT_LAYERS')
            # Continue with last time step of previous run
            if ('eta_current' in NETPARS):
                NETPARS['eta_init']=NETPARS['eta_current']
        else:
            print("Couldn't find network")
            exit()
    else:
        NETPARS={}
        parse_net_pars.parse_text_file(parms['net'],NETPARS,lname='layers')
    NETPARS['net']=parms['net']
    NETPARS['output_net']=parms['output_net']
    NETPARS['train']=parms['TRAIN']
    NETPARS['use_existing']=parms['USE_EXISTING']
    run_compare.main_new(NETPARS)
    fo=open(NETPARS['output_net']+'.pars','w')
    pickle.dump(NETPARS,fo)
    manage_OUTPUT.print_OUTPUT(NETPARS)
else:
    net_names=[]
    #net_names.append('net_trans')
    #net_names.append('net_trans1')
    net_names.append('net_32_2_32_2_256')
    for net_name in net_names:
        fo=open(net_name+'.pars','r')
        NETPARS=pickle.load(fo)
        NETPARS['net_name']=net_name
        NETPARS['train']=False
        NETPARS['num_seqs']=40 # Number of pairs of sequences to create.
        NETPARS['slength']=6 # Length of sequences.
        NETPARS['gr']=False
        NETPARS['seed']=1234985
        NETPARS['corr_shift']=4 # Range of computation of correlation for each original window size.
        compare.run_network_on_all_pairs(NETPARS)