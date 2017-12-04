import os
import sys

import numpy as np

import manage_OUTPUT
import parse_net_pars
from source import run_class

parms={}
parms['net']='igor2_maxout'
parms['output_net']=None
parms['mod_net']=None
parms['train']=True
parms['mult']=1
parms['use_existing']=False
parms['start']=0


parms=manage_OUTPUT.process_args(sys.argv,parms)
parms['start']=np.int(parms['start'])
parms['mult']=np.int(parms['mult'])
if (parms['output_net'] is None):
    parms['output_net']=parms['net']
print 'XXX:',parms['output_net']

nets=[]
output_nets=[]
if (parms['mult']>parms['start']):
    for m in np.arange(parms['start'],parms['mult'],1):
        nets.append(parms['net']+'_'+str(m))
        output_nets.append(parms['output_net']+'_'+str(m+1))
# start=1 and mult=1 output is net_0
elif (parms['mult'] == 1):
    nets.append(parms['net'])
    output_nets.append(parms['output_net']+'_0')
elif (parms['mult'] > 1):
    nets.append(parms['net'])
    output_nets.append(parms['output_net']+'_0_'+str(parms['mult']-1))
else:
    print 'Don\'t know which net to use'
    exit(0)
if (not parms['train']):
    parms['use_existing']=True

agg=None
for i,ne in enumerate(nets):

    if (parms['mult']>1 and parms['mult']-parms['start']>1):
        #if (not parms['use_existing']):
            parms['seed']=np.random.randint(0,200000)
            if (i==0):
                agg=[]
    if (parms['use_existing']):
         # if (os.path.isfile(ne+'.pars')):
         #    fo=open(ne+'.pars','r')
         #    NETPARS=pickle.load(fo)
         print('network',ne+'.txt')
         if (os.path.isfile(ne+'.txt')):
            print('read parameter file',ne+'.txt')
            NETPARS={}
            parse_net_pars.parse_text_file(ne,NETPARS,lname='layers',dump=False)
            if ("write_sparse" in NETPARS):
                    del NETPARS["write_sparse"]
            # Modifications of parameters come from mod_net_name
            if parms['mod_net'] is not None:
             if (parms['start']==parms['mult']): # and parms['train']):
                parse_net_pars.parse_text_file(parms['mod_net'],NETPARS,lname='INSERT_LAYERS', dump=False)
                if ("write_sparse" in NETPARS):
                    del NETPARS["write_sparse"]
             else:
                 # A sequence of modifications to the basic parameters
                f=open(parms['mod_net']+'.txt','r')
                t=0
                for line in f:
                    if (line[0]=='#'):
                        continue
                    gd=None
                    lp=parse_net_pars.process_network_line(line,gd)
                    if (t==i):
                        break
                    t+=1
                f.close()
                print('LP',lp)
                NETPARS[lp['dict']]=lp
                del NETPARS[lp['dict']]['dict']
            # Continue with last time step of previous run
            if ('eta_current' in NETPARS):
                 NETPARS['eta_init']=NETPARS['eta_current']
         else:
            print("Couldn't find network")
            exit()
    else:
        NETPARS={}
        parse_net_pars.parse_text_file(parms['net'],NETPARS,lname='layers', dump=False)
        if ("write_sparse" in NETPARS):
                del NETPARS["write_sparse"]
        if (i==0):
            np.random.seed(NETPARS['seed'])


    for key, value in parms.iteritems():
        if (type(parms[key]) is dict):
            if (key not in NETPARS):
                NETPARS[key]={}
            for skey in parms[key]:
                NETPARS[key][skey]=parms[key][skey]
        else:
            NETPARS[key]=parms[key]

    NETPARS['net']=ne
    NETPARS['output_net']=output_nets[i]
    LAYERS=NETPARS['layers']
    if (len(LAYERS) and 'concat' not in LAYERS[-1]):
            if ('hinge' not in NETPARS or not NETPARS['hinge']):
                LAYERS[-1]['non_linearity']='softmax'
            else:
                # for l in LAYERS:
                #     if ('non_linearity' in l):
                #         l['non_linearity']=lasagne.nonlinearities.sigmoid
                LAYERS[-1]['non_linearity']='linear'
    NETPARS['layers']=LAYERS
    # NETPARS['output']=parms['output']
    # # Command line overrides.
    # if ('num_train' in parms):
    #     NETPARS['num_train']=parms['num_train']
    parse_net_pars.dump_pars(NETPARS)

            #print '\n'
    [NETPARS,out]= run_class.main_new(NETPARS)

    many=False

    if agg is None:
        agg=out[2]
        y=out[-1]
        acc=np.mean(np.argmax(agg,axis=1)==y)
    else:
        CL=None
        if ('Classes' in NETPARS):
            CL=np.array(NETPARS['Classes'])
        many=True
        if (len(agg)==0):
            ag=np.argmax(out[2],axis=1) #out[2]
            if (CL is not None):
                ag=CL[ag]
            agg=np.zeros(out[2].shape)
            agg[np.array(range(out[2].shape[0])),ag]=1
        else:
            ag=np.argmax(out[2],axis=1)
            if (CL is not None):
                ag=CL[ag]
            aggt=np.zeros(out[2].shape)
            aggt[np.array(range(out[2].shape[0])),ag]=1
            agg=agg+aggt
        y=out[-1]
    acc=np.mean(np.argmax(agg,axis=1)==y)
    #agg=None
    print('step',i,'aggegate accuracy',acc)

print('NNN:',NETPARS['output_net'])
print('DONE')
if (NETPARS['train']):
    manage_OUTPUT.print_OUTPUT(name=NETPARS['output'])


