import parse_net_pars
import run_class
import os
import numpy as np
import sys
import manage_OUTPUT
parms={}
parms['net']='igor2_maxout'
parms['output_net']=None
parms['mod_net']=None
parms['train']=True
parms['mult']=1
parms['use_existing']=False
parms['start']=0
parms['output']='OUTPUT'

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
elif (parms['mult'] == 1):
    nets.append(parms['net'])
    output_nets.append(parms['output_net']+'_0')
else:
    print 'Don\'t know which net to use'
    exit(0)
if (not parms['train']):
    parms['use_existing']=True

agg=None
for i,ne in enumerate(nets):

    if (parms['mult']>1 and i>parms['start']):
        parms['use_existing']=True
    if (parms['use_existing']):
         # if (os.path.isfile(ne+'.pars')):
         #    fo=open(ne+'.pars','r')
         #    NETPARS=pickle.load(fo)
         print('network',ne+'.txt')
         if (os.path.isfile(ne+'.txt')):
            print('read parameter file',ne+'.txt')
            NETPARS={}
            parse_net_pars.parse_text_file(ne,NETPARS,lname='layers',dump=True)
            # Modifications of parameters come from mod_net_name
            if (parms['mod_net'] is not None): # and parms['train']):
             if (parms['mult']==1):
                parse_net_pars.parse_text_file(parms['mod_net'],NETPARS,lname='INSERT_LAYERS', dump=True)
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
        parse_net_pars.parse_text_file(parms['net'],NETPARS,lname='layers', dump=True)
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
    # NETPARS['output']=parms['output']
    # # Command line overrides.
    # if ('num_train' in parms):
    #     NETPARS['num_train']=parms['num_train']
    [NETPARS,out]=run_class.main_new(NETPARS)
    if agg is None:
        agg=out[2]
        y=out[-1]
    else:
        agg=agg+out[2]
    acc=np.mean(np.argmax(agg,axis=1)==y)
    agg=None
    print('aggegate accuracy',acc)

print('NNN:',NETPARS['output_net'])
print('DONE')
if (NETPARS['train']):
    manage_OUTPUT.print_OUTPUT(name=NETPARS['output'])


