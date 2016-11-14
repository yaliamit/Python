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
parms['TRAIN']=True
parms['mult']=1
parms['USE_EXISTING']=False
parms['start']=0
parms['output']='OUTPUT'

parms=manage_OUTPUT.process_args(sys.argv,parms)

if (parms['output_net'] is None):
    parms['output_net']=parms['net']
print 'XXX:',parms['output_net']

nets=[]
output_nets=[]
for m in np.arange(parms['start'],parms['mult'],1):
    nets.append(parms['net']+'_'+str(m))
    output_nets.append(parms['output_net']+'_'+str(m+1))
if (not parms['TRAIN']):
    parms['USE_EXISTING']=True

agg=None
for i,ne in enumerate(nets):

    if (parms['mult']>1 and i>parms['start']):
        parms['USE_EXISTING']=True
    if (parms['USE_EXISTING']):
         # if (os.path.isfile(ne+'.pars')):
         #    fo=open(ne+'.pars','r')
         #    NETPARS=pickle.load(fo)
         print('network',ne+'.txt')
         if (os.path.isfile(ne+'.txt')):
            print('read parameter file',ne+'.txt')
            NETPARS={}
            parse_net_pars.parse_text_file(ne,NETPARS,lname='layers',dump=True)
            # Modifications of parameters come from mod_net_name
            if (parms['mod_net'] is not None and parms['TRAIN']):
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

    NETPARS['mod_net']=parms['mod_net']
    NETPARS['net']=ne
    NETPARS['use_existing']=parms['USE_EXISTING']
    NETPARS['train']=parms['TRAIN']
    NETPARS['seed']=np.int32(np.random.rand()*1000000)
    NETPARS['output_net']=output_nets[i]
    [NETPARS,out]=run_class.main_new(NETPARS)
    if agg is None:
        agg=out[2]
        y=out[-1]
    else:
        agg=agg+out[2]
    acc=np.mean(np.argmax(agg,axis=1)==y)
    agg=None
    print('aggegate accuracy',acc)

if (NETPARS['train']):
    manage_OUTPUT.print_OUTPUT(name=NETPARS['output'])


