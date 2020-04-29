import torch
from models_opt import STVAE_OPT
from models_mix import STVAE_mix
from models import STVAE
from models_mix_by_class import STVAE_mix_by_class
import numpy as np
import sys
import aux
from Conv_data import get_data
import network
from edges import pre_edges
from torch_edges import Edge
from get_net_text import get_network
import argparse
import os
import mprep
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
    description='')

if 'Linux' in os.uname():
    pyt='/opt/anaconda3_beta/bin/python'
else:
    pyt='python'

args=aux.process_args(parser)


lnti, layers_dict = mprep.get_network(args.layers)

fin= open('_pars/'+args.par_file,'r')
lines = [line.rstrip('\n') for line in fin]
fin.close()




oldn=None
for i,d in enumerate(layers_dict):

    nn=d['name']

    if  'final' in nn or 'input' in nn or 'drop' in nn or (i<len(layers_dict)-1 and 'pool' in layers_dict[i+1]['name']):
       print('skip '+nn)
    else:
        fout = open('_pars/t_par', 'w')
        for l in lines:
            doo=True
            if 'hid' in l:
                doo=False
            if 'dense_final' in l and doo:
                fout.write(l+';parent:['+nn+']\n')
            else:
                fout.write(l+'\n')
        if (args.embedd):
            fout.write('--embedd_layer=' + nn+'\n')
        fout.write('--update_layers\n')
        if ('pool' in nn):
            fout.write(layers_dict[i-1]['name']+'\n')
        else:
            fout.write(nn+'\n')
        fout.write('dense_final')

        fout.close()
        emb='cl'
        if args.embedd:
            emb='emb'
        outn='network_' + nn + '_' + emb
        com=pyt+' main_opt.py @_pars/t_par --model_out='+outn
        if oldn is not None:
            com=com+' --reinit --model='+oldn
        os.system(com)
        oldn=outn


print("helo")






