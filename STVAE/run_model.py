import torch
import network
import numpy as np
import pylab as py
import mprep
import os
import argparse
import aux
import eagerpy as ep
import foolbox
from foolbox import accuracy, samples
import foolbox.attacks as fa
from Conv_data import get_data
from torch_edges import Edge
from torch import nn

class fb_network(nn.Module):
    def __init__(self,args,device):
        super(fb_network, self).__init__()
        self.args=args
        self.dv=device
        sh = [0,24,32,32]
        self.lnti, self.layers_dict = mprep.get_network(args.layers, sh=sh)
        self.model=network.network(self.dv, self.args, self.layers_dict, self.lnti).to(self.dv)
        sm = torch.load('_output/network.pt', map_location='cpu')
        temp = torch.zeros(1, sh[1], sh[2], sh[3]).to(self.dv)
        bb = self.model.forward(temp)
        self.model.load_state_dict(sm['model.state.dict'])
        self.ed=Edge(self.dv,dtr=.03).to(self.dv)

    def forward(self,input):
        edges = self.ed(input)
        out = self.model(edges)
        return out

def run_data(args):

    f_model=fb_network(args,device).to(device)
    f_model.eval()
    fmodel = foolbox.models.PyTorchModel(f_model, bounds=(0, 1))

    PARS = {}
    PARS['data_set'] = args.dataset
    PARS['num_train'] = 10
    PARS['nval'] = args.nval

    train, val, test, image_dim = get_data(PARS)

    images=torch.from_numpy(train[0].transpose(0,3,1,2))
    labels=torch.from_numpy(np.argmax(train[1], axis=1))
    #images, labels = ep.astensors(*samples(fmodel, dataset="cifar10", batchsize=1))
    #images, labels = samples(fmodel, dataset="cifar10", batchsize=16)


   #ed=Edge(device,dtr=.03)
    #edges=ep.astensor(ed(images))


    print(accuracy(fmodel,images,labels))

    # epsilons = [
    #     0.0,
    #     0.0005,
    #     0.001,
    #     0.0015,
    #     0.002,
    #     0.003,
    #     0.005,
    #     0.01,
    #     0.02,
    #     0.03,
    #     0.1,
    #     0.3,
    #     0.5,
    #     1.0,
    # ]

    epsilons = [
        1.0,
    ]

    attack=fa.BoundaryAttack() #(LinfPGD()

    advs, _, success = attack(fmodel, images, labels, epsilons=epsilons)
    assert success.shape == (len(epsilons), len(images))
    success_ = success.detach().numpy()
    assert success_.dtype == np.bool

    ad=advs[0]
    print(success_)
    orig_class=torch.max(f_model(images),dim=1)
    adv_class=torch.max(f_model(ad),dim=1)
    adn=ad.numpy()
    both=np.concatenate((train[0].transpose(0,3,1,2)),adn,axis=0)
    bb=aux.create_img(both,3,32,32,len(adn),2)
    py.imshow(bb)
    ss = ' '.join([str(elem) for elem in orig_class])
    py.text(-3, -3, ss)
    ss = ' '.join([str(elem) for elem in adv_class])
    py.text(32, -3, ss)
    py.savefig('adv')



    # print(attack)
    # print("  ", 1.0 - success_.mean(axis=-1).round(2))
    #
    # return out




os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
        description='Variational Autoencoder with Spatial Transformation')


args=aux.process_args(parser)

use_gpu = args.gpu and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpu - 1) if use_gpu else "cpu")
print(device)


PARS = {}
PARS['data_set'] = 'cifar10'
PARS['num_train'] = 100
PARS['nval'] = 0

print(foolbox.__version__)
#train, val, test, image_dim = get_data(PARS)

#images=ep.astensor(torch.from_numpy(train[0].transpose(0,3,1,2)).float())
#labels=ep.astensor(torch.from_numpy(np.argmax(train[1], axis=1)).long())


run_data(args)