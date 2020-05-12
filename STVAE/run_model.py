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
from models_transforms import Edge
from torch import nn
from PIL import Image, ImageDraw, ImageFont, ImageOps

class fb_network(nn.Module):
    def __init__(self,args,sh,device):
        super(fb_network, self).__init__()
        self.args=args
        self.dv=device
        self.edges=args.edges
        #sh = [0,24,32,32]
        nc=sh[2]
        if (args.edges):
            nc*=8
        sh=[0,nc,sh[0],sh[1]]
        self.lnti, self.layers_dict = mprep.get_network(args.layers, nf=nc)
        self.model=network.network(self.dv, self.args, self.layers_dict, self.lnti).to(self.dv)
        sm = torch.load('_output/'+args.model[0], map_location='cpu')
        temp = torch.zeros(1, sh[1], sh[2], sh[3]).to(self.dv)
        bb = self.model.forward(temp)
        self.model.load_state_dict(sm['model.state.dict'])
        self.ed=Edge(self.dv,dtr=.03).to(self.dv)

    def forward(self,input):
        if (self.edges):
            input = self.ed(input)
        out = self.model(input)
        return out

categs=['airp','auto','bird','cat','deer','dog','frog','horse','ship','truck']

def save_image(bb,orig_class,adv_class,h,m):
    xdim=bb.shape[0]
    ydim=bb.shape[1]
    img = Image.new('RGB',(ydim+40,xdim+40), color=(255,255,255) )
    imga=Image.fromarray(np.uint8(255.*bb))
    img.paste(imga,(20,20))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("Arial.ttf", 10)
    t=20
    for o in orig_class:
        draw.text((t,2), categs[o], 0, font=font)
        t+=h
    xd=20
    for aa in adv_class:
        t=20
        for a in aa:
            draw.text((t,xd+h),categs[a],0,font=font)
            t+=h
        xd+=(h+m)

    img.save("adv.tif", compression="tiff_deflate", save_all=True)

    print("Saved the sampled images")

def run_data(args):

    PARS = {}
    PARS['data_set'] = args.dataset
    PARS['num_train'] = args.num_train
    PARS['nval'] = args.nval
    train, val, test, image_dim = get_data(PARS)
    ii=np.array(range(test[0].shape[0]))
    np.random.shuffle(ii)
    test=[test[0][ii][0:args.num_train],test[1][ii][0:args.num_train]]

    f_model=fb_network(args,test[0][0].shape,device).to(device)
    f_model.eval()
    fmodel = foolbox.models.PyTorchModel(f_model, bounds=(0, 1))




    images=torch.from_numpy(test[0].transpose(0,3,1,2)).to(device)
    labels=torch.from_numpy(np.argmax(test[1], axis=1)).to(device)
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
        1.,
        .1,
        .01,
    ]


    st=args.nti
    print('steps',st)
    attack=fa.BoundaryAttack(steps=st) #(LinfPGD()
    #attack=fa.LinfPGD(steps=st)
    advs, wh, success = attack(fmodel, images, labels, epsilons=epsilons)
    assert success.shape == (len(epsilons), len(images))
    success_ = success.detach().cpu().numpy()
    assert success_.dtype == np.bool


    print(success_)
    orig_class=(torch.argmax(f_model(images),dim=1)).cpu().numpy()
    adv_class=[]
    for a in advs:
        adv_class+=[(torch.argmax(f_model(a),dim=1)).cpu().numpy()]
    ll=len(epsilons)
    la=len(advs[0])
    cc=np.zeros((la*(1+ll),3,32,32))
    cc[0:((1+ll)*la):(1+ll)]=test[0].transpose(0,3,1,2)
    for t in range(1,ll+1):
        cc[t:((1+ll)*la):(1+ll)]=advs[t-1].cpu().numpy()
    #both=np.concatenate((train[0].transpose(0,3,1,2),adn),axis=0)
    bb=aux.create_img(cc,3,32,32,la,ll+1,15)
    save_image(bb,orig_class,adv_class,32,15)
    # py.imshow(bb)
    # py.axis('off')
    # ss = ' '.join([str(elem) for elem in orig_class])
    # py.text(-3, -3, ss)
    # ss = ' '.join([str(elem) for elem in adv_class])
    # py.text(-3, 73, ss)
    # py.savefig('adv')



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