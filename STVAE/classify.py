import torch
from models_opt import STVAE_OPT
from models_mix_try import STVAE_mix
from models import STVAE
from models_opt_mix import STVAE_OPT_mix
from models_mix_by_class import STVAE_mix_by_class
from models_opt_mix_by_class import STVAE_OPT_mix_by_class
import numpy as np
import os
import sys
import argparse
import time


def classify(train,test,image_dim,opt_pre,opt_post,opt_mix,opt_class,device,args,fout,locs):
    run_classify(train, opt_pre, opt_post, opt_mix, opt_class, device, args, fout, locs,'train')
    run_classify(test, opt_pre, opt_post, opt_mix, opt_class, device, args, fout, locs,'test')


def run_classify(train,opt_pre,opt_post,opt_mix,opt_class,device,args,fout,locs,type):
    VV=[]
    for cl in range(10):
        h=train[0].shape[1]
        w=train[0].shape[2]
        model=locs['STVAE'+opt_post+opt_mix+opt_class](h, w,  device, args).to(device)
        ex_file = opt_pre + opt_class + args.type + '_' + args.transformation + '_' + str(args.num_hlayers) + '_mx_' + str(args.n_mix) + '_sd_' + str(args.sdim) + '_cl_' +str(cl)
        model.load_state_dict(torch.load(args.output_prefix + '_output/' + ex_file + '.pt', map_location=device))
        V=run_epoch_classify(model,train,args.nti)
        VV+=[V]
    VVV=np.stack(VV,axis=1)
    hy=np.argmin(VVV,axis=1)
    y = np.argmax(train[1], axis=1)
    acc=np.mean(np.equal(hy,y))
    fout.write('====> {} Accuracy {:.4F}\n'.format(type,acc))


def run_epoch_classify(model, train, num_mu_iter=10):

        mu, logvar, pi = model.initialize_mus(train[0], True)

        model.eval()

        model.setup_id(model.bsz)
        tr = train[0].transpose(0, 3, 1, 2)
        y = np.argmax(train[1],axis=1)
        V=[]
        for j in np.arange(0, len(y), model.bsz):
            print(j)
            data = torch.from_numpy(tr[j:j + model.bsz]).float().to(model.dv)
            if ('OPT' in model.__class__.__name__):
                model.update_s(mu[j:j + model.bsz, :], logvar[j:j + model.bsz, :], pi[j:j + model.bsz], model.mu_lr[0])
                for it in range(num_mu_iter):
                    model.compute_loss_and_grad(data, 'test', model.optimizer_s, opt='mu')
                s_mu = model.mu.view(-1, model.n_mix, model.s_dim).transpose(0,1)
                pi=model.pi
            else:
                s_mu, s_var, pi = model.encoder_mix(data.view(-1, model.x_dim))
                s_mu = s_mu.view(-1, model.n_mix, model.s_dim).transpose(0,1)

            with torch.no_grad():
                recon_batch = model.decoder_and_trans(s_mu)
                b = model.mixed_loss_pre(recon_batch, data, pi.shape[1])
                vy, by= torch.min(b,1)
                V+=[vy.detach().cpu().numpy()]


        V=np.concatenate(V)

        return V



