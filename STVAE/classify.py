import torch
from models_opt import STVAE_OPT
from models_mix import STVAE_mix
from models import STVAE
#from models_opt_mix import STVAE_OPT_mix
from models_mix_by_class import STVAE_mix_by_class
#from models_opt_mix_by_class import STVAE_OPT_mix_by_class
import numpy as np
import os
import sys
import argparse
import time


def classify(train,test,image_dim,opt_pre,opt_post,opt_mix,opt_class,device,args,fout,locs):
    run_classify(train, opt_pre, opt_post, opt_mix, opt_class, device, args, fout, locs,'train')
    run_classify(test, opt_pre, opt_post, opt_mix, opt_class, device, args, fout, locs,'test')
    fout.write('DONE\n')

def dens_apply(model,s_mu,s_logvar,lpi,pi,rho):
        n_mix=pi.shape[1]
        s_mu = s_mu.view(-1, n_mix, model.s_dim)
        s_logvar = s_logvar.view(-1, n_mix, model.s_dim)
        sd=torch.exp(s_logvar/2)
        var=sd*sd

        # Sum along last coordinate to get negative log density of each component.
        KD_dens=-0.5 * torch.sum(1 + s_logvar - s_mu ** 2 - var, dim=2)
        KD_disc=lpi-rho+torch.logsumexp(rho,0)
        KD=torch.sum(pi * (KD_dens + KD_disc),dim=1)
        return KD

def run_classify(train,opt_pre,opt_post,opt_mix,opt_class,device,args,fout,locs,type):
    VV=[]
    for cl in range(10):
        t1 = time.time()
        fout.write(str(cl)+'\n')
        fout.flush()
        h=train[0].shape[1]
        w=train[0].shape[2]
        model=locs['STVAE'+opt_mix+opt_class](h, w,  device, args).to(device)
        ex_file = opt_pre + opt_class + args.type + '_' + args.transformation + '_' + str(args.num_hlayers) + '_mx_' + str(args.n_mix) + '_sd_' + str(args.sdim) + '_cl_' +str(cl)
        model.load_state_dict(torch.load(args.output_prefix + '_output/' + ex_file + '.pt', map_location=device))
        V=run_epoch_classify(model,train,args.nti,fout)
        VV+=[V]
        fout.write('classify: {0} in {1:5.3f} seconds\n'.format(cl,time.time()-t1))

    VVV=np.stack(VV,axis=1)
    hy=np.argmin(VVV,axis=1)
    y = np.argmax(train[1], axis=1)
    acc=np.mean(np.equal(hy,y))
    fout.write('====> {} Accuracy {:.4F}\n'.format(type,acc))


def run_epoch_classify(model, train, num_mu_iter,fout):

        mu, logvar, pi = model.initialize_mus(train[0], True)

        model.eval()

        model.setup_id(model.bsz)
        tr = train[0].transpose(0, 3, 1, 2)
        y = np.argmax(train[1],axis=1)
        V=[]
        for j in np.arange(0, len(y), model.bsz):
            if (np.mod(j,2000)==0):
                fout.write(str(j)+'\n')
                fout.flush()
            data = torch.from_numpy(tr[j:j + model.bsz]).float().to(model.dv)
            if (model.opt):
                model.update_s(mu[j:j + model.bsz, :], logvar[j:j + model.bsz, :], pi[j:j + model.bsz], model.mu_lr[0])
                for it in range(num_mu_iter):
                    model.compute_loss_and_grad(data, 'test', model.optimizer_s, opt='mu')
                s_mu=model.mu
                s_var=model.logvar
                ss_mu = model.mu.view(-1, model.n_mix, model.s_dim).transpose(0,1)
                tpi=torch.softmax(model.pi,dim=1)
            else:
                s_mu, s_var, tpi = model.encoder_mix(data.view(-1, model.x_dim))
                ss_mu = s_mu.view(-1, model.n_mix, model.s_dim).transpose(0,1)
            with torch.no_grad():
                recon_batch = model.decoder_and_trans(ss_mu)
                lpi=torch.log(tpi)
                b = model.mixed_loss_pre(recon_batch, data)
                KD=dens_apply(model,s_mu,s_var,lpi,tpi,model.rho)
                recloss = torch.sum(tpi*b,dim=1)
                vy=recloss+KD
                #vy, by= torch.min(b,1)
                V+=[vy.detach().cpu().numpy()]


        V=np.concatenate(V)

        return V



