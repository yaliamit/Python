import numpy as np
import torch
import os
import sys
import argparse
import time
import aux
from model_new import CLEAN




os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
    description='Variational Autoencoder with Spatial Transformation'
)


def create_model(args, x_dim, y_dim, lst, train_data, train_text, device, fout):
    model = CLEAN(device, x_dim, y_dim, args).to(device)
    model.lst = lst
    # Run it on a small batch to initialize some modules that need to know dimensions of output
    model.run_epoch(train_data[0:model.bsz], train_text, 0, fout, 'test')

    # Output all parameters
    tot_pars = 0
    for keys, vals in model.state_dict().items():
        fout.write(keys + ',' + str(np.array(vals.shape)) + '\n')
        tot_pars += np.prod(np.array(vals.shape))
    fout.write('tot_pars,' + str(tot_pars) + '\n')
    return model

def train_model(model, args, train_data_shift, test_data_shift, fout):
    for epoch in range(args.nepoch):

        t1 = time.time()
        # If optimizing over shifts and scales for each image
        if (args.OPT and epoch >= args.pre_nepoch):
            # with current network parameters find best scale and shift for each image -> train_data_choice_shift

            with torch.no_grad():
                train_data_choice_shift, rxtr, msmx = model.get_loss_shift(train_data_shift, train_text_shift, epoch, fout,
                                                                     'shift_train')
            # Run an iteration of the network training on the chosen shifts/scales

            for ine in range(args.within_nepoch):
                rtx = model.run_epoch(train_data_choice_shift, train_text, epoch, fout, 'train')

            # Get the results on the test data using the optimal transformation for each image.
            with torch.no_grad():
                model.get_loss_shift(test_data_shift, test_text_shift, epoch, fout,
                                                                    'shift_test')

        # Try training simply on the augmented training set without optimization
        else:
            model.run_epoch(train_data_shift, train_text_shift, epoch, fout, 'train')
            # Then test on original test set.
            model.run_epoch(test_data, test_text, epoch, fout, 'test')

        # fout.write('test: in {:5.3f} seconds\n'.format(time.time()-t3))
        fout.write('epoch: {0} in {1:5.3f} seconds\n'.format(epoch, time.time() - t1))
        fout.flush()


def setup():

    args=aux.process_args(parser)

    use_gpu=0
    if (torch.cuda.is_available()):
        use_gpu = args.gpu
    if (use_gpu and not args.CONS):
        fout=open('OUT.txt','w')
    else:
        args.CONS=True
        fout=sys.stdout

    fout.write(str(args)+'\n')
    args.fout=fout
    fout.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cuda_string="cuda:"+str(use_gpu-1)
    device = torch.device(cuda_string if use_gpu else "cpu")
    fout.write('Device,'+str(device)+'\n')
    fout.write('USE_GPU,'+str(use_gpu)+'\n')

    # Total number of copies per image.
    lst=1
    if (len(args.S)>0 and len(args.T)>0 and len(args.Z)>=0):
        lst=len(args.S)*len(args.T)*(len(args.Z)+1)

    return args, device, lst, fout

args, device, lst, fout=setup()
# Assume data is stored in an hdf5 file, split the data into 80% training and 20% test.
train_data,  train_text, test_data, test_text, aa, x_dim, y_dim = aux.get_data(args, fout)

# Create the shifts and scales for train and test data
train_data_shift, train_text_shift=aux.add_shifts_new(train_data,train_text,args)
fout.write('num train shifted '+str(train_data_shift.shape[0])+'\n')
if (args.OPT):
    test_data_shift, test_text_shift=aux.add_shifts_new(test_data,test_text,args)

model=create_model(args,x_dim,y_dim,lst,train_data,train_text, device, fout)
# Get the model



if (args.run_existing):
    model.load_state_dict(torch.load(args.output_prefix + '_output/' + args.model + '.pt', map_location=device))
else:
    train_model(model,args,train_data_shift,test_data_shift, fout)
# Run one more time on test
if (args.OPT):
    with torch.no_grad():
        test_data_choice_shift,rx,msmx=model.get_loss_shift(test_data_shift, test_text_shift, 0, fout,'test')

    aux.show_shifts(test_data_choice_shift[0:80], test_data[0:80], msmx, rx, model.x_dim, 'shifts_' + str(args.nepoch),args.aa, args.lenc)

else:
    rx=model.run_epoch(test_data, test_text,0,fout, 'test')
# Get resulting labels for each image.
rxx=np.int32(np.array(rx)).ravel()
tt=np.array([args.aa[i] for i in rxx]).reshape(len(test_text),args.lenc)
# Create tif file that pastes the computed labeling below the original image for each test image
aux.create_image(test_data,tt,model.x_dim,'try')

# Store the trained model.
if not os.path.isfile('_output'):
    os.system('mkdir _output')
torch.save(model.state_dict(),'_output/'+args.model+'.pt')
fout.write("DONE\n")
fout.flush()

if (not args.CONS):
    fout.close()