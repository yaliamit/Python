import torch
import torchvision
from torchvision import transforms as TR
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

c10=torchvision.datasets.SVHN("SVHN", split="train",
        transform=TR.ToTensor(), target_transform=None, download=True)


data_loader = torch.utils.data.DataLoader(c10,batch_size=1000)

OUT=[]

INP=[]
L=[]
for d,l in data_loader:
        inp, lab = d, l
        INP+=[inp]
        L+=[l]

INP=torch.cat(INP)
L=torch.cat(L)


with h5py.File('SVHN_train.hdf5','w') as f:
    dset1=f.create_dataset("data",data=numpy.uint8(255*INP.numpy()))
    dset2=f.create_dataset("labels",data=numpy.uint8(L.numpy()))

import numpy as np
with h5py.File('SVHN_train.hdf5', 'r') as f:
    data=f['data']
    print(data.shape)
    labels=f['labels']
    print(labels.shape)
    dat=np.uint8(data)
print('{0:5.3f}s'.format(time.time() - t1))




print("hello")

